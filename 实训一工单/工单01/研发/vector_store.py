# -*- coding: utf-8 -*-
"""
向量存储模块
支持 FAISS（语义）+ BM25（关键词）混合检索
针对中文财务文档优化
"""

import os
import re
import time
import math
from collections import Counter
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    VECTOR_STORE_PATH, TOP_K_RESULTS, BATCH_SIZE,
    EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
    HYBRID_ALPHA,
)

# ── 中文同义词扩展表 ──

_SYNONYM_MAP = {
    # 领域/客户类 —— 关键一对一同义词替换
    "军用": "军用 国防",
    "国防": "国防 军用",
    "军工": "军工 国防",
    "军方": "军方 国防",
    # 财政/经营类
    "收入": "收入 营业收入",
    "营收": "营收 收入",
    "利润": "利润 净利润",
    # 主营业务
    "主营": "主营 主营业务",
    # 时间
    "年度": "年度 年",
}


def expand_query(query: str) -> str:
    """
    查询扩展：用关键同义词补充原查询，提升 BM25 关键词召回率

    例: "军用领域收入" → "军用 国防 领域收入 营业收入"
    保持扩展适度，避免稀释精准匹配的分值
    """
    expanded = query
    for word, replacement in _SYNONYM_MAP.items():
        if word in query:
            expanded = expanded.replace(word, replacement)
    # 去重
    tokens = expanded.split()
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " ".join(unique)


# ── 中文分词器（轻量，无需 jieba）──

def tokenize(text: str) -> List[str]:
    """
    中文分词：保留中文词、英文单词、数字
    示例: "武汉兴图新科2023年军用收入" → ["武汉", "兴图新科", "2023", "年", "军用", "收入"]
    """
    # 英文单词 + 数字
    tokens = re.findall(r'[a-zA-Z]+|\d+(?:\.\d+)?', text)
    # 中文：逐字符（因为中文词之间无空格）
    # 但连续中文字符作为一个词更好（"收入" vs "收"+"入"）
    cn_words = re.findall(r'[\u4e00-\u9fff]+', text)
    for w in cn_words:
        # 对长中文词也拆成单个字，保留双字词匹配
        if len(w) <= 4:
            tokens.append(w)
        else:
            # 长词拆成重叠双字词
            tokens.extend([w[i:i+2] for i in range(len(w)-1)])
            # 也保留整体
            tokens.append(w)
    return [t.lower() for t in tokens if t]


class BM25Index:
    """轻量 BM25 索引（Okapi BM25）—— 预分词 + 倒排索引优化版"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1          # 饱和度参数
        self.b = b            # 长度归一化参数
        self.doc_freqs = {}   # {term: set_of_doc_ids}
        self.doc_lengths = []  # [len1, len2, ...]
        self.doc_tokens = []   # [[tok1, tok2, ...], ...]  预分词结果
        self.doc_ids = []      # 有序 doc_id 列表
        self.avgdl = 0.0
        self.N = 0
        # 倒排索引: {term: [(doc_idx, tf), ...]}  — 搜索时只处理含该 term 的文档
        self.inverted = {}

    def build(self, doc_ids: List[int], texts: List[str]):
        """构建 BM25 索引（预分词 + 倒排索引）"""
        start = time.time()
        self.doc_ids = list(doc_ids)
        self.N = len(doc_ids)

        self.doc_tokens = []
        self.doc_lengths = []
        total_len = 0
        self.inverted = {}
        self.doc_freqs = {}

        for d_id, text in zip(doc_ids, texts):
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_len += length

            # 统计 term 频率 & 构建倒排
            seen = set()
            term_counts = {}
            for tok in tokens:
                term_counts[tok] = term_counts.get(tok, 0) + 1
                if tok not in seen:
                    self.doc_freqs.setdefault(tok, set()).add(d_id)
                    seen.add(tok)

            for term, tf in term_counts.items():
                self.inverted.setdefault(term, []).append(
                    (len(self.doc_tokens) - 1, tf)
                )

        self.avgdl = total_len / self.N if self.N > 0 else 0
        elapsed = time.time() - start
        if elapsed > 0.1:
            print(f"  [BM25] 索引构建完成, {self.N} 个文档, "
                  f"耗时: {elapsed:.2f}s")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        BM25 检索（倒排索引加速）
        
        利用预构建的倒排索引，只对含查询词的文档计算分数，
        避免全库遍历和重复分词。
        """
        if self.N == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        unique_terms = set(query_tokens)
        scores = {}

        # 使用倒排索引：只处理含查询词的文档
        for term in unique_terms:
            df = len(self.doc_freqs.get(term, set()))
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, tf in self.inverted.get(term, []):
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )
                score = idf * numerator / denominator
                doc_id = self.doc_ids[doc_idx]
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # 排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = [
            {"chunk_id": d_id, "score": s, "rank": i + 1}
            for i, (d_id, s) in enumerate(ranked[:top_k])
        ]
        return results

    def clear(self):
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.doc_tokens.clear()
        self.doc_ids.clear()
        self.inverted.clear()
        self.avgdl = 0.0
        self.N = 0


class VectorStore:
    """向量存储与混合检索器（FAISS + BM25）"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model_path = EMBEDDING_MODEL_PATH
        self.dimension = EMBEDDING_DIMENSION
        self.model = SentenceTransformer(self.model_path)
        # 嵌入模型预热：第一次 encode() 包含 ONNX/PyTorch 初始化，
        # 加载模型到内存并做一次 dummy encode，避免首次查询慢 0.5s+
        _warmup_start = time.time()
        self.model.encode(["预热"], show_progress_bar=False,
                          normalize_embeddings=True)
        _warmup_elapsed = time.time() - _warmup_start
        self.index = None          # FAISS IndexIDMap
        self.bm25 = BM25Index()    # BM25 关键词索引
        self.total_vectors = 0
        print(f"[向量存储] 嵌入模型: {self.model_name}, 维度: {self.dimension}"
              f" (预热: {_warmup_elapsed:.3f}s)")
        print(f"[向量存储] 混合检索: FAISS(α={HYBRID_ALPHA}) + BM25(β={1-HYBRID_ALPHA:.1f})")

    # ---- 索引管理 ----

    def _create_index(self):
        """创建 FAISS IndexIDMap（余弦相似度 = InnerProduct）"""
        base = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base)
        self.total_vectors = 0

    def build_index(self, ids: List[int], texts: List[str]):
        """
        构建 FAISS + BM25 双索引
        """
        start = time.time()
        count = len(ids)
        print(f"[向量存储] 构建索引, {count} 个文本块")

        # 1. FAISS 语义索引
        embeddings = self.embed_texts(texts)
        ids_arr = np.array(ids, dtype=np.int64)
        self._create_index()
        self.index.add_with_ids(embeddings, ids_arr)
        self.total_vectors = count

        # 2. BM25 关键词索引
        self.bm25.build(ids, texts)

        elapsed = time.time() - start
        print(f"  [向量存储] 索引构建完成, {count} 个向量+BM25, "
              f"耗时: {elapsed:.2f}s")

    # ---- 混合检索 ----

    def search(self, query: str, top_k: int = TOP_K_RESULTS
               ) -> List[Dict[str, Any]]:
        """
        混合搜索：FAISS 语义（alpha权重） + BM25 关键词（1-alpha权重）
        两路分数分别归一化后加权融合
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("向量索引为空，请先构建索引")

        # 1. FAISS 语义检索（取 2 倍候选，便于融合排序）
        faiss_k = top_k * 2
        query_vec = self.embed_texts([query])
        distances, labels = self.index.search(query_vec, faiss_k)

        faiss_results = {}
        for i in range(len(labels[0])):
            label = labels[0][i]
            if label < 0:
                continue
            faiss_results[int(label)] = float(distances[0][i])

        # 2. BM25 关键词检索（使用扩展后的查询）
        expanded_query = expand_query(query)
        bm25_results_raw = self.bm25.search(expanded_query, top_k=faiss_k)
        bm25_results = {r["chunk_id"]: r["score"] for r in bm25_results_raw}

        # 3. 分数归一化（Min-Max 到 [0, 1]）
        all_ids = set(faiss_results.keys()) | set(bm25_results.keys())

        def _minmax(scores_dict):
            if not scores_dict:
                return {}
            vals = list(scores_dict.values())
            vmin, vmax = min(vals), max(vals)
            if vmax - vmin < 1e-8:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - vmin) / (vmax - vmin) for k, v in scores_dict.items()}

        faiss_norm = _minmax(faiss_results)
        bm25_norm = _minmax(bm25_results)

        # 4. 加权融合
        alpha = HYBRID_ALPHA
        combined = {}
        for cid in all_ids:
            faiss_score = faiss_norm.get(cid, 0.0)
            bm25_score = bm25_norm.get(cid, 0.0)
            combined[cid] = alpha * faiss_score + (1 - alpha) * bm25_score

        # 5. 排序取 top_k
        ranked = sorted(combined.items(), key=lambda x: -x[1])

        results = []
        for i, (cid, score) in enumerate(ranked[:top_k]):
            results.append({
                "chunk_id": cid,
                "score": round(score, 4),
                "rank": i + 1,
                "faiss_score": round(faiss_results.get(cid, 0), 4),
                "bm25_score": round(bm25_results.get(cid, 0), 4),
            })

        print(f"  [混合检索] '{query[:40]}...' -> {len(results)} 个结果 (alpha={alpha})")
        return results

    # ---- 嵌入 ----

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """sentence-transformers 本地推理，L2 归一化"""
        start = time.time()
        embs = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embs = np.array(embs, dtype=np.float32)
        elapsed = time.time() - start
        print(f"  [嵌入生成] {len(texts)} 个, 耗时: {elapsed:.3f}s")
        return embs

    # ---- 持久化 ----

    def save_index(self, path: str = VECTOR_STORE_PATH):
        """保存 FAISS 索引"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        print(f"[向量存储] FAISS 索引已保存: {path}.faiss")

    def load_index(self, path: str = VECTOR_STORE_PATH) -> bool:
        """加载 FAISS 索引（BM25 需额外通过 rebuild_bm25 构建）"""
        faiss_path = f"{path}.faiss"
        if not os.path.exists(faiss_path):
            print(f"[向量存储] 未找到索引文件: {faiss_path}")
            return False

        start = time.time()
        self.index = faiss.read_index(faiss_path)
        self.total_vectors = self.index.ntotal
        self.dimension = self.index.d
        print(f"[向量存储] FAISS 索引已加载, {self.total_vectors} 个向量, "
              f"耗时: {time.time() - start:.2f}s")
        return True

    def rebuild_bm25(self, chunk_texts: Dict[int, str]):
        """从 chunk 文本重建 BM25 索引（FAISS 加载后调用）"""
        ids = list(chunk_texts.keys())
        texts = list(chunk_texts.values())
        self.bm25.build(ids, texts)
        print(f"[向量存储] BM25 索引已重建, {len(ids)} 个文档")

    # ---- 工具 ----

    def get_index_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_vectors": self.total_vectors,
        }

    def has_index(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
