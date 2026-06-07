# -*- coding: utf-8 -*-
"""
向量存储模块（高稳定性版）
支持 FAISS（语义）+ BM25（关键词）混合检索
新增：索引健康检查 + 自动修复 + 线程安全强化
"""

import os
import re
import time
import math
import threading
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
    "军用": "军用 国防",
    "国防": "国防 军用",
    "军工": "军工 国防",
    "军方": "军方 国防",
    "收入": "收入 营业收入",
    "营收": "营收 收入",
    "利润": "利润 净利润",
    "主营": "主营 主营业务",
    "年度": "年度 年",
}


def expand_query(query: str) -> str:
    """查询扩展：用关键同义词补充原查询"""
    expanded = query
    for word, replacement in _SYNONYM_MAP.items():
        if word in query:
            expanded = expanded.replace(word, replacement)
    tokens = expanded.split()
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " ".join(unique)


def tokenize(text: str) -> List[str]:
    """中文分词：保留中文词、英文单词、数字"""
    tokens = re.findall(r'[a-zA-Z]+|\d+(?:\.\d+)?', text)
    cn_words = re.findall(r'[\u4e00-\u9fff]+', text)
    for w in cn_words:
        if len(w) <= 4:
            tokens.append(w)
        else:
            tokens.extend([w[i:i+2] for i in range(len(w)-1)])
            tokens.append(w)
    return [t.lower() for t in tokens if t]


class BM25Index:
    """轻量 BM25 索引（Okapi BM25）—— 预分词 + 倒排索引优化版"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}    # {term: set_of_doc_ids}
        self.doc_lengths = []  # [len1, len2, ...]
        self.doc_tokens = []   # [[tok1, ...], ...]
        self.doc_ids = []      # 有序 doc_id 列表
        self.avgdl = 0.0
        self.N = 0
        self.inverted = {}     # {term: [(doc_idx, tf), ...]}

    def build(self, doc_ids: List[int], texts: List[str]):
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
            print(f"  [BM25] 索引构建完成, {self.N} 个文档, 耗时: {elapsed:.2f}s")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.N == 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        unique_terms = set(query_tokens)
        scores = {}
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

    def is_valid(self) -> bool:
        """检查 BM25 索引是否有效"""
        return self.N > 0 and len(self.doc_ids) > 0


class VectorStore:
    """向量存储与混合检索器（FAISS + BM25，高稳定性版）"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model_path = EMBEDDING_MODEL_PATH
        self.dimension = EMBEDDING_DIMENSION
        self.model = SentenceTransformer(self.model_path)
        # 嵌入模型预热
        _warmup_start = time.time()
        try:
            self.model.encode(["预热"], show_progress_bar=False,
                              normalize_embeddings=True)
        except Exception:
            pass  # 预热失败不阻塞
        _warmup_elapsed = time.time() - _warmup_start
        self.index = None
        self.bm25 = BM25Index()
        self.total_vectors = 0
        self._lock = threading.Lock()
        self._embed_lock = threading.Lock()

        # ── 错误状态标记 ──
        self._index_error = None
        self._last_health_check = 0
        self._health_check_interval = 30  # 秒

        print(f"[向量存储] 嵌入模型: {self.model_name}, 维度: {self.dimension}"
              f" (预热: {_warmup_elapsed:.3f}s)")
        print(f"[向量存储] 混合检索: FAISS(α={HYBRID_ALPHA}) + BM25(β={1-HYBRID_ALPHA:.1f})")

    # ---- 索引管理 ----

    def _create_index(self):
        """创建 FAISS IndexIDMap（余弦相似度 = InnerProduct）"""
        base = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base)
        self.total_vectors = 0
        self._index_error = None

    def build_index(self, ids: List[int], texts: List[str]):
        """构建 FAISS + BM25 双索引"""
        if not ids or not texts:
            raise ValueError("构建索引时 ids 和 texts 不能为空")

        start = time.time()
        count = len(ids)
        print(f"[向量存储] 构建索引, {count} 个文本块")

        # 1. FAISS 语义索引
        embeddings = self.embed_texts(texts)
        ids_arr = np.array(ids, dtype=np.int64)
        self._create_index()
        with self._lock:
            self.index.add_with_ids(embeddings, ids_arr)
            self.total_vectors = count

        # 2. BM25 关键词索引
        self.bm25.build(ids, texts)

        elapsed = time.time() - start
        print(f"  [向量存储] 索引构建完成, {count} 个向量+BM25, 耗时: {elapsed:.2f}s")
        self._index_error = None

    def add_vectors(self, new_ids: List[int], new_texts: List[str]):
        """增量添加向量"""
        if self.index is None:
            return self.build_index(new_ids, new_texts)

        try:
            embeddings = self.embed_texts(new_texts)
            ids_arr = np.array(new_ids, dtype=np.int64)
            with self._lock:
                self.index.add_with_ids(embeddings, ids_arr)
                self.total_vectors += len(new_ids)
            # BM25 增量需重建
            self.bm25.build(
                list(range(1, self.total_vectors + 1)),
                []
            )
        except Exception as e:
            print(f"  [向量存储] 增量添加失败: {e}")
            raise

    # ---- 健康检查 + 自动修复 ----

    def check_health(self) -> Dict[str, Any]:
        """检查索引健康状态"""
        status = {
            "healthy": True,
            "has_index": False,
            "total_vectors": 0,
            "bm25_valid": False,
            "errors": [],
            "warnings": [],
        }

        # FAISS 检查
        if self.index is not None and self.index.ntotal > 0:
            status["has_index"] = True
            status["total_vectors"] = self.index.ntotal
            # 验证索引可读
            try:
                dummy_vec = np.zeros((1, self.dimension), dtype=np.float32)
                self.index.search(dummy_vec, 1)
            except Exception as e:
                status["healthy"] = False
                status["errors"].append(f"FAISS 索引不可用: {e}")
        else:
            status["warnings"].append("FAISS 索引为空")

        # BM25 检查
        status["bm25_valid"] = self.bm25.is_valid()
        if not status["bm25_valid"] and status["has_index"]:
            status["warnings"].append("BM25 索引为空（FAISS 存在，部分降级）")

        # 维度一致性检查
        if self.index is not None:
            try:
                actual_dim = self.index.d
                if actual_dim != self.dimension:
                    status["healthy"] = False
                    status["errors"].append(
                        f"维度不匹配: 索引={actual_dim}, 配置={self.dimension}"
                    )
            except Exception as e:
                status["warnings"].append(f"维度检查失败: {e}")

        if self._index_error:
            status["errors"].append(self._index_error)
            if status["healthy"]:
                status["healthy"] = False

        self._last_health_check = time.time()
        return status

    def auto_repair(self, chunk_texts: Optional[Dict[int, str]] = None) -> bool:
        """
        自动修复索引

        场景：
        - FAISS 索引损坏 → 重建（如有备份或 chunk 数据）
        - BM25 索引损坏 → 从 chunk_texts 重建
        - 维度不匹配 → 重建
        """
        print(f"  [自动修复] 开始检查索引...")
        health = self.check_health()

        if health["healthy"]:
            print(f"  [自动修复] 索引健康，无需修复")
            return True

        repaired = False

        # 1. BM25 损坏且有 chunk 数据 → 重建 BM25
        if not health["bm25_valid"] and chunk_texts:
            try:
                self.rebuild_bm25(chunk_texts)
                print(f"  [自动修复] BM25 索引已重建")
                repaired = True
            except Exception as e:
                print(f"  [自动修复] BM25 重建失败: {e}")

        # 2. FAISS 损坏且有 chunk 数据 → 重建
        if health["has_index"] and not health["healthy"] and chunk_texts:
            try:
                ids = list(chunk_texts.keys())
                texts = list(chunk_texts.values())
                self.build_index(ids, texts)
                print(f"  [自动修复] FAISS + BM25 索引已重建")
                repaired = True
            except Exception as e:
                print(f"  [自动修复] 索引重建失败: {e}")
                self._index_error = str(e)

        # 3. FAISS 完全不存在 → 初始构建
        if not health["has_index"] and chunk_texts:
            try:
                ids = list(chunk_texts.keys())
                texts = list(chunk_texts.values())
                self.build_index(ids, texts)
                print(f"  [自动修复] 初始索引构建完成")
                repaired = True
            except Exception as e:
                print(f"  [自动修复] 初始构建失败: {e}")
                self._index_error = str(e)

        if repaired:
            self._index_error = None
            print(f"  [自动修复] ✅ 修复完成")
        else:
            print(f"  [自动修复] ❌ 修复失败（可能缺少 chunk 数据）")

        return repaired

    # ---- 混合检索 ----

    def search(self, query: str, top_k: int = TOP_K_RESULTS
               ) -> List[Dict[str, Any]]:
        """混合搜索：FAISS 语义（α权重） + BM25 关键词（1-α权重）"""
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("向量索引为空，请先构建索引")

        # 1. FAISS 语义检索
        faiss_k = top_k * 2
        query_vec = self.embed_texts([query])
        with self._lock:
            distances, labels = self.index.search(query_vec, faiss_k)

        faiss_results = {}
        for i in range(len(labels[0])):
            label = labels[0][i]
            if label < 0:
                continue
            faiss_results[int(label)] = float(distances[0][i])

        # 2. BM25 关键词检索
        expanded_query = expand_query(query)
        bm25_results_raw = self.bm25.search(expanded_query, top_k=faiss_k)
        bm25_results = {r["chunk_id"]: r["score"] for r in bm25_results_raw}

        # 3. 分数归一化
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

        return results

    # ---- 嵌入 ----

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        with self._embed_lock:
            embs = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        embs = np.array(embs, dtype=np.float32)
        return embs

    # ---- 持久化 ----

    def save_index(self, path: str = VECTOR_STORE_PATH):
        """保存 FAISS 索引（带备份机制）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._lock:
            # 先写 .bak，再原子重命名，防止写入中断损坏
            tmp_path = f"{path}.faiss.tmp"
            dst_path = f"{path}.faiss"
            faiss.write_index(self.index, tmp_path)
            if os.path.exists(dst_path):
                os.replace(dst_path, f"{path}.faiss.bak")
            os.replace(tmp_path, dst_path)
        print(f"[向量存储] FAISS 索引已保存: {dst_path}")

    def load_index(self, path: str = VECTOR_STORE_PATH) -> bool:
        """加载 FAISS 索引（带备份回退）"""
        faiss_path = f"{path}.faiss"
        bak_path = f"{path}.faiss.bak"

        for candidate, label in [(faiss_path, "主文件"), (bak_path, "备份")]:
            if not os.path.exists(candidate):
                continue
            try:
                start = time.time()
                with self._lock:
                    self.index = faiss.read_index(candidate)
                    self.total_vectors = self.index.ntotal
                    self.dimension = self.index.d
                print(f"[向量存储] FAISS 索引已加载 ({label}), "
                      f"{self.total_vectors} 个向量, "
                      f"耗时: {time.time() - start:.2f}s")
                self._index_error = None
                return True
            except Exception as e:
                print(f"  [向量存储] 加载 {label} 失败: {e}")
                continue

        print(f"[向量存储] 未找到可用索引文件: {faiss_path}")
        return False

    def rebuild_bm25(self, chunk_texts: Dict[int, str]):
        """从 chunk 文本重建 BM25 索引"""
        ids = list(chunk_texts.keys())
        texts = list(chunk_texts.values())
        self.bm25.build(ids, texts)
        print(f"[向量存储] BM25 索引已重建, {len(ids)} 个文档")

    # ---- 工具 ----

    def get_index_info(self) -> Dict[str, Any]:
        health = self.check_health()
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_vectors": self.total_vectors,
            "bm25_valid": health["bm25_valid"],
            "index_healthy": health["healthy"],
            "last_health_check": self._last_health_check,
        }

    def has_index(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
