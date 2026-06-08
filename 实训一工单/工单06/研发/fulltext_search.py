"""
全文检索模块 — 基于倒排索引
支持：布尔查询（AND/OR/NOT）、短语匹配、模糊匹配（编辑距离）
目标：准确率 > 90%，召回率 > 95%，响应 < 3s
"""

import os
import re
import json
import math
import pickle
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import jieba

logger = logging.getLogger(__name__)


class InvertedIndex:
    """
    倒排索引实现
    - 文档级倒排表: term -> {doc_id: term_frequency}
    - 支持布尔查询、短语匹配、模糊匹配
    - BM25 评分
    """

    def __init__(self):
        self.documents = []             # [{doc_id, text, source, chunk_id}, ...]
        self.inverted_index = defaultdict(dict)  # term -> {doc_id: freq}
        self.doc_lengths = []            # 每个文档的长度（词数）
        self.avg_doc_length = 0.0
        self.total_docs = 0
        self.k1 = 1.5   # BM25 参数
        self.b = 0.75   # BM25 参数
        self._built = False
        self._idf_cache = {}

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        text = text.lower()
        # 使用 jieba 进行中文分词
        words = jieba.lcut(text)
        # 过滤停用词和过短的词
        stopwords = self._get_stopwords()
        tokens = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            if len(w) < 2 and not w.isdigit():
                continue
            if w in stopwords:
                continue
            tokens.append(w)
        return tokens

    def _get_stopwords(self) -> set:
        """获取中文停用词表"""
        return {
            "的", "了", "是", "在", "和", "与", "及", "就", "也", "都", "而",
            "且", "或", "但", "被", "把", "从", "对", "到", "以", "为", "由",
            "于", "之", "这", "那", "哪", "什么", "怎么", "如何", "多少", "每个",
            "各", "有", "不", "很", "能", "会", "要", "可", "该", "这个", "那个",
            "一个", "请", "问", "回答", "提供", "分别", "属于", "来自",
            "中", "上", "下", "前", "后", "大", "小", "多", "少",
            "来", "去", "第", "等", "还", "没", "而", "且"
        }

    def build(self, documents: List[dict]):
        """
        构建倒排索引

        参数:
            documents: [{"text": "...", "source": "file.pdf", "chunk_id": 0}, ...]
        """
        self.documents = documents
        self.total_docs = len(documents)
        self.doc_lengths = [0] * self.total_docs

        for doc_id, doc in enumerate(documents):
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)

            # 统计词频
            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                self.inverted_index[term][doc_id] = freq

        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)
        self._idf_cache = {}
        self._built = True
        logger.info(f"倒排索引构建完成: {self.total_docs} 文档, {len(self.inverted_index)} 词项")

    def _idf(self, term: str) -> float:
        """计算 IDF"""
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = len(self.inverted_index.get(term, {}))
        idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = idf
        return idf

    def _bm25_score(self, term: str, doc_id: int) -> float:
        """BM25 评分"""
        freq = self.inverted_index.get(term, {}).get(doc_id, 0)
        if freq == 0:
            return 0.0
        idf = self._idf(term)
        doc_len = self.doc_lengths[doc_id]
        score = idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))
        return score

    def _normalize_score(self, scores: Dict[int, float]) -> Dict[int, float]:
        """归一化分数到 [0, 1]"""
        if not scores:
            return scores
        max_score = max(scores.values())
        if max_score == 0:
            return scores
        return {k: v / max_score for k, v in scores.items()}

    def search_boolean(self, query: str, top_k: int = 20) -> List[Tuple[dict, float]]:
        """
        布尔查询
        支持: AND, OR, NOT (大写), 默认空格 OR 连接
        """
        query = query.strip()
        if not query:
            return []

        # 解析布尔表达式
        terms = re.split(r'\s+(AND|OR|NOT)\s+', query)
        current_op = "OR"
        include_terms = []
        exclude_terms = []

        i = 0
        while i < len(terms):
            token = terms[i].strip()
            if token == "AND":
                current_op = "AND"
                i += 1
                continue
            elif token == "OR":
                current_op = "OR"
                i += 1
                continue
            elif token == "NOT":
                current_op = "NOT"
                i += 1
                continue
            elif not token:
                i += 1
                continue

            # 对中文 query 进行分词
            tokens = self._tokenize(token)
            if not tokens:
                tokens = [token]

            if current_op == "NOT":
                exclude_terms.extend(tokens)
            elif current_op == "AND":
                include_terms.append(tokens)  # AND group
            else:
                include_terms.append(tokens)  # OR group

            current_op = "OR"
            i += 1

        # 计算文档集合
        if not include_terms and not exclude_terms:
            return []

        # 合并 inclusion terms (OR 语义)
        candidate_docs = set()
        for term_group in include_terms:
            group_docs = set()
            for term in term_group:
                group_docs.update(self.inverted_index.get(term, {}).keys())
            candidate_docs.update(group_docs)

        # AND 交集的文档需在所有 AND groups 中都出现
        # 简化: 对每一组 AND token，只保留都在的文档
        for term_group in include_terms:
            if len(term_group) > 1:
                and_docs = set(range(self.total_docs))
                for term in term_group:
                    and_docs &= set(self.inverted_index.get(term, {}).keys())
                candidate_docs &= and_docs

        # 排除 NOT 文档
        for term in exclude_terms:
            exclude_docs = set(self.inverted_index.get(term, {}).keys())
            candidate_docs -= exclude_docs

        # BM25 评分
        scores = {}
        for doc_id in candidate_docs:
            score = 0.0
            for term_group in include_terms:
                for term in term_group:
                    score += self._bm25_score(term, doc_id)
            scores[doc_id] = score

        scores = self._normalize_score(scores)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [(self.documents[doc_id], score) for doc_id, score in sorted_docs]

    def search_phrase(self, query: str, top_k: int = 20) -> List[Tuple[dict, float]]:
        """
        短语匹配 — 精确短语匹配
        要求文档中包含连续的查询词
        """
        query = query.strip()
        if not query:
            return []

        # 对查询进行分词
        terms = self._tokenize(query)
        if not terms or len(terms) < 2:
            # 单次查询回退到布尔
            return self.search_boolean(query, top_k)

        # 找到包含所有词项的文档
        candidate_docs = None
        for term in terms:
            term_docs = set(self.inverted_index.get(term, {}).keys())
            if candidate_docs is None:
                candidate_docs = term_docs
            else:
                candidate_docs &= term_docs

        if not candidate_docs:
            return []

        # 在候选文档中验证短语连续性
        scores = {}
        for doc_id in candidate_docs:
            doc_text = self.documents[doc_id].get("text", "").lower()
            # 检查查询是否作为连续子串出现
            query_lower = query.lower().strip()
            # 去除所有空格
            query_compact = re.sub(r'\s+', '', query_lower)
            doc_compact = re.sub(r'\s+', '', doc_text)
            if query_compact in doc_compact:
                # 精确匹配短语，加高分
                count = doc_compact.count(query_compact)
                scores[doc_id] = 1.0 + (count * 0.1)
            else:
                # 部分匹配，用 BM25 兜底
                for term in terms:
                    scores[doc_id] = scores.get(doc_id, 0.0) + self._bm25_score(term, doc_id)

        scores = self._normalize_score(scores)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[doc_id], score) for doc_id, score in sorted_docs]

    def _edit_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def search_fuzzy(self, query: str, top_k: int = 20, max_edits: int = 2) -> List[Tuple[dict, float]]:
        """
        模糊匹配 — 基于编辑距离
        对查询词进行模糊匹配，召回拼写/分词变体
        """
        query = query.strip()
        if not query:
            return []

        terms = self._tokenize(query)
        if not terms:
            return []

        # 为每个查询词找到模糊匹配的词项
        fuzzy_matches = {}  # term -> [(matched_term, distance), ...]
        for term in terms:
            matches = []
            for index_term in self.inverted_index.keys():
                dist = self._edit_distance(term, index_term)
                if dist <= max_edits:
                    matches.append((index_term, dist))
            matches.sort(key=lambda x: x[1])
            fuzzy_matches[term] = matches[:5]  # 最多5个模糊匹配

        # 用模糊匹配结果检索
        scores = {}
        for doc_id in range(self.total_docs):
            score = 0.0
            for query_term, matches in fuzzy_matches.items():
                if not matches:
                    continue
                # 使用最佳匹配的分数
                best_term, best_dist = matches[0]
                term_score = self._bm25_score(best_term, doc_id)
                if term_score > 0:
                    # 编辑距离惩罚
                    distance_penalty = 1.0 / (1.0 + best_dist * 0.5)
                    score += term_score * distance_penalty
                else:
                    # 尝试所有模糊匹配
                    for match_term, dist in matches:
                        ts = self._bm25_score(match_term, doc_id)
                        if ts > 0:
                            score += ts * (1.0 / (1.0 + dist * 0.5))
                            break
            if score > 0:
                scores[doc_id] = score

        scores = self._normalize_score(scores)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[doc_id], score) for doc_id, score in sorted_docs]

    def search(self, query: str, mode: str = "boolean", top_k: int = 20) -> List[Tuple[dict, float]]:
        """
        统一检索接口

        参数:
            query: 查询文本
            mode: "boolean" | "phrase" | "fuzzy"
            top_k: 返回数
        """
        if not self._built:
            logger.warning("倒排索引尚未构建")
            return []

        if mode == "phrase":
            return self.search_phrase(query, top_k)
        elif mode == "fuzzy":
            return self.search_fuzzy(query, top_k)
        else:
            return self.search_boolean(query, top_k)

    def save(self, path: str):
        """保存倒排索引到磁盘"""
        os.makedirs(path, exist_ok=True)
        data = {
            "documents": self.documents,
            "inverted_index": dict(self.inverted_index),
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "total_docs": self.total_docs,
        }
        with open(os.path.join(path, "inverted_index.pkl"), "wb") as f:
            pickle.dump(data, f)
        logger.info(f"倒排索引已保存: {path}")

    def load(self, path: str) -> bool:
        """从磁盘加载倒排索引"""
        pkl_path = os.path.join(path, "inverted_index.pkl")
        if not os.path.exists(pkl_path):
            return False
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.inverted_index = defaultdict(dict, data["inverted_index"])
        self.doc_lengths = data["doc_lengths"]
        self.avg_doc_length = data["avg_doc_length"]
        self.total_docs = data["total_docs"]
        self._built = True
        logger.info(f"倒排索引已加载: {self.total_docs} 文档")
        return True
