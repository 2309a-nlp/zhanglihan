"""
混合检索模块 — 融合向量检索 + 全文检索
支持：加权融合、RRF（Reciprocal Rank Fusion）、动态权重
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    混合检索器
    支持以下融合策略：
    1. 加权融合（weighted）：向量分数 * w_v + 全文分数 * w_f
    2. RRF（Reciprocal Rank Fusion）：1/(k + rank) 融合
    3. 级联融合（cascade）：先用全文粗筛，再用向量精排
    """

    def __init__(self, vector_store, fulltext_index,
                 vector_weight: float = 0.6, fulltext_weight: float = 0.4,
                 rrf_k: int = 60):
        """
        参数:
            vector_store: VectorStore 实例
            fulltext_index: InvertedIndex 实例
            vector_weight: 向量检索权重
            fulltext_weight: 全文检索权重
            rrf_k: RRF 常数
        """
        self.vector_store = vector_store
        self.fulltext_index = fulltext_index
        self.vector_weight = vector_weight
        self.fulltext_weight = fulltext_weight
        self.rrf_k = rrf_k

    def search_weighted(self, query: str, query_embedding: np.ndarray,
                        ft_mode: str = "boolean",
                        top_k: int = 10,
                        vector_top_k: int = 20,
                        fulltext_top_k: int = 20) -> List[tuple]:
        """
        加权融合检索

        1. 分别执行向量检索和全文检索
        2. 对分数进行归一化
        3. 按权重加权求和
        """
        # 向量检索
        vector_results = self.vector_store.search(query_embedding, top_k=vector_top_k)
        vector_scores = {id(doc): score for doc, score in vector_results}

        # 全文检索
        fulltext_results = self.fulltext_index.search(query, mode=ft_mode, top_k=fulltext_top_k)

        if not vector_results and not fulltext_results:
            return []

        # 归一化向量分数
        if vector_results:
            v_scores = [s for _, s in vector_results]
            v_max, v_min = max(v_scores), min(v_scores)
            v_range = v_max - v_min if v_max != v_min else 1.0
        else:
            v_range = 1.0

        # 归一化全文分数
        if fulltext_results:
            f_scores = [s for _, s in fulltext_results]
            f_max, f_min = max(f_scores), min(f_scores)
            f_range = f_max - f_min if f_max != f_min else 1.0
        else:
            f_range = 1.0

        # 合并分数
        combined = {}
        doc_map = {}

        for doc, score in vector_results:
            doc_id = id(doc)
            norm_score = (score - v_min) / v_range if v_range > 0 else score
            combined[doc_id] = norm_score * self.vector_weight
            doc_map[doc_id] = doc

        for doc, score in fulltext_results:
            doc_id = id(doc)
            norm_score = score / f_range if f_range > 0 else score
            if doc_id in combined:
                combined[doc_id] += norm_score * self.fulltext_weight
            else:
                combined[doc_id] = norm_score * self.fulltext_weight
                doc_map[doc_id] = doc

        # 排序
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        results = [(doc_map[doc_id], score) for doc_id, score in sorted_results[:top_k]]

        return results

    def search_rrf(self, query: str, query_embedding: np.ndarray,
                   ft_mode: str = "boolean",
                   top_k: int = 10,
                   vector_top_k: int = 20,
                   fulltext_top_k: int = 20) -> List[tuple]:
        """
        RRF 融合检索
        RRF_score = sum(1/(k + rank_i))
        """
        k = self.rrf_k

        # 向量检索
        vector_results = self.vector_store.search(query_embedding, top_k=vector_top_k)
        # 全文检索
        fulltext_results = self.fulltext_index.search(query, mode=ft_mode, top_k=fulltext_top_k)

        if not vector_results and not fulltext_results:
            return []

        # RRF 评分
        rrf_scores = {}
        doc_map = {}

        for rank, (doc, _) in enumerate(vector_results, 1):
            doc_id = id(doc)
            rrf_scores[doc_id] = 1.0 / (k + rank)
            doc_map[doc_id] = doc

        for rank, (doc, _) in enumerate(fulltext_results, 1):
            doc_id = id(doc)
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += 1.0 / (k + rank)
            else:
                rrf_scores[doc_id] = 1.0 / (k + rank)
                doc_map[doc_id] = doc

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = [(doc_map[doc_id], score) for doc_id, score in sorted_results[:top_k]]

        return results

    def search_cascade(self, query: str, query_embedding: np.ndarray,
                       ft_mode: str = "boolean",
                       top_k: int = 10,
                       fulltext_top_k: int = 50,
                       vector_top_k: int = 10) -> List[tuple]:
        """
        级联融合检索
        先用全文检索粗筛（大候选池），再用向量检索精排
        """
        # 全文粗筛
        fulltext_results = self.fulltext_index.search(query, mode=ft_mode, top_k=fulltext_top_k)
        if not fulltext_results:
            # 全文检索无结果，回退到纯向量检索
            return self.vector_store.search(query_embedding, top_k=top_k)

        # 提取全文检索到的文本
        fulltext_texts = set()
        for doc, _ in fulltext_results:
            fulltext_texts.add(doc.get("text", ""))

        # 向量检索全部文档，但只保留在全文结果中的文档
        all_vector_results = self.vector_store.search(query_embedding, top_k=len(self.vector_store.documents))

        # 过滤：只保留全文检索也命中的文档
        filtered = []
        for doc, score in all_vector_results:
            if doc.get("text", "") in fulltext_texts:
                filtered.append((doc, score))

        if not filtered:
            # 没有交集，返回全文结果
            return fulltext_results[:top_k]

        # 用向量分数排序
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]

    def search(self, query: str, query_embedding: np.ndarray,
               strategy: str = "weighted",
               ft_mode: str = "boolean",
               top_k: int = 10,
               **kwargs) -> List[tuple]:
        """
        统一混合检索接口

        参数:
            query: 查询文本
            query_embedding: 查询向量
            strategy: "weighted" | "rrf" | "cascade"
            ft_mode: 全文检索模式
            top_k: 返回结果数
        """
        if strategy == "rrf":
            return self.search_rrf(query, query_embedding, ft_mode, top_k, **kwargs)
        elif strategy == "cascade":
            return self.search_cascade(query, query_embedding, ft_mode, top_k, **kwargs)
        else:
            return self.search_weighted(query, query_embedding, ft_mode, top_k, **kwargs)
