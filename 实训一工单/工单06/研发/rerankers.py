"""
重排模块 — 提供 3 种重排算法
1. 基于 LLM 的重排器（BGE-Reranker-v2-m3）
2. 基于 TF-IDF 的重排器（轻量快速）
3. 基于用户反馈的自适应重排器（在线学习用户偏好）
"""

import os
import json
import math
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ==================== 1. LLM 重排器 ====================

class LLMReranker:
    """
    基于 BGE-Reranker-v2-m3 的 LLM 重排器
    使用交叉编码器（cross-encoder）对 query-doc 对进行精确评分
    """

    def __init__(self, model_path: str = None, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # 自动检测本地模型路径
        if model_path is None:
            local_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models", model_name.split("/")[-1]
            )
            if os.path.exists(local_path):
                self.model_path = local_path
            else:
                self.model_path = None
        else:
            self.model_path = model_path
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载 reranker 模型"""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # 优先使用本地模型
            if self.model_path and os.path.exists(self.model_path):
                load_path = self.model_path
                logger.info(f"加载本地重排模型: {load_path}")
            else:
                load_path = self.model_name
                logger.info(f"加载在线重排模型: {load_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                load_path, trust_remote_code=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                load_path, trust_remote_code=True
            )
            self._model.eval()
            logger.info("LLM 重排模型加载完成")
        except Exception as e:
            logger.warning(f"LLM 重排模型加载失败: {e}，回退到 TF-IDF 重排器")
            self._model = None

    def rerank(self, query: str, documents: List[dict],
               texts_key: str = "text", top_k: Optional[int] = None) -> List[Tuple[dict, float]]:
        """
        对检索结果进行 LLM 重排

        参数:
            query: 查询文本
            documents: 文档列表 [{text, source, ...}]
            texts_key: 文档中文本字段名
            top_k: 返回前多少条

        返回:
            [(document, score), ...]
        """
        if not documents:
            return []

        self._load_model()
        if self._model is None:
            # 模型加载失败，使用 TF-IDF 回退
            fallback = TfidfReranker()
            return fallback.rerank(query, documents, texts_key, top_k)

        import torch

        texts = [doc.get(texts_key, "") for doc in documents]
        pairs = [[query, text] for text in texts]

        try:
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            with torch.no_grad():
                outputs = self._model(**inputs, return_dict=True)
                scores = outputs.logits.view(-1).float().numpy().tolist()

            # 组合文档和分数
            results = list(zip(documents, scores))
            results.sort(key=lambda x: x[1], reverse=True)

            if top_k and top_k < len(results):
                results = results[:top_k]

            return results
        except Exception as e:
            logger.warning(f"LLM 重排推理失败: {e}，回退到 TF-IDF")
            fallback = TfidfReranker()
            return fallback.rerank(query, documents, texts_key, top_k)


# ==================== 2. TF-IDF 重排器 ====================

class TfidfReranker:
    """
    基于 TF-IDF + Cosine Similarity 的轻量重排器
    无需 GPU，速度快，适合冷启动和低资源场景
    """

    def __init__(self):
        self._vectorizer = None

    def _get_vectorizer(self):
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 6),
                max_features=50000,
                sublinear_tf=True,
                norm="l2",
            )
        return self._vectorizer

    def rerank(self, query: str, documents: List[dict],
               texts_key: str = "text", top_k: Optional[int] = None) -> List[Tuple[dict, float]]:
        """
        TF-IDF 重排
        将 query 和文档在同一 TF-IDF 空间计算余弦相似度
        """
        if not documents:
            return []

        texts = [doc.get(texts_key, "") for doc in documents]

        # 用查询+文档一起做 TF-IDF 拟合
        all_texts = [query] + texts
        vectorizer = self._get_vectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vec = tfidf_matrix[0:1]
            doc_vecs = tfidf_matrix[1:]

            similarities = cosine_similarity(query_vec, doc_vecs).flatten()

            results = list(zip(documents, similarities))
            results.sort(key=lambda x: x[1], reverse=True)

            if top_k and top_k < len(results):
                results = results[:top_k]

            return results
        except Exception as e:
            logger.warning(f"TF-IDF 重排失败: {e}")
            # 返回原始顺序
            results = [(doc, 0.0) for doc in documents]
            if top_k:
                results = results[:top_k]
            return results


# ==================== 3. 自适应重排器 ====================

class AdaptiveReranker:
    """
    基于用户反馈的自适应重排器
    - 记录用户点击/采纳的文档
    - 维护每个文档（或文档特征）的偏好分数
    - 使用 exponential decay 平滑历史反馈
    - 支持在线学习
    """

    def __init__(self, feedback_db: str = None, learning_rate: float = 0.1, decay: float = 0.95):
        self.feedback_db = feedback_db or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "user_feedback.json"
        )
        self.learning_rate = learning_rate
        self.decay = decay
        self._preferences = defaultdict(float)   # doc_id -> preference_score
        self._history = defaultdict(list)        # doc_id -> [(timestamp, feedback), ...]
        self._load_feedback()

    def _load_feedback(self):
        """加载持久化的用户反馈"""
        if os.path.exists(self.feedback_db):
            try:
                with open(self.feedback_db, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._preferences = defaultdict(float, data.get("preferences", {}))
                self._history = defaultdict(
                    list,
                    {k: [(ts, fb) for ts, fb in v] for k, v in data.get("history", {}).items()}
                )
                logger.info(f"已加载 {len(self._preferences)} 条用户反馈偏好")
            except Exception as e:
                logger.warning(f"加载反馈数据失败: {e}")

    def _save_feedback(self):
        """持久化用户反馈"""
        try:
            data = {
                "preferences": dict(self._preferences),
                "history": {
                    k: list(v) for k, v in self._history.items()
                },
            }
            os.makedirs(os.path.dirname(self.feedback_db), exist_ok=True)
            with open(self.feedback_db, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存反馈数据失败: {e}")

    def record_feedback(self, query: str, doc_text: str, liked: bool):
        """
        记录用户反馈

        参数:
            query: 用户问题
            doc_text: 文档文本内容
            liked: True=用户采纳/点击, False=用户忽略/负反馈
        """
        import time
        doc_id = hash(doc_text)

        # 更新偏好分数
        old_score = self._preferences[doc_id]
        change = self.learning_rate * (1.0 if liked else -1.0)
        new_score = old_score + change
        self._preferences[doc_id] = max(-5.0, min(5.0, new_score))

        # 记录历史
        self._history[doc_id].append((time.time(), 1 if liked else 0))

        # 衰减旧反馈
        self._decay_old_feedback()

        self._save_feedback()

    def _decay_old_feedback(self):
        """对历史反馈进行时间衰减"""
        import time
        now = time.time()
        for doc_id in list(self._history.keys()):
            self._history[doc_id] = [
                (ts, fb) for ts, fb in self._history[doc_id]
                if now - ts < 86400 * 30  # 保留30天
            ]
            if not self._history[doc_id]:
                del self._history[doc_id]

    def rerank(self, query: str, documents: List[dict],
               texts_key: str = "text", top_k: Optional[int] = None) -> List[Tuple[dict, float]]:
        """
        基于用户偏好的自适应重排
        先用 TF-IDF 得到基础分，再用偏好分加权
        """
        if not documents:
            return []

        # 先用 TF-IDF 计算基础分数
        tfidf_reranker = TfidfReranker()
        base_results = tfidf_reranker.rerank(query, documents, texts_key)

        # 加入偏好分数
        results = []
        for doc, base_score in base_results:
            doc_text = doc.get(texts_key, "")
            doc_id = hash(doc_text)
            pref_score = self._preferences.get(doc_id, 0.0)

            # 偏好分数归一化到 [-0.5, 0.5] 并叠加
            pref_norm = math.tanh(pref_score * 0.3)
            combined_score = base_score + pref_norm * 0.3
            results.append((doc, combined_score))

        results.sort(key=lambda x: x[1], reverse=True)

        if top_k and top_k < len(results):
            results = results[:top_k]

        return results


# ==================== 工厂函数 ====================

RERANKER_REGISTRY = {
    "llm": LLMReranker,
    "tfidf": TfidfReranker,
    "adaptive": AdaptiveReranker,
}


def get_reranker(name: str = "llm", **kwargs):
    """
    获取指定的重排器实例

    参数:
        name: "llm" | "tfidf" | "adaptive"
        **kwargs: 传递给重排器的参数

    返回:
        重排器实例
    """
    cls = RERANKER_REGISTRY.get(name)
    if cls is None:
        logger.warning(f"未知的重排器: {name}，使用 LLM 重排器")
        cls = LLMReranker
    return cls(**kwargs)


def rerank_results(query: str, documents: List[dict],
                   reranker_name: str = "llm",
                   texts_key: str = "text",
                   top_k: Optional[int] = None) -> List[Tuple[dict, float]]:
    """
    统一重排接口

    参数:
        query: 查询文本
        documents: 待重排文档列表
        reranker_name: "llm" | "tfidf" | "adaptive"
        texts_key: 文档中文本字段名
        top_k: 返回数

    返回:
        重排后的 [(doc, score), ...]
    """
    reranker = get_reranker(reranker_name)
    return reranker.rerank(query, documents, texts_key, top_k)
