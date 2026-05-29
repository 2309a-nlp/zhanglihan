# -*- coding: utf-8 -*-
"""
重排序模块
使用 bge-reranker-v2-m3 对检索结果进行精排
"""
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 全局缓存重排序模型（避免每次请求都加载）
_reranker = None
_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _load_reranker():
    """延迟加载重排序模型（依赖 rag_service.py 已设置离线环境变量）"""
    global _reranker
    if _reranker is None:
        logger.info(f"正在加载重排序模型: {_RERANKER_MODEL_NAME}...")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _reranker = {
                "model": AutoModelForSequenceClassification.from_pretrained(
                    _RERANKER_MODEL_NAME,
                    trust_remote_code=True,
                    local_files_only=True
                ),
                "tokenizer": AutoTokenizer.from_pretrained(
                    _RERANKER_MODEL_NAME,
                    trust_remote_code=True,
                    local_files_only=True
                ),
            }
            logger.info("重排序模型加载完成")
        except Exception as e:
            logger.warning(f"重排序模型加载失败（离线不可用）: {e}")
            raise
    return _reranker


def rerank(query: str, documents: List[str], top_k: int = None) -> List[Tuple[str, float]]:
    """
    对检索结果进行重排序

    参数：
        query: 用户问题
        documents: 待重排序的文档片段列表
        top_k: 返回前多少条（默认全部返回）

    返回：
        按相关性从高到低排序的 [(文本, 分数), ...]
    """
    if not documents:
        return []

    try:
        reranker = _load_reranker()
        model = reranker["model"]
        tokenizer = reranker["tokenizer"]

        # 构造输入对： (query, doc)  pairs
        pairs = [[query, doc] for doc in documents]

        # 使用 bge-reranker-v2-m3 的 compute_score 方法
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        scores = scores.detach().numpy().tolist()

        # 组合文档和分数，按分数降序排序
        results = list(zip(documents, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k and top_k < len(results):
            results = results[:top_k]

        logger.info(f"重排序完成: 输入{len(documents)}条, 返回{len(results)}条, Top1分数={results[0][1]:.4f}")
        return results

    except Exception as e:
        logger.warning(f"重排序失败（退回原始排序）: {e}")
        # 失败时返回原始顺序
        return [(doc, 0.0) for doc in documents]


def rerank_documents(query: str, docs_list: list, top_k: int = None) -> list:
    """
    对 Document 对象列表进行重排序

    参数：
        query: 用户问题
        docs_list: [Document, ...] 类型的列表，每个有 page_content 属性
        top_k: 返回前多少条

    返回：
        重排序后的 [Document, ...] 列表
    """
    if not docs_list:
        return []

    # 提取文本
    texts = [doc.page_content for doc in docs_list]

    # 重排序
    reranked = rerank(query, texts, top_k)

    # 按重排序后的顺序重新排列 docs_list
    reranked_texts = [item[0] for item in reranked]
    reranked_scores = {item[0]: item[1] for item in reranked}

    # 按重排序后的顺序重建 docs_list
    doc_map = {doc.page_content: doc for doc in docs_list}
    result = []
    for text in reranked_texts:
        if text in doc_map:
            doc = doc_map[text]
            doc.metadata["rerank_score"] = reranked_scores.get(text, 0.0)
            result.append(doc)

    return result
