"""
RAG 系统初始化与主检索入口
整合：向量检索 + 全文检索 + 混合检索 + 重排
"""

import os
import time
import logging
import numpy as np
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# 全局组件
_vector_store = None
_fulltext_index = None
_hybrid_search = None
_initialized = False
_chunks = []
_embedding_model = None


def init_rag(force_rebuild: bool = False, embedding_model: Optional[str] = None) -> bool:
    """
    初始化 RAG 系统

    参数:
        force_rebuild: 强制重建索引
        embedding_model: 嵌入模型名称
    """
    global _vector_store, _fulltext_index, _hybrid_search, _initialized, _chunks, _embedding_model

    from config import (
        DATA_DIR, VECTOR_STORE_DIR, FULLTEXT_INDEX_DIR,
        DEFAULT_EMBEDDING_MODEL, VECTOR_TOP_K, HYBRID_TOP_K,
        HYBRID_WEIGHTS,
    )
    from pdf_processor import process_pdfs
    from embeddings import embed_texts, get_available_models
    from vector_store import VectorStore
    from fulltext_search import InvertedIndex
    from hybrid_search import HybridSearch

    model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
    _embedding_model = model_name

    # 加载或构建索引
    if not force_rebuild and os.path.exists(VECTOR_STORE_DIR) and os.path.exists(FULLTEXT_INDEX_DIR):
        logger.info("尝试从磁盘加载索引...")
        _vector_store = VectorStore()
        _fulltext_index = InvertedIndex()

        vs_loaded = _vector_store.load(VECTOR_STORE_DIR)
        ft_loaded = _fulltext_index.load(FULLTEXT_INDEX_DIR)

        if vs_loaded and ft_loaded:
            _chunks = _vector_store.documents
            _hybrid_search = HybridSearch(
                _vector_store, _fulltext_index,
                vector_weight=HYBRID_WEIGHTS["vector_weight"],
                fulltext_weight=HYBRID_WEIGHTS["fulltext_weight"],
            )
            _initialized = True
            logger.info(f"索引加载完成: {len(_chunks)} 文档")
            return True
        else:
            logger.info("索引文件不完整，重新构建...")

    # 处理 PDF
    logger.info("处理 PDF 文档...")
    chunks = process_pdfs(DATA_DIR)
    if not chunks:
        logger.warning("没有找到可处理的文档")
        return False
    _chunks = chunks
    logger.info(f"共 {len(chunks)} 个文档片段")

    # 生成嵌入向量
    logger.info(f"生成嵌入向量 (model={model_name})...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=model_name)
    logger.info(f"嵌入向量生成完成: {embeddings.shape}")

    # 构建向量索引
    logger.info("构建 FAISS 向量索引...")
    _vector_store = VectorStore()
    _vector_store.build(chunks, embeddings)
    _vector_store.save(VECTOR_STORE_DIR)

    # 构建倒排索引
    logger.info("构建倒排索引...")
    _fulltext_index = InvertedIndex()
    _fulltext_index.build(chunks)
    _fulltext_index.save(FULLTEXT_INDEX_DIR)

    # 构建混合检索器
    _hybrid_search = HybridSearch(
        _vector_store, _fulltext_index,
        vector_weight=HYBRID_WEIGHTS["vector_weight"],
        fulltext_weight=HYBRID_WEIGHTS["fulltext_weight"],
    )

    _initialized = True
    logger.info("RAG 系统初始化完成")
    return True


def query(question: str,
          search_type: str = "hybrid",
          hybrid_strategy: str = "weighted",
          ft_mode: str = "boolean",
          reranker_name: str = "llm",
          top_k: int = 10,
          embedding_model: Optional[str] = None) -> dict:
    """
    执行检索并生成回答

    参数:
        question: 用户问题
        search_type: "vector" | "fulltext" | "hybrid"
        hybrid_strategy: "weighted" | "rrf" | "cascade" (仅 hybrid 模式)
        ft_mode: "boolean" | "phrase" | "fuzzy" (仅 fulltext 模式)
        reranker_name: "llm" | "tfidf" | "adaptive" | "none"
        top_k: 返回结果数
        embedding_model: 嵌入模型名称

    返回:
        {"answer": "...", "sources": [...], "elapsed": 0.0, "retrieval_time": 0.0}
    """
    global _vector_store, _fulltext_index, _hybrid_search, _initialized, _embedding_model

    if not _initialized:
        return {"answer": "⚠️ 系统未初始化，请先初始化。", "sources": [], "elapsed": 0.0, "retrieval_time": 0.0}

    from embeddings import embed_query
    from rerankers import rerank_results

    start = time.time()

    # 生成查询向量
    model = embedding_model or _embedding_model
    query_vec = embed_query(question, model_name=model)

    # 检索
    retrieval_start = time.time()

    if search_type == "vector":
        raw_results = _vector_store.search(query_vec, top_k=top_k * 2)
    elif search_type == "fulltext":
        raw_results = _fulltext_index.search(question, mode=ft_mode, top_k=top_k * 2)
    else:  # hybrid
        raw_results = _hybrid_search.search(
            question, query_vec,
            strategy=hybrid_strategy,
            ft_mode=ft_mode,
            top_k=top_k * 2,
        )

    retrieval_time = time.time() - retrieval_start

    if not raw_results:
        return {
            "answer": "未在文档中找到相关信息，请尝试换个问法。",
            "sources": [],
            "elapsed": time.time() - start,
            "retrieval_time": retrieval_time,
        }

    # 重排
    if reranker_name != "none":
        rerank_start = time.time()
        # 解包 (doc, score) 元组，只传文档列表给重排器
        docs_to_rerank = [doc for doc, _ in raw_results]
        reranked = rerank_results(question, docs_to_rerank, reranker_name=reranker_name, top_k=top_k)
        # 重排器返回 [(doc, new_score), ...]
        raw_results = [(doc, score) for doc, score in reranked]
        logger.info(f"重排耗时: {time.time() - rerank_start:.3f}s")
    else:
        raw_results = raw_results[:top_k]

    # 提取文档列表（用于 LLM 上下文）
    docs = [doc for doc, score in raw_results]
    for doc, score in raw_results:
        doc["rerank_score"] = score if isinstance(score, (int, float)) else float(score)

    # 生成 LLM 回答
    from qa_engine import generate_answer

    answer = generate_answer(question, docs)

    elapsed = time.time() - start

    return {
        "answer": answer,
        "sources": [
            {
                "text": doc.get("text", "")[:200],
                "source": doc.get("source", "未知"),
                "score": doc.get("rerank_score", 0),
            }
            for doc in docs[:5]
        ],
        "elapsed": elapsed,
        "retrieval_time": retrieval_time,
        "total_results": len(raw_results),
    }


def get_db_stats() -> dict:
    """获取系统状态统计"""
    global _vector_store, _fulltext_index, _initialized, _chunks, _embedding_model

    from config import EMBEDDING_MODELS

    # 获取可用嵌入模型列表
    available_models = list(EMBEDDING_MODELS.keys())

    # 当前使用模型信息
    current_model_info = {}
    if _embedding_model and _embedding_model in EMBEDDING_MODELS:
        current_model_info = EMBEDDING_MODELS[_embedding_model]

    stats = {
        "initialized": _initialized,
        "chunks": len(_chunks) if _chunks else 0,
        "vector_dim": _vector_store.dimension if _vector_store and _vector_store._built else 0,
        "inverted_index_terms": len(_fulltext_index.inverted_index) if _fulltext_index and _fulltext_index._built else 0,
        "embedding_model": _embedding_model or "none",
        "current_model_info": current_model_info,
        "available_models": available_models,
    }
    return stats


def reset_conversation():
    """重置对话"""
    from qa_engine import reset_conversation as _reset
    _reset()


def switch_embedding_model(model_name: str) -> bool:
    """切换嵌入模型（需要重建索引）"""
    global _embedding_model
    from config import EMBEDDING_MODELS
    if model_name not in EMBEDDING_MODELS:
        logger.error(f"未知的嵌入模型: {model_name}")
        return False
    _embedding_model = model_name
    return True
