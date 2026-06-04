# -*- coding: utf-8 -*-
"""
向量数据库模块（支持多角色）
每个角色（Medical, Finance, Law, Education, Psychology）有独立的 FAISS + BM25 索引
通过 load_role_index(role_name) 动态加载指定角色的索引
"""
import os
import pickle
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLES_DIR = os.path.join(CURRENT_DIR, "roles")

# 支持的角色列表
ALL_ROLES = ["Medical", "Finance", "Law", "Education", "Psychology"]

# 全局状态
_current_role = None
vectorstore = None
faiss_index_ready = False
bm25_model = None
text_chunks = []
is_hybrid_ready = False


def _get_role_index_dir(role_name: str) -> str:
    return os.path.join(ROLES_DIR, role_name)


def _get_embeddings_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _backend_dir = os.path.dirname(CURRENT_DIR)
    _model_path = os.path.join(_backend_dir, "bge-m3")
    return HuggingFaceEmbeddings(
        model_name=_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_role_index(role_name: str) -> bool:
    """加载指定角色的 FAISS + BM25 索引"""
    global _current_role, vectorstore, faiss_index_ready
    global bm25_model, text_chunks, is_hybrid_ready

    role_index_dir = _get_role_index_dir(role_name)
    faiss_path = os.path.join(role_index_dir, "index.faiss")
    bm25_path = os.path.join(role_index_dir, "bm25.pkl")
    chunks_path = os.path.join(role_index_dir, "chunks.pkl")

    if _current_role == role_name and faiss_index_ready:
        logger.info(f"[{role_name}] 索引已加载")
        return True

    if not os.path.exists(faiss_path):
        logger.warning(f"[{role_name}] 索引不存在，请先运行 build_index_v2.py")
        _current_role = None
        vectorstore = None
        faiss_index_ready = False
        bm25_model = None
        text_chunks = []
        is_hybrid_ready = False
        return False

    try:
        from langchain_community.vectorstores import FAISS
        embeddings_model = _get_embeddings_model()

        vectorstore = FAISS.load_local(
            role_index_dir, embeddings_model,
            allow_dangerous_deserialization=True
        )
        with open(bm25_path, "rb") as f:
            bm25_model = pickle.load(f)
        with open(chunks_path, "rb") as f:
            text_chunks = pickle.load(f)

        _current_role = role_name
        faiss_index_ready = True
        is_hybrid_ready = True
        logger.info(f"[{role_name}] 索引加载成功 ({len(text_chunks)} chunks)")
        return True
    except Exception as e:
        logger.error(f"[{role_name}] 加载失败: {e}")
        _current_role = None
        vectorstore = None
        faiss_index_ready = False
        bm25_model = None
        text_chunks = []
        is_hybrid_ready = False
        return False


def similarity_search_with_scores(query: str, k: int = 5) -> List[Tuple]:
    """对当前加载的角色索引执行相似度搜索"""
    if not faiss_index_ready or vectorstore is None:
        logger.warning("FAISS 索引未就绪")
        return []
    try:
        return vectorstore.similarity_search_with_score(query, k=k)
    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []


def get_current_role() -> Optional[str]:
    return _current_role


def is_index_ready() -> bool:
    return faiss_index_ready


__all__ = [
    "similarity_search_with_scores",
    "faiss_index_ready",
    "bm25_model",
    "text_chunks",
    "is_hybrid_ready",
    "ALL_ROLES",
    "load_role_index",
    "get_current_role",
    "is_index_ready",
]
