"""
嵌入模型模块 — 支持 bge-small-zh-v1.5 / m3e-base 等模型
支持切换嵌入模型、批量编码、归一化
"""

import os
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# 全局缓存
_EMBEDDING_MODELS = {}

def _get_model(model_name: str):
    """延迟加载嵌入模型"""
    global _EMBEDDING_MODELS
    from config import EMBEDDING_MODELS, MODELS_DIR

    if model_name in _EMBEDDING_MODELS:
        return _EMBEDDING_MODELS[model_name]

    cfg = EMBEDDING_MODELS.get(model_name)
    if not cfg:
        raise ValueError(f"未知的嵌入模型: {model_name}，可用: {list(EMBEDDING_MODELS.keys())}")

    local_path = cfg["local_path"]
    huggingface_name = cfg["model_name"]

    logger.info(f"加载嵌入模型: {model_name} ({huggingface_name})...")

    if os.path.exists(local_path):
        model_path = local_path
    else:
        model_path = huggingface_name

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, trust_remote_code=True)
        _EMBEDDING_MODELS[model_name] = model
        logger.info(f"嵌入模型加载完成: {model_name}, dim={cfg['dim']}")
        return model
    except Exception as e:
        logger.error(f"加载嵌入模型失败: {e}")
        raise


def embed_texts(texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
    """
    批量编码文本为向量

    参数:
        texts: 文本列表
        model_name: 嵌入模型名称，默认使用 DEFAULT_EMBEDDING_MODEL

    返回:
        numpy array, shape=(n, dim)
    """
    from config import DEFAULT_EMBEDDING_MODEL
    if model_name is None:
        model_name = DEFAULT_EMBEDDING_MODEL

    model = _get_model(model_name)

    # 对于 bge 模型，需要添加 instruction prefix
    if "bge" in model_name:
        texts = [f"为这句话生成表示以用于检索相关文章：{t}" if t else t for t in texts]

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32,
    )
    return np.array(embeddings)


def embed_query(query: str, model_name: Optional[str] = None) -> np.ndarray:
    """编码单个查询"""
    return embed_texts([query], model_name=model_name)[0]


def get_embedding_dim(model_name: Optional[str] = None) -> int:
    """获取嵌入维度"""
    from config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL
    if model_name is None:
        model_name = DEFAULT_EMBEDDING_MODEL
    return EMBEDDING_MODELS[model_name]["dim"]


def get_available_models() -> list:
    """获取可用嵌入模型列表"""
    from config import EMBEDDING_MODELS
    return list(EMBEDDING_MODELS.keys())

