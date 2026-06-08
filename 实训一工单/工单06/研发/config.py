"""
RAG 多轮对话问答系统 — 配置文件
支持：向量检索 + 全文检索 + 混合检索 + 多种嵌入模型 + 多种重排算法
"""

import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
FULLTEXT_INDEX_DIR = os.path.join(BASE_DIR, "fulltext_index")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# ========== 向量检索配置 ==========
EMBEDDING_MODELS = {
    "bge-small-zh-v1.5": {
        "model_name": "BAAI/bge-small-zh-v1.5",
        "local_path": os.path.join(MODELS_DIR, "bge-small-zh-v1.5"),
        "dim": 512,
        "description": "轻量级中文嵌入模型，快速高效",
    },
    "m3e-base": {
        "model_name": "moka-ai/m3e-base",
        "local_path": os.path.join(MODELS_DIR, "m3e-base"),
        "dim": 768,
        "description": "M3E 基础版中文嵌入，语义理解强",
    },
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "local_path": os.path.join(MODELS_DIR, "bge-m3"),
        "dim": 1024,
        "description": "BGE-M3 多语言嵌入，支持稠密+稀疏+多向量",
    },
}
DEFAULT_EMBEDDING_MODEL = "bge-small-zh-v1.5"
VECTOR_TOP_K = 20
VECTOR_SIMILARITY_THRESHOLD = 0.3

# ========== 全文检索配置（倒排索引）==========
FULLTEXT_TOP_K = 20
FT_MODE_BOOLEAN = "boolean"
FT_MODE_PHRASE = "phrase"
FT_MODE_FUZZY = "fuzzy"
FT_MODE_DEFAULT = "boolean"

# ========== 混合检索配置 ==========
HYBRID_TOP_K = 10
HYBRID_WEIGHTS = {"vector_weight": 0.6, "fulltext_weight": 0.4}

# ========== 重排算法配置 ==========
RERANK_TOP_K = 10
RERANKER_MODEL_PATH = os.path.join(MODELS_DIR, "bge-reranker-v2-m3")
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
USE_LOCAL_RERANKER = True
ADAPTIVE_LEARNING_RATE = 0.1
ADAPTIVE_DECAY = 0.95
USER_FEEDBACK_DB = os.path.join(BASE_DIR, "user_feedback.json")

# ========== LLM 配置 ==========
LLM_PROVIDERS = {
    "deepseek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
    },
    "openai": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
}
DEFAULT_LLM = "deepseek"
LLM_TIMEOUT = 30

# ========== 文档处理 ==========
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MAX_FILE_SIZE_MB = 50

# ========== 会话 ==========
MAX_HISTORY_LENGTH = 6
RESPONSE_TIMEOUT = 3.0
CACHE_MAX_SIZE = 100

# API_KEY_FILE 直接硬编码在 load_api_key() 中


def load_api_key():
    if os.path.exists("C:\\Users\\ASUSTeK\\Desktop\\2309B\\工单\\工单06\\.api_key"):
        with open("C:\\Users\\ASUSTeK\\Desktop\\2309B\\工单\\工单06\\.api_key", "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    return os.environ.get("DEEPSEEK_API_KEY", "")
