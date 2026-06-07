# -*- coding: utf-8 -*-
"""
工单03 — 系统配置（速度优化 + 水印处理版）
核心目标：首次问答全链路 < 3s，高并发下稳定运行
新增特性：PDF水印检测与自动清洗
"""

import os

# ==================== API 配置 ====================

def _load_api_key() -> str:
    """从 .api_key 文件或环境变量读取 API key"""
    key_file = os.path.join(os.path.dirname(__file__), ".api_key")
    if os.path.exists(key_file):
        import base64
        with open(key_file, "r") as f:
            raw_key = f.read().strip()
        try:
            return base64.b64decode(raw_key).decode("utf-8")
        except Exception:
            return raw_key
    return os.getenv("DEEPSEEK_API_KEY", "")

DEEPSEEK_API_KEY = _load_api_key()

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ==================== 嵌入模型配置 ====================
EMBEDDING_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bge-small-zh-v1.5")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512

# ==================== 向量数据库配置 ====================
VECTOR_STORE_PATH = "./vector_store/faiss_index"

# ==================== 文档处理配置 ====================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 3

# ==================== 水印处理配置 ====================
# 是否启用 PDF 水印自动检测和清洗
WATERMARK_REMOVAL_ENABLED = True
# 水印清洗模式：
#   "aggressive"  — 激进：检测页眉/页脚/重复背景文字，并过滤水印页
#   "conservative" — 保守：仅清理明显的水印文字（如"水印"字样）
WATERMARK_CLEAN_MODE = "aggressive"
# 水印过滤阈值：当某页非水印内容占比低于此值时，判定为水印页
WATERMARK_PAGE_THRESHOLD = 0.3
# 已知水印关键词（大小写不敏感）
WATERMARK_KEYWORDS = [
    "水印", "watermark", "draft", "confidential",
    "仅供内部", "仅供预览", "预先披露", "申报稿",
    "sample", "草稿", "机密", "测试页", "样板",
]

# ==================== LLM 配置（速度优化核心）====================
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 64
LLM_CONTEXT_WINDOW = 2

# LLM 超时设置（单位：秒）
LLM_REQUEST_TIMEOUT = 8.0
LLM_STREAM_TIMEOUT = 10.0

# ==================== HTTP 连接池（减少 TCP 握手）====================
LLM_MAX_CONNECTIONS = 20
LLM_MAX_KEEPALIVE = 10
LLM_KEEPALIVE_EXPIRY = 60.0

# ==================== 缓存配置 ====================
CACHE_ENABLED = True
CACHE_EXPIRE_TIME = 1800
CACHE_MAX_SIZE = 256
SEMANTIC_CACHE_THRESHOLD = 0.92

# ==================== 二分路检索配置 ====================
BATCH_SIZE = 32
HYBRID_ALPHA = 0.6

# ==================== MySQL 存储配置 ====================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "pdf_qa")

MYSQL_POOL_MIN = 2
MYSQL_POOL_MAX = 5

# ==================== 路径配置 ====================
DOCS_DIR = "./docs"
UPLOAD_DIR = "./uploads"

os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== Stability Config ====================

# --- API Retry & Circuit Breaker ---
LLM_RETRY_MAX_ATTEMPTS = 3
LLM_RETRY_BASE_DELAY = 1.0
LLM_RETRY_MAX_DELAY = 8.0
LLM_CIRCUIT_BREAKER_THRESHOLD = 5
LLM_CIRCUIT_BREAKER_COOLDOWN = 30

# --- Health Check ---
HEALTH_CHECK_INTERVAL = 30
HEALTH_CHECK_TIMEOUT = 5.0

# --- DB Reconnect ---
DB_RECONNECT_MAX_ATTEMPTS = 3
DB_RECONNECT_INTERVAL = 2.0

# --- Input Validation ---
INPUT_MAX_LENGTH = 500
INPUT_MIN_LENGTH = 1

# --- Auto Recovery ---
INDEX_AUTO_REBUILD = True
AUTO_RECOVERY_INTERVAL = 60
