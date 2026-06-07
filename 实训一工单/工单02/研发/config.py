# -*- coding: utf-8 -*-
"""
工单02 系统配置（速度优化版）
"""

import os

# API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-e518151a7d6a4f63b00ada8fe38211ae")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 嵌入模型配置
EMBEDDING_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bge-small-zh-v1.5")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512

# 向量数据库配置
VECTOR_STORE_PATH = "./vector_store/faiss_index"

# 文档处理配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 3

# LLM 配置（速度优化核心）
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 64
LLM_CONTEXT_WINDOW = 2

# LLM 超时设置
LLM_REQUEST_TIMEOUT = 8.0
LLM_STREAM_TIMEOUT = 10.0

# HTTP 连接池
LLM_MAX_CONNECTIONS = 20
LLM_MAX_KEEPALIVE = 10
LLM_KEEPALIVE_EXPIRY = 60.0

# 缓存配置
CACHE_ENABLED = True
CACHE_EXPIRE_TIME = 1800
CACHE_MAX_SIZE = 256
SEMANTIC_CACHE_THRESHOLD = 0.92

# 二分路检索配置
BATCH_SIZE = 32
HYBRID_ALPHA = 0.6

# MySQL 存储配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "pdf_qa")
MYSQL_POOL_MIN = 2
MYSQL_POOL_MAX = 5

# 路径配置
DOCS_DIR = "./docs"
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)