# -*- coding: utf-8 -*-
"""
System configuration module
"""

import os
from dotenv import load_dotenv

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(_CONFIG_DIR, ".env")
load_dotenv(dotenv_path)

# == API Configuration ==
# DeepSeek API configuration (reads from .env with fallback)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# == Embedding Model Configuration ==
# Use local model under project directory (no download needed)
EMBEDDING_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bge-small-zh-v1.5")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512

# == Vector Store Configuration ==
VECTOR_STORE_PATH = "./vector_store/faiss_index"

# == Document Processing Configuration ==
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 3

# == LLM Configuration ==
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
LLM_CONTEXT_WINDOW = 2

# == Performance and Retrieval Configuration ==
CACHE_ENABLED = True
CACHE_EXPIRE_TIME = 1800
BATCH_SIZE = 32

# Hybrid retrieval weights (FAISS semantic + BM25 keyword)
HYBRID_ALPHA = 0.6

# == MySQL Storage Configuration ==
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "pdf_qa")

# == Path Configuration ==
DOCS_DIR = "./docs"
UPLOAD_DIR = "./uploads"

# Ensure directories exist
os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
