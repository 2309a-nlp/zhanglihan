# -*- coding: utf-8 -*-
# 全局配置
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """全局配置"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 服务配置
    HOST: str = "127.0.0.1"
    PORT: int = 8001

    # DeepSeek API 配置（主 LLM 后端）
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL_NAME = "deepseek-chat"

        # LLM 后端选择: "deepseek_api"（默认）| "ollama" | "vllm" | "sglang"
    LLM_BACKEND = os.getenv("LLM_BACKEND", "deepseek_api").strip().lower()

    # vLLM 配置（WSL2）
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://172.30.214.232:8001")
    VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "deepseek-chat") # Qwen2.5-1.5B

    # SGLang 配置
    SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:30000")
    SGLANG_MODEL_NAME = os.getenv("SGLANG_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

    # Ollama 配置（本地）
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")

    # 前端静态文件路径
    STATIC_DIR = os.path.join(BASE_DIR, "dist")

    # 重排序配置
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").strip().lower() == "true"
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))

    # RAG 检索配置
    RAG_RETRIEVAL_K: int = int(os.getenv("RAG_RETRIEVAL_K", "5"))

    # 相似度阈值
    _st = os.getenv("RAG_SIMILARITY_THRESHOLD", "0.65").strip()
    RAG_SIMILARITY_THRESHOLD: float = float(_st) if _st else 0.65


settings = Settings()

