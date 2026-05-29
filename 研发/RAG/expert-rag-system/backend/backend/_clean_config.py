# -*- coding: utf-8 -*-
import re
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')

with open(p, 'r', encoding='utf-8') as f:
    c = f.read()

# Remove DashScope section
c = re.sub(
    r'    # 👇 通义千问 \(DashScope\) 配置\n    # API Key.*?\n    DASHSCOPE_API_KEY = os\.getenv\("DASHSCOPE_API_KEY"\)\n',
    '', c
)

# Remove vLLM section
c = re.sub(
    r'    # 👇 vLLM 配置.*?\n    VLLM_BASE_URL.*?\n    VLLM_MODEL_NAME.*?\n',
    '', c
)

# Remove QWEN_MODEL_NAME and DASHSCOPE_EMBEDDING_MODEL
c = re.sub(r'    QWEN_MODEL_NAME = "qwen-plus"\n', '', c)
c = re.sub(r'    DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v1"\n', '', c)

# Remove RAG_FAISS_MAX_DISTANCE section
c = re.sub(
    r'    # FAISS 向量检索的最大距离阈值.*?\n    _md = os\.getenv\("RAG_FAISS_MAX_DISTANCE".*?\n    RAG_FAISS_MAX_DISTANCE.*?\n',
    '', c
)

# Clean up LLM_BACKEND comment
c = c.replace(
    '    # 可选值: "deepseek_api"（默认，远程API）, "ollama"（Windows本地Ollama）, "vllm"（WSL2 vLLM）',
    '    # 可选值: "deepseek_api"（默认）| "ollama"（本地）'
)

# Remove test comment
c = c.replace(
    '    LLM_BACKEND = os.getenv("LLM_BACKEND", "deepseek_api").strip().lower()\n    # 测试Ollama本地模型 解开这个注释\n    # LLM_BACKEND = "ollama"',
    '    LLM_BACKEND = os.getenv("LLM_BACKEND", "deepseek_api").strip().lower()'
)

# Remove comment about RAG_SIMILARITY_THRESHOLD mentioning old 0.50
c = c.replace(
    '    # 注意：rag_service.py 中实际使用 0.50（L2≈1.0）作为语义相似度的硬阈值\n',
    ''
)

with open(p, 'w', encoding='utf-8') as f:
    f.write(c)

print("Config cleaned successfully!")
