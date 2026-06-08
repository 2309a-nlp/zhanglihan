"""
QA 引擎 — LLM 问答 + 多轮对话管理
支持：DeepSeek/OpenAI API、多轮对话历史、语义缓存、流式输出
"""

import os
import time
import json
import base64
import logging
import urllib.request
from typing import List, Optional

from config import load_api_key

logger = logging.getLogger(__name__)


# ---------- 语义缓存 ----------
_SEMANTIC_CACHE = {}
_CACHE_MAX = 100


# ---------- 对话历史 ----------
_conversation_history = []
MAX_HISTORY = 6


# ---------- 熔断器 ----------
_CIRCUIT_BREAKER = {"failures": 0, "last_fail": 0, "open": False}
_CIRCUIT_THRESHOLD = 3
_CIRCUIT_RESET = 60

# ==================== 系统提示词 ====================

SYSTEM_PROMPT_TEMPLATE = """你是一个专业的 RAG 问答助手，精通中文和英文。请基于提供的文档内容回答问题。

## 回答规则
1. **准确性**：仅基于提供的文档内容回答，不要编造信息
2. **完整性**：如果文档包含多期数据，必须全部列出
3. **结构化**：用表格呈现多期数据，清晰美观
4. **简洁**：直接回答问题，不要赘述
5. **引用**：指出信息来源（文件名）
6. **语言**：用提问的语言回答
7. **透明**：如果文档中没有足够信息，说明缺少什么

## 对话历史
{history}

## 相关文档内容（重排后）
{context}

## 当前问题
{question}

## 回答：
"""


# ==================== API 调用 ====================
def _check_circuit() -> bool:
    """检查熔断器状态"""
    cb = _CIRCUIT_BREAKER
    if cb["open"]:
        if time.time() - cb["last_fail"] > _CIRCUIT_RESET:
            cb["open"] = False  # 半开
            return True
        return False
    return True


def _record_failure():
    """记录调用失败"""
    cb = _CIRCUIT_BREAKER
    cb["failures"] += 1
    cb["last_fail"] = time.time()
    if cb["failures"] >= _CIRCUIT_THRESHOLD:
        cb["open"] = True
        logger.warning("LLM 熔断器已打开")


def _record_success():
    """记录调用成功"""
    _CIRCUIT_BREAKER["failures"] = 0


def call_llm(messages: List[dict], provider: str = "deepseek",
             stream: bool = False, timeout: int = 40) -> str:
    """
    调用 LLM API

    参数:
        messages: 消息列表 [{"role": "user"/"assistant"/"system", "content": "..."}]
        provider: "deepseek" | "openai"
        stream: 是否流式输出
        timeout: 超时秒数

    返回:
        LLM 回复文本
    """
    from config import LLM_PROVIDERS

    if not _check_circuit():
        return "⚠️ 服务暂时不可用，请稍后再试。"

    config = LLM_PROVIDERS.get(provider)
    if not config:
        return f"未知的 LLM 提供者: {provider}"

    api_key = load_api_key()
    if not api_key:
        return "⚠️ 未配置 API Key，请在 .api_key 文件中设置。"

    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512,
        "stream": stream,
    }

    headers = {
        "Content-Type": "application/json"
    }

    if provider == "deepseek":
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            config["api_url"],
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream:
                # 流式读取
                full_text = ""
                for line in resp:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        chunk_str = line[6:]
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                        except json.JSONDecodeError:
                            continue
                _record_success()
                return full_text
            else:
                result = json.loads(resp.read().decode("utf-8"))
                _record_success()
                return result["choices"][0]["message"]["content"]

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        logger.error(f"LLM API HTTP 错误 {e.code}: {error_body}")
        _record_failure()
        return f"⚠️ API 调用失败 (HTTP {e.code})"
    except Exception as e:
        logger.error(f"LLM API 调用异常: {e}")
        _record_failure()
        return f"⚠️ 服务暂时不可用: {str(e)[:100]}"


# ==================== 多轮对话管理 ====================

def reset_conversation():
    """重置对话历史"""
    global _conversation_history
    _conversation_history = []


def get_history() -> str:
    """获取格式化的对话历史"""
    if not _conversation_history:
        return "暂无对话历史。"
    lines = []
    for msg in _conversation_history[-MAX_HISTORY:]:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines)


# ==================== 语义缓存 ====================

def _get_cache_key(question: str, top_k: int) -> str:
    """生成缓存 key"""
    return f"{question}:{top_k}"


def _check_cache(question: str, top_k: int) -> Optional[str]:
    """检查语义缓存"""
    key = _get_cache_key(question, top_k)
    return _SEMANTIC_CACHE.get(key)


def _set_cache(question: str, top_k: int, answer: str):
    """设置语义缓存"""
    key = _get_cache_key(question, top_k)
    if len(_SEMANTIC_CACHE) >= _CACHE_MAX:
        _SEMANTIC_CACHE.pop(next(iter(_SEMANTIC_CACHE)))
    _SEMANTIC_CACHE[key] = answer


# ==================== 主查询接口 ====================

def generate_answer(question: str, context_docs: List[dict],
                    provider: str = "deepseek") -> str:
    """
    生成回答

    参数:
        question: 用户问题
        context_docs: 检索到的上下文文档列表
        provider: LLM 提供者
    """
    # 检查缓存
    cached = _check_cache(question, len(context_docs))
    if cached:
        logger.info("语义缓存命中")
        return cached

    # 构建上下文
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        text = doc.get("text", doc.get("page_content", ""))[:500]
        source = doc.get("source", "未知")
        score = doc.get("rerank_score", doc.get("score", 0))
        context_parts.append(
            f"[来源 {i}] 文件: {source} | 相关度: {score:.4f}\n{text}\n"
        )
    context = "\n---\n".join(context_parts)

    # 构建对话历史
    history_lines = []
    for msg in _conversation_history[-MAX_HISTORY:]:
        role = "用户" if msg["role"] == "user" else "助手"
        history_lines.append(f"{role}: {msg['content'][:300]}")
    history = "\n".join(history_lines)

    # 构建 prompt
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        history=history,
        context=context,
        question=question,
    )

    messages = [{"role": "user", "content": prompt}]

    # 调用 LLM
    answer = call_llm(messages, provider=provider)

    # 更新对话历史
    _conversation_history.append({"role": "user", "content": question})
    _conversation_history.append({"role": "assistant", "content": answer})

    # 缓存答案
    _set_cache(question, len(context_docs), answer)

    return answer
