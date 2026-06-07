# -*- coding: utf-8 -*-
"""
LLM 问答模块（urllib 替代 httpx 版）
因本机 httpx 0.28.1 SSL 握手失败，改用 urllib.request
"""

import os
import re
import json
import time
import hashlib
import urllib.request
import threading
from typing import List, Dict, Any, Optional, Generator

import numpy as np

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_BASE,
    DEEPSEEK_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_CONTEXT_WINDOW,
    LLM_REQUEST_TIMEOUT,
    LLM_STREAM_TIMEOUT,
    CACHE_ENABLED,
    CACHE_EXPIRE_TIME,
    CACHE_MAX_SIZE,
    SEMANTIC_CACHE_THRESHOLD,
)


class _SemanticCache:
    """语义缓存（LRU + 余弦相似度）"""
    def __init__(self, max_size: int = CACHE_MAX_SIZE,
                 threshold: float = SEMANTIC_CACHE_THRESHOLD,
                 expire_time: int = CACHE_EXPIRE_TIME):
        self._max_size = max_size
        self._threshold = threshold
        self._expire_time = expire_time
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict] = {}
        self._order: List[str] = []

    def _is_expired(self, entry: Dict) -> bool:
        return (time.time() - entry["timestamp"]) > self._expire_time

    def _evict_lru(self):
        while len(self._cache) > self._max_size and self._order:
            oldest = self._order.pop(0)
            del self._cache[oldest]

    def get(self, query: str, query_embedding: Optional[np.ndarray] = None) -> Optional[str]:
        if query_embedding is None:
            return None
        with self._lock:
            expired = [k for k, v in self._cache.items() if self._is_expired(v)]
            for k in expired:
                del self._cache[k]
                if k in self._order:
                    self._order.remove(k)
            best_key = None
            best_sim = 0.0
            for key, entry in self._cache.items():
                emb = entry.get("embedding")
                if emb is None:
                    continue
                sim = float(np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
            if best_key is not None and best_sim >= self._threshold:
                if best_key in self._order:
                    self._order.remove(best_key)
                self._order.append(best_key)
                return self._cache[best_key]["response"]
        return None

    def set(self, query: str, response: str, query_embedding: Optional[np.ndarray] = None):
        if not CACHE_ENABLED:
            return
        with self._lock:
            key = hashlib.md5(query.encode()).hexdigest()
            entry = {
                "response": response,
                "embedding": query_embedding,
                "timestamp": time.time(),
            }
            self._cache[key] = entry
            if key not in self._order:
                self._order.append(key)
            self._evict_lru()

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)



class LLMQA:
    """大模型问答器（urllib 版，绕过 httpx SSL 问题）"""

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_API_BASE.rstrip("/")
        self.model = DEEPSEEK_MODEL
        self._semantic_cache = _SemanticCache()
        self._warmed_up = False
        self._warmup_lock = threading.Lock()
        # 连接池：复用 urllib 的 HTTPConnectionPool (keep-alive)
        self._pool = urllib.request.HTTPHandler()
        # 构建 opener 支持 keep-alive
        self._opener = urllib.request.build_opener(self._pool)

    def _request(self, payload: dict, stream: bool = False) -> dict:
        """发送非流式请求到 DeepSeek API"""
        url = self.base_url + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/event-stream" if stream else "application/json",
            },
            method="POST",
        )
        resp = self._opener.open(req, timeout=LLM_REQUEST_TIMEOUT)
        resp_data = resp.read().decode("utf-8")
        return json.loads(resp_data)

    def _request_stream(self, payload: dict):
        """发送流式请求，逐 token 解析 SSE"""
        url = self.base_url + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        resp = self._opener.open(req, timeout=LLM_REQUEST_TIMEOUT + LLM_STREAM_TIMEOUT)
        # 逐行解析 SSE
        buffer = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        pass

    # ==================== 预热 ====================

    def warmup(self):
        if self._warmed_up:
            return
        with self._warmup_lock:
            if self._warmed_up:
                return
            start = time.time()
            try:
                self._request({
                    "model": self.model,
                    "messages": [{"role": "user", "content": "预热"}],
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": 2,
                    "stream": False,
                })
                self._warmed_up = True
                elapsed = time.time() - start
                print(f"  [LLM预热] 连接就绪, 耗时 {elapsed:.2f}s")
            except Exception as e:
                print(f"  [LLM预热] 跳过 ({e})")

    # ==================== 语言检测 ====================

    @staticmethod
    def detect_language(text: str) -> str:
        if not text:
            return "zh"
        cn_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        en_words = len(re.findall(r"[a-zA-Z]+", text))
        return "zh" if cn_chars > en_words else "en"

    @staticmethod
    def build_lang_hint(lang: str) -> str:
        if lang == "en":
            return (
                "IMPORTANT: Answer in English. "
                "Respond to the user in the same language as their question.\n"
            )
        return "重要：请用中文回答。根据用户问题的语言使用对应的语言回复。\n"

    # ==================== Prompt 构建 ====================

    def _build_rag_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        lang = self.detect_language(query)
        lang_hint = self.build_lang_hint(lang)
        context_lines = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk["content"].strip()
            context_lines.append(f"[{i}] {text}")
        context = "\n\n".join(context_lines)
        prompt = (
            f"{lang_hint}\n"
            "基于以下文档回答问题。\n\n"
            "【规则】\n"
            "1. 严格依据文档，不编造\n"
            "2. 信息不足则说：文档中无相关信息\n"
            "3. 简洁回答，引用文档编号\n\n"
            f"【参考文档】\n{context}\n\n"
            f"【问题】\n{query}\n\n"
            "【回答】\n"
        )
        return prompt

    def _build_system_prompt(self, history_text: str = "") -> str:
        if history_text:
            return f"你是专业的知识问答助手。\n严格基于提供的文档内容回答，不要编造信息。\n根据用户问题的语言使用对应的语言回复。\n\n{history_text}"
        return "你是专业的知识问答助手。严格基于提供的文档内容回答，不要编造信息。根据用户问题的语言使用对应的语言回复。"

    # ==================== 语义缓存 ====================

    def _encode_for_cache(self, query: str) -> np.ndarray:
        nums = re.findall(r"\d+", query)
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
        all_tokens = nums + words
        emb = np.zeros(128, dtype=np.float32)
        for token in all_tokens:
            h = hashlib.md5(token.encode()).digest()
            idx = int.from_bytes(h[:2], "big") % 128
            emb[idx] += 1.0
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb

    # ==================== LLM 调用（流式）====================

    def _call_llm_stream(self, messages: List[Dict]):
        start_time = time.time()
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
                "stream": True,
            }
            full_answer = ""
            for token in self._request_stream(payload):
                full_answer += token
                yield token

            elapsed = time.time() - start_time
            self._stream_result = {
                "answer": full_answer.strip(),
                "elapsed_time": round(elapsed, 4),
                "source": "api",
            }
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"API 调用失败: {str(e)}"
            print(f"  [LLM错误] {error_msg}")
            self._stream_result = {
                "answer": error_msg,
                "elapsed_time": round(elapsed, 4),
                "source": "error",
            }
            yield ""

    # ==================== 流式回答 ====================

    def chat_answer_stream(self, query: str, context_chunks: List[Dict],
                           history: Optional[List[Dict]] = None):
        self._stream_result = None

        if CACHE_ENABLED:
            query_emb = self._encode_for_cache(query)
            cached = self._semantic_cache.get(query, query_emb)
            if cached:
                self._stream_result = {
                    "answer": cached,
                    "elapsed_time": 0,
                    "source": "semantic_cache",
                }
                yield cached
                return

        prompt = self._build_rag_prompt(query, context_chunks)
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for h in history[-(LLM_CONTEXT_WINDOW * 2):]:
                messages.append(h)

        messages.append({"role": "user", "content": prompt})

        for token in self._call_llm_stream(messages):
            yield token

        if self._stream_result and self._stream_result["source"] == "api":
            final_answer = self._stream_result["answer"]
            self._semantic_cache.set(query, final_answer, query_emb)

    def close(self):
        """urllib 自动管理连接，无需显式关闭"""
        pass
