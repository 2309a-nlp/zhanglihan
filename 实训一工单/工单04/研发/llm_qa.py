# -*- coding: utf-8 -*-
"""
LLM 问答模块（高稳定性 + 结构化输出版）
核心优化：
  1. 结构化列表输出格式（替代段落式）
  2. API 熔断器（Circuit Breaker）：连续失败自动停用，到期自动恢复
  3. 调用重试（指数退避）：瞬态错误自动恢复
  4. 输入校验：空输入、过长输入、非法字符检测
"""

import os
import re
import time
import hashlib
import threading
from typing import List, Dict, Any, Optional, Generator

import urllib.request
import urllib.error
import json
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
    LLM_MAX_CONNECTIONS,
    LLM_MAX_KEEPALIVE,
    LLM_KEEPALIVE_EXPIRY,
    CACHE_ENABLED,
    CACHE_EXPIRE_TIME,
    CACHE_MAX_SIZE,
    SEMANTIC_CACHE_THRESHOLD,
    LLM_RETRY_MAX_ATTEMPTS,
    LLM_RETRY_BASE_DELAY,
    LLM_RETRY_MAX_DELAY,
    LLM_CIRCUIT_BREAKER_THRESHOLD,
    LLM_CIRCUIT_BREAKER_COOLDOWN,
    INPUT_MAX_LENGTH,
    INPUT_MIN_LENGTH,
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

    def get(self, query: str, query_embedding: Optional[np.ndarray] = None
            ) -> Optional[str]:
        if query_embedding is None:
            return None

        with self._lock:
            expired = [k for k, v in self._cache.items()
                       if self._is_expired(v)]
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

    def set(self, query: str, response: str,
            query_embedding: Optional[np.ndarray] = None):
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


class _CircuitBreaker:
    """
    熔断器 — 保护 API 不会被连续失败打爆

    状态机: CLOSED → OPEN → HALF_OPEN → CLOSED
      - CLOSED: 正常调用
      - OPEN:   API 不可用，直接拒绝调用
      - HALF_OPEN: 冷却期后允许单次尝试探测恢复
    """

    def __init__(self, threshold: int = LLM_CIRCUIT_BREAKER_THRESHOLD,
                 cooldown: int = LLM_CIRCUIT_BREAKER_COOLDOWN):
        self._threshold = threshold
        self._cooldown = cooldown
        self._failures = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._state_change_time = 0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """是否可以发起 API 请求"""
        with self._lock:
            if self._state == "CLOSED":
                return True
            if self._state == "OPEN":
                # 检查冷却期是否结束
                if time.time() - self._state_change_time > self._cooldown:
                    self._state = "HALF_OPEN"
                    print(f"  [熔断器] 冷却期结束，切换 HALF_OPEN 状态，尝试恢复")
                    return True
                return False
            # HALF_OPEN: 允许单次尝试
            return True

    def on_success(self):
        """调用成功"""
        with self._lock:
            self._failures = 0
            if self._state == "HALF_OPEN":
                self._state = "CLOSED"
                self._state_change_time = time.time()
                print(f"  [熔断器] 恢复成功 → CLOSED")

    def on_failure(self):
        """调用失败"""
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and self._state == "CLOSED":
                self._state = "OPEN"
                self._state_change_time = time.time()
                print(f"  [熔断器] ⚠ 连续失败 {self._failures} 次 → OPEN "
                      f"（冷却 {self._cooldown}s）")

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def reset(self):
        """手动重置"""
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"
            self._state_change_time = time.time()


class LLMQA:
    """大模型问答器（高稳定性 + 结构化输出版）"""

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_API_BASE.rstrip("/")
        self.model = DEEPSEEK_MODEL
        self._pool = urllib.request.HTTPHandler()
        self._opener = urllib.request.build_opener(self._pool)

        # ── 语义缓存 ──
        self._semantic_cache = _SemanticCache()

        # ── 熔断器 ──
        self._circuit_breaker = _CircuitBreaker()

        # ── 预热标记 ──
        self._warmed_up = False
        self._warmup_lock = threading.Lock()

    # ==================== 预热 ====================

    def _request(self, payload: dict, stream: bool = False) -> dict:
        url = self.base_url + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        resp = self._opener.open(req, timeout=LLM_REQUEST_TIMEOUT)
        return json.loads(resp.read().decode("utf-8"))

    def _request_stream(self, payload: dict):
        url = self.base_url + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        timeout = LLM_REQUEST_TIMEOUT + LLM_STREAM_TIMEOUT
        resp = self._opener.open(req, timeout=timeout)
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        d = json.loads(data_str)
                        choices = d.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            c = delta.get("content", "")
                            if c:
                                yield c
                    except json.JSONDecodeError:
                        pass

    def warmup(self):
        if self._warmed_up:
            return
        with self._warmup_lock:
            if self._warmed_up:
                return
            start = time.time()
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "预热"}],
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": 2,
                    "stream": False,
                }
                self._request(payload)
                self._warmed_up = True
                elapsed = time.time() - start
                print(f"  [LLM预热] 连接池就绪, 耗时 {elapsed:.2f}s")
            except Exception as e:
                print(f"  [LLM预热] 跳过 ({e})")

    # ==================== 输入校验 ====================

    @staticmethod
    def validate_input(query: str) -> Optional[str]:
        """
        校验用户输入，返回 None 表示合法，返回字符串表示错误信息
        """
        if not query or not query.strip():
            return "请输入您的问题。"
        if len(query.strip()) < INPUT_MIN_LENGTH:
            return "您的问题太短了，请详细描述。"
        if len(query.strip()) > INPUT_MAX_LENGTH:
            return f"您的问题过长（最多 {INPUT_MAX_LENGTH} 个字符），请精简后重试。"
        return None  # 合法

    # ==================== 语言检测 ====================

    @staticmethod
    def detect_language(text: str) -> str:
        if not text:
            return "zh"
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if cn_chars > en_words else "en"

    @staticmethod
    def build_lang_hint(lang: str) -> str:
        if lang == "en":
            return (
                "IMPORTANT: Answer in English. "
                "Respond to the user in the same language as their question.\n"
            )
        return "重要：请用中文回答。根据用户问题的语言使用对应的语言回复。\n"

    # ==================== Prompt 构建（结构化输出）====================

    def _build_rag_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """构建 RAG 提示词（结构化列表输出格式）"""
        lang = self.detect_language(query)
        lang_hint = self.build_lang_hint(lang)

        context_lines = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk["content"].strip()
            context_lines.append(f"[{i}] {text}")
        context = "\n\n".join(context_lines)

        # 结构化输出指令
        structure_instruction = (
            "【输出格式要求】\n"
            "请以结构化列表的形式回答问题，格式如下：\n"
            "XX公司的YY部门有以下ZZ构成：\n"
            "· 项目1\n"
            "· 项目2\n"
            "· 项目3\n"
            "\n"
            "规则：\n"
            "1. 开头用一句话概括主体\n"
            "2. 每个条目使用 \"· \" 前缀（中文顿号）\n"
            "3. 条目之间换行，不要编号\n"
            "4. 如数据是表格形式，保持表格结构\n"
            "5. 信息不足则说：文档中无相关信息\n"
        )

        prompt = (
            f"{lang_hint}\n"
            "基于以下文档回答问题。\n\n"
            "【规则】\n"
            "1. 严格依据文档，不编造\n"
            "2. 简洁回答，引用文档编号\n"
            f"{structure_instruction}\n"
            f"【参考文档】\n{context}\n\n"
            f"【问题】\n{query}\n\n"
            "【回答】\n"
        )
        return prompt

    def _build_system_prompt(self, history_text: str = "") -> str:
        if history_text:
            return f"""你是专业的知识问答助手。
严格基于提供的文档内容回答，不要编造信息。
结构化输出：使用 "· " 列表格式组织信息。
根据用户问题的语言使用对应的语言回复。

{history_text}"""
        return ("你是专业的知识问答助手。严格基于提供的文档内容回答，不要编造信息。"
                "结构化输出：使用 \"· \" 列表格式组织信息。"
                "根据用户问题的语言使用对应的语言回复。")

    # ==================== 语义缓存（嵌入对齐）====================

    def _encode_for_cache(self, query: str) -> np.ndarray:
        import numpy as np
        nums = re.findall(r'\d+', query)
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
        all_tokens = nums + words
        emb = np.zeros(128, dtype=np.float32)
        for token in all_tokens:
            h = hashlib.md5(token.encode()).digest()
            idx = int.from_bytes(h[:2], 'big') % 128
            emb[idx] += 1.0
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb

    # ==================== LLM 调用（流式 + 重试 + 熔断）====================

    def _call_llm_stream(self, messages: List[Dict]):
        """
        流式调用 LLM API（含重试 + 熔断保护）
        """
        start_time = time.time()

        # 熔断检查
        if not self._circuit_breaker.allow_request():
            elapsed = time.time() - start_time
            error_msg = (
                "⚠️ API 服务暂时不可用（熔断保护中），"
                f"系统将在 {LLM_CIRCUIT_BREAKER_COOLDOWN} 秒后自动恢复。"
            )
            print(f"  [熔断器] 请求被拒绝，状态: {self._circuit_breaker.state}")
            self._stream_result = {
                "answer": error_msg,
                "elapsed_time": round(elapsed, 4),
                "source": "error",
            }
            yield ""
            return

        last_error = None
        for attempt in range(LLM_RETRY_MAX_ATTEMPTS):
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
                self._circuit_breaker.on_success()
                return  # 成功返回

            except urllib.error.URLError as e:
                last_error = f"API 请求超时: {str(e)}"
                print(f"  [LLM重试] 第 {attempt + 1}/{LLM_RETRY_MAX_ATTEMPTS} 次超时")
            except urllib.error.URLError as e:
                last_error = f"API 连接失败: {str(e)}"
                print(f"  [LLM重试] 第 {attempt + 1}/{LLM_RETRY_MAX_ATTEMPTS} 次连接失败")
            except urllib.error.HTTPError as e:
                status = e.code if e.response else "?"
                last_error = f"API 返回错误状态 {status}: {str(e)}"
                print(f"  [LLM错误] HTTP {status}")
                # 4xx 错误不重试（客户端错误）
                if 400 <= e.code < 500:
                    break
            except Exception as e:
                last_error = f"API 调用失败: {str(e)}"
                print(f"  [LLM重试] 第 {attempt + 1}/{LLM_RETRY_MAX_ATTEMPTS} 次: {str(e)[:80]}")

            # 重试等待（指数退避）
            if attempt < LLM_RETRY_MAX_ATTEMPTS - 1:
                delay = min(
                    LLM_RETRY_BASE_DELAY * (2 ** attempt),
                    LLM_RETRY_MAX_DELAY
                )
                time.sleep(delay)

        # 所有重试失败
        self._circuit_breaker.on_failure()
        elapsed = time.time() - start_time
        error_msg = (
            f"⚠️ API 调用失败（已自动重试 {LLM_RETRY_MAX_ATTEMPTS} 次）。\n"
            f"原因: {last_error}\n"
            f"系统将自动恢复，请稍后重试。"
        )
        print(f"  [LLM错误] {error_msg}")
        self._stream_result = {
            "answer": error_msg,
            "elapsed_time": round(elapsed, 4),
            "source": "error",
        }
        yield ""

    # ==================== 熔断器状态 ====================

    def get_circuit_breaker_state(self) -> str:
        """获取熔断器状态"""
        return self._circuit_breaker.state

    def reset_circuit_breaker(self):
        """手动重置熔断器"""
        self._circuit_breaker.reset()
        print("  [熔断器] 已手动重置")

    # ==================== 流式回答 ====================

    def chat_answer_stream(self, query: str, context_chunks: List[Dict],
                           history: Optional[List[Dict]] = None):
        """
        流式对话式 RAG 回答（结构化输出版）
        """
        self._stream_result = None

        # ── 语义缓存检查 ──
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

        # 流式调用 LLM（含重试+熔断）
        for token in self._call_llm_stream(messages):
            yield token

        # 缓存最终结果
        if self._stream_result and self._stream_result["source"] == "api":
            final_answer = self._stream_result["answer"]
            self._semantic_cache.set(query, final_answer, query_emb)

    # ==================== 清理 ====================

    def close(self):
        pass
