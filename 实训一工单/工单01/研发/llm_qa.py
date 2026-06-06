# -*- coding: utf-8 -*-
"""
LLM 问答模块
负责调用大模型进行基于 RAG 的对话问答
支持中英文双语、对话历史、缓存优化
"""

import os
import re
import time
import hashlib
from typing import List, Dict, Any, Optional

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_BASE,
    DEEPSEEK_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_CONTEXT_WINDOW,
    CACHE_ENABLED,
    CACHE_EXPIRE_TIME,
)


class LLMQA:
    """大模型问答器（对话式 RAG）"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
        )
        self.model = DEEPSEEK_MODEL
        self.cache = {}

    # ── 语言检测 ──

    @staticmethod
    def detect_language(text: str) -> str:
        """检测文本语言：'zh' 或 'en'"""
        if not text:
            return "zh"
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if cn_chars > en_words else "en"

    @staticmethod
    def build_lang_hint(lang: str) -> str:
        """根据语言返回回答语言指令"""
        if lang == "en":
            return (
                "IMPORTANT: Answer in English. "
                "Respond to the user in the same language as their question.\n"
            )
        return "重要：请用中文回答。根据用户问题的语言使用对应的语言回复。\n"

    # ── 缓存 ──

    def _get_cache_key(self, query: str, context: Optional[str] = None) -> str:
        raw = f"{query}|{context or ''}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        if not CACHE_ENABLED:
            return None
        if cache_key in self.cache:
            data = self.cache[cache_key]
            if time.time() - data["timestamp"] < CACHE_EXPIRE_TIME:
                return data["response"]
            else:
                del self.cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, response: str):
        if CACHE_ENABLED:
            self.cache[cache_key] = {
                "response": response,
                "timestamp": time.time(),
            }

    # ── Prompt 构建 ──

    def _build_rag_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """构建 RAG 提示词（双语自适应）"""
        lang = self.detect_language(query)
        lang_hint = self.build_lang_hint(lang)

        # 拼接上下文
        context_lines = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk["content"].strip()
            context_lines.append(f"[{i}] {text}")
        context = "\n\n".join(context_lines)

        prompt = (
            f"{lang_hint}\n"
            "基于以下文档内容回答问题。\n\n"
            "【规则】\n"
            "1. 严格依据文档回答，不要编造信息\n"
            "2. 文档没有足够信息时，回答：根据文档内容，无法找到相关信息\n"
            "3. 回答简洁，引用对应文档段落编号\n\n"
            f"【参考文档】\n{context}\n\n"
            f"【问题】\n{query}\n\n"
            "【回答】\n"
        )
        return prompt

    def _build_system_prompt(self, history_text: str = "") -> str:
        """构建系统提示词"""
        if history_text:
            return f"""你是专业的知识问答助手。
严格基于提供的文档内容回答，不要编造信息。
根据用户问题的语言使用对应的语言回复。

{history_text}"""
        return "你是专业的知识问答助手。严格基于提供的文档内容回答，不要编造信息。根据用户问题的语言使用对应的语言回复。"

    # ── LLM 调用（流式）──

    def _call_llm_stream(self, messages: List[Dict], max_retries=2):
        """流式调用 LLM API，逐 token 产出（带重试机制）"""
        import time as _time
        start_time = _time.time()
        last_error = None
        
        for attempt in range(1 + max_retries):
            try:
                if attempt > 0:
                    wait = 2 ** attempt
                    print(f"  [LLM重试] 第{attempt}次重试，等待{wait}秒...")
                    _time.sleep(wait)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    stream=True,
                    timeout=60,
                )
                
                full_answer = ""
                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        token = delta.content
                        full_answer += token
                        yield token

                elapsed = _time.time() - start_time
                self._stream_result = {
                    "answer": full_answer.strip(),
                    "elapsed_time": round(elapsed, 4),
                    "source": "api",
                }
                return  # 成功则退出
                
            except Exception as e:
                last_error = e
                elapsed = _time.time() - start_time
                error_msg = f"API 调用失败: {str(e)}"
                print(f"  [LLM错误] 尝试{attempt+1}/{1+max_retries}: {error_msg}")
        
        # 所有重试都失败
        final_error = f"API 调用失败（重试{max_retries}次后）: {str(last_error)}"
        print(f"  [LLM错误] {final_error}")
        self._stream_result = {
            "answer": final_error,
            "elapsed_time": round(_time.time() - start_time, 4),
            "source": "error",
        }
        yield ""

    # ── 流式回答 ──

    def chat_answer_stream(self, query: str, context_chunks: List[Dict],
                           history: Optional[List[Dict]] = None):
        """
        流式对话式 RAG 回答（逐 token 产出）
        yield: 文本 token
        迭代结束后读取 self._stream_result 获取完整结果
        """
        self._stream_result = None

        # 构建 RAG 提示
        prompt = self._build_rag_prompt(query, context_chunks)

        # 缓存检查
        cache_key = self._get_cache_key(
            query, str([c["content"] for c in context_chunks])
        )
        cached = self._check_cache(cache_key)
        if cached:
            self._stream_result = {
                "answer": cached,
                "elapsed_time": 0,
                "source": "cache",
            }
            yield cached
            return

        # 构建 messages
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for h in history[-(LLM_CONTEXT_WINDOW * 2):]:
                messages.append(h)

        messages.append({"role": "user", "content": prompt})

        # 流式调用 LLM
        for token in self._call_llm_stream(messages):
            yield token

        # 缓存最终结果
        if self._stream_result and self._stream_result["source"] == "api":
            self._set_cache(cache_key, self._stream_result["answer"])

    # ── 流式回答 ──