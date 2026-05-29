# -*- coding: utf-8 -*-
# backend/services/memory.py
from langchain_core.messages import HumanMessage, AIMessage

# 全局会话记忆库：{ "session_id": [HumanMessage, AIMessage, ...] }
# 全局字典  Key 是 session_id（用户名）
conversation_memory = {}

def get_history(session_id: str):
    """获取指定会话的历史记录"""
    return conversation_memory.get(session_id, [])

def add_to_history(session_id: str, question: str, answer: str):
    """将新的一轮对话存入记忆库"""
    # 如果这个用户第一次说话，先给他建个空列表
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    # 追加“人话”
    conversation_memory[session_id].append(HumanMessage(content=question))
    # 追加“AI回复”
    conversation_memory[session_id].append(AIMessage(content=answer))