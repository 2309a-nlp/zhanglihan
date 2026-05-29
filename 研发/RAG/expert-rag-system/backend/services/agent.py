# backend/services/agent.py
from services.rag_service import call_rag_llm
from services.memory import get_history, add_to_history



def run_chat(question: str, session_id: str = "default_session", role: str = "Medical", user_id: str = None):
    """
    智能体主入口：
    1. 获取历史记忆（短期对话记忆）
    2. 根据角色加载对应的知识库索引，调用 RAG 生成回答（含长期记忆注入）
    3. 更新短期历史记忆 + 长期记忆
    4. 返回回答和上下文
    """
    # 1. session_id 本身就是用户名，直接用作 user_id
    if user_id is None:
        user_id = session_id
    # 2. 获取该会话的短期对话历史记录
    history = get_history(session_id)

    # 3. 调用 RAG：传入角色 + user_id（用于长期记忆）
    answer, contexts, meta = call_rag_llm(
        question=question,
        chat_history=history,
        role=role,
        user_id=user_id
    )

    # 4. 将本轮对话存入短期记忆库
    add_to_history(session_id, question, answer)

    # 5. 返回回答、上下文、meta
    return answer, contexts, meta