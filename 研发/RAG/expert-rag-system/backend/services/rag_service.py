# -*- coding: utf-8 -*-
import os
import sys
import logging
import jieba

# 设置 HuggingFace 离线模式（避免网络超时），必须在导入 transformers 之前
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 导入 vector_db 模块（动态加载角色索引）
from vector_db import (
    similarity_search_with_scores as faiss_search,
    load_role_index,
    get_current_role,
    bm25_model as vector_bm25,
    text_chunks,
)

# 导入长期记忆模块
from services.long_term_memory import (
    build_memory_context,
    update_memory_from_dialog,
)

# 导入重排序模块
from services.reranker import rerank_documents

from config import settings

# 角色中文名称映射
ROLE_NAMES_CN = {
    "Medical": "医学",
    "Finance": "金融",
    "Law": "法律",
    "Education": "教育",
    "Psychology": "心理",
}

# 角色系统提示词前缀
ROLE_SYSTEM_PROMPTS = {
    "Medical": "你是一位专业的医学顾问，擅长回答疾病诊断、治疗方案、用药建议等医学相关问题。",
    "Finance": "你是一位专业的金融分析师，擅长回答投资理财、股票基金、市场分析等金融相关问题。",
    "Law": "你是一位专业的法律顾问，擅长回答法律条文、案例分析、法律程序等法律相关问题。",
    "Education": "你是一位专业的教育专家，擅长回答教学方法、教育政策、学生发展等教育相关问题。",
    "Psychology": "你是一位专业的心理咨询师，擅长回答心理健康、情绪管理、人际关系等心理相关问题。",
}

# ==================== LLM 后端选择函数 ====================
def _create_llm():
    """根据 LLM_BACKEND 配置创建对应的 LLM 实例"""
    backend = settings.LLM_BACKEND

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        logger.info(f"使用 Ollama 后端: {settings.OLLAMA_MODEL_NAME} ({settings.OLLAMA_BASE_URL})")
        return ChatOllama(
            model=settings.OLLAMA_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.5,
        )

    elif backend == "vllm":
        logger.info(f"使用 vLLM 后端: {settings.VLLM_BASE_URL}")
        return ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"{settings.VLLM_BASE_URL}/v1",
            model_name=settings.VLLM_MODEL_NAME,
            temperature=0.5,
        )

    elif backend == "sglang":
        logger.info(f"使用 SGLang 后端: {settings.SGLANG_BASE_URL}")
        return ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"{settings.SGLANG_BASE_URL}/v1",
            model_name=settings.SGLANG_MODEL_NAME,
            temperature=0.5,
        )

    else:
        logger.info("使用 DeepSeek API 后端")
        return ChatOpenAI(
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
            temperature=0.5,
        )
llm = _create_llm()


def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def l2_distance_to_similarity(l2: float) -> float:
    return 1.0 / (1.0 + float(l2))


def hybrid_search(question: str, k=4, alpha=0.5):
    """混合检索：结合当前角色的向量检索(FAISS)和关键词检索(BM25)"""
    if vector_bm25 is None or not text_chunks:
        # 纯向量检索
        vr = faiss_search(question, k=k)
        results = []
        for doc, l2 in vr:
            sim = l2_distance_to_similarity(l2)
            results.append((doc, sim))
        return results

    vector_results = faiss_search(question, k=k)
    if not vector_results:
        return []

    docs_list = []
    vector_sim_list = []
    for doc, l2_score in vector_results:
        sim = l2_distance_to_similarity(l2_score)
        docs_list.append(doc)
        vector_sim_list.append(sim)

    tokenized_q = list(jieba.cut(question))
    bm25_scores = vector_bm25.get_scores(tokenized_q)

    bm25_scores_for_docs = []
    for doc in docs_list:
        content = doc.page_content
        best_score = 0.0
        for i, chunk in enumerate(text_chunks):
            chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            if chunk_text == content:
                best_score = bm25_scores[i]
                break
        bm25_scores_for_docs.append(best_score)

    if not docs_list:
        return []

    max_vector = max(vector_sim_list) if vector_sim_list else 1
    max_bm25 = max(bm25_scores_for_docs) if bm25_scores_for_docs else 1

    combined = []
    for i, doc in enumerate(docs_list):
        norm_vector = vector_sim_list[i] / max_vector if max_vector > 0 else 0
        norm_bm25 = bm25_scores_for_docs[i] / max_bm25 if max_bm25 > 0 else 0
        combined_score = (alpha * norm_vector) + ((1 - alpha) * norm_bm25)
        combined.append((doc, combined_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]


def call_rag_llm(question: str, chat_history=None, role: str = "Medical", user_id: str = None):
    """
    RAG 核心入口（支持多角色 + 长期记忆）
    根据 role 加载对应的知识库索引进行检索
    根据 user_id 注入长期记忆并在对话后更新
    """
    if chat_history is None:
        chat_history = []

    SEMANTIC_THRESHOLD = 0.55

    meta = {
        "kb_available": False,
        "kb_hit": False,
        "answer_mode": "llm_only_no_index",
        "source_label": "知识库未初始化...",
        "citations": [],
        "best_similarity": None,
        "similarity_threshold": SEMANTIC_THRESHOLD,
        "current_role": role,
    }

    try:
        # 1. 注入长期记忆（用户个人信息）
        memory_context = ""
        if user_id:
            memory_context = build_memory_context(user_id)
            if memory_context:
                logger.info(f"注入长期记忆: [{memory_context}]")

        # 2. 根据角色加载对应的知识库索引
        index_loaded = load_role_index(role) if role else False
        kb_available = index_loaded

        docs_list = []
        raw_similarity = None

        if kb_available:
            meta["kb_available"] = True
            meta["source_label"] = f"已加载【{ROLE_NAMES_CN.get(role, role)}】知识库，检索中..."

            vector_results = faiss_search(question, k=settings.RAG_RETRIEVAL_K)

            if vector_results:
                top_doc, top_l2 = vector_results[0]
                raw_similarity = l2_distance_to_similarity(top_l2)
                meta["best_similarity"] = raw_similarity

                if raw_similarity >= SEMANTIC_THRESHOLD:
                    hybrid_results = hybrid_search(
                        question, k=settings.RAG_RETRIEVAL_K, alpha=0.6
                    )
                    if hybrid_results:
                        docs_list = [d for d, _ in hybrid_results]

            if docs_list:
                # === 重排序：对检索结果精排 ===
                if settings.USE_RERANKER and len(docs_list) > 1:
                    try:
                        docs_list = rerank_documents(question, docs_list, top_k=settings.RERANK_TOP_K)
                        logger.info(f"[{role}] 重排序完成，保留 {len(docs_list)} 条")
                    except Exception as re_e:
                        logger.warning(f"重排序失败，使用原始排序: {re_e}")

                logger.info(f"[{role}] 知识库命中，语义相似度 {raw_similarity:.2f}")
                meta["kb_hit"] = True
                meta["answer_mode"] = "rag_with_kb"
                meta["source_label"] = (
                    f"【{ROLE_NAMES_CN.get(role, role)}】知识库检索："
                    f"Top1 语义相关度 {raw_similarity:.2f}，基于检索片段作答"
                    + ("（已重排序）" if settings.USE_RERANKER else "")
                )

                role_prefix = ROLE_SYSTEM_PROMPTS.get(role, "")
                # 长期记忆放在最前面，用明显标记强调
                memory_line = ""
                if memory_context:
                    memory_line = f"\n【重要：请记住以下关于用户的信息，并在回答中恰当使用（如称呼用户的名字）】\n{memory_context}\n"
                system_rules = (
                    f"{role_prefix}{memory_line}"
                    "你是严谨的文档问答助手。你的回答必须严格基于下方参考资料中的内容。"
                    "如果参考资料中有相关信息，请详细引用并组织回答；"
                    "如果参考资料不足以完整回答问题，请诚实说明哪些信息来自资料，"
                    "哪些是你补充的常识。不要编造参考资料中不存在的信息。"
                )
                human_template = "参考资料：\n{context}\n\n当前问题：{question}"
                invoke_kw = {
                    "chat_history": chat_history,
                    "question": question,
                    "context": "\n---\n".join([doc.page_content for doc in docs_list]),
                }
            else:
                reason = ""
                if raw_similarity is not None:
                    reason = f"语义相似度 {raw_similarity:.2f} < 阈值 {SEMANTIC_THRESHOLD}"
                    logger.info(f"[{role}] 知识库无相关信息（{reason}）")
                else:
                    reason = "FAISS 检索无结果"
                    logger.info(f"[{role}] 知识库检索无结果")

                meta["kb_hit"] = False
                meta["answer_mode"] = "llm_only"
                meta["source_label"] = (
                    f"【{ROLE_NAMES_CN.get(role, role)}】知识库中未检索到相关信息（{reason}），"
                    "回答由大模型生成"
                )

                role_prefix = ROLE_SYSTEM_PROMPTS.get(role, "")
                memory_line = ""
                if memory_context:
                    memory_line = f"\n【重要：请记住以下关于用户的信息，并在回答中恰当使用（如称呼用户的名字）】\n{memory_context}\n"
                system_rules = f"{role_prefix}{memory_line}\n你是通用问答助手，请根据你的知识直接回答。"
                human_template = "当前问题：{question}"
                invoke_kw = {"chat_history": chat_history, "question": question}
        else:
            logger.warning(f"[{role}] 知识库索引未加载，使用大模型回答")
            meta["kb_hit"] = False
            meta["answer_mode"] = "llm_only"
            meta["source_label"] = (
                f"【{ROLE_NAMES_CN.get(role, role)}】知识库索引未加载，回答由大模型生成"
            )

            role_prefix = ROLE_SYSTEM_PROMPTS.get(role, "")
            memory_line = ""
            if memory_context:
                memory_line = f"\n【重要：请记住以下关于用户的信息，并在回答中恰当使用（如称呼用户的名字）】\n{memory_context}\n"
            system_rules = f"{role_prefix}{memory_line}\n你是通用问答助手，请根据你的知识直接回答。"
            human_template = "当前问题：{question}"
            invoke_kw = {"chat_history": chat_history, "question": question}

        # --- 统一调用大模型 ---
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_rules),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", human_template),
        ])

        chain = prompt_template | llm
        response = chain.invoke(invoke_kw)
        final_answer = content_to_text(response.content)
        contexts_for_eval = [doc.page_content for doc in docs_list]

        # === 3. 对话后更新长期记忆（异步提取关键信息）===
        if user_id:
            try:
                update_memory_from_dialog(user_id, question, final_answer, llm)
            except Exception as mem_e:
                logger.warning(f"更新长期记忆失败（不影响回答）: {mem_e}")

        return final_answer, contexts_for_eval, meta

    except Exception as e:
        logger.error(f"回答生成失败: {e}")
        meta["source_label"] = f"处理出错：{e}"
        return f"回答生成失败: {str(e)}", [], meta
