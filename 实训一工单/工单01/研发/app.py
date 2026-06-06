# -*- coding: utf-8 -*-
"""
工单01 - 智能对话问答系统（Streamlit 界面）
基于 RAG 技术的 PDF 文档对话机器人
支持中英文双语、对话历史、文档管理
"""

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_engine import QAEngine


# ── 页面配置 ──
st.set_page_config(
    page_title="工单智能问答系统",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource
def get_engine():
    """全局单例引擎"""
    eng = QAEngine()
    eng.initialize()
    return eng


def init_session():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    if "last_response" not in st.session_state:
        st.session_state.last_response = None


def main():
    init_session()
    engine = get_engine()

    # ── 侧边栏：系统控制 ──
    with st.sidebar:
        st.title("💬 文档问答系统")
        st.caption("基于 RAG · bge-small-zh-v1.5 · DeepSeek")

        st.divider()
        st.subheader("📚 文档管理")

        # 上传
        uploaded = st.file_uploader("上传 PDF", type="pdf", label_visibility="collapsed")
        if uploaded:
            save_dir = "uploads"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("解析和索引中..."):
                ok = engine.add_pdf(save_path)
            if ok:
                st.success(f"✅ {uploaded.name}")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ 添加失败")

        # 文档列表
        docs = engine.list_docs()
        if not docs:
            st.info("暂无文档，请上传 PDF")
        else:
            for d in docs:
                cols = st.columns([3, 1])
                cols[0].write(f"📄 {d['filename'][:20]}")
                cols[1].write(f"{d['total_chunks']}块")
                if st.button("🗑", key=f"del_{d['id']}", help="删除"):
                    engine.remove_doc(d["id"])
                    st.rerun()

        st.divider()

        # 系统状态
        st.subheader("⚙️ 系统状态")
        status = engine.get_system_status()
        st.metric("文档数", status["total_documents"])
        st.metric("向量数", status["total_vectors"])
        st.caption(f"LLM: {status['llm_model']}")
        st.caption(f"嵌入: {status['embedding_model']}")

        # 清空对话
        st.divider()
        if st.button("🗑 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_query = None
            st.session_state.last_response = None
            st.rerun()

    # ── 主区域：对话 ──
    st.title("🔍 文档智能问答")

    # 聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 如果是助手回复且有来源，显示参考片段
            if msg["role"] == "assistant" and "source_chunks" in msg:
                with st.expander("📎 参考文档片段"):
                    for i, src in enumerate(msg["source_chunks"], 1):
                        score = src.get("score", 0)
                        st.markdown(f"**片段 {i}** (相似度: {score:.4f})")
                        st.text(src["content"][:200])
                        if i < len(msg["source_chunks"]):
                            st.divider()

    # 输入框
    if prompt := st.chat_input("输入您的问题（支持中英文）...", key="chat_input"):
        # 显示用户消息
        st.chat_message("user").markdown(prompt)

        # 获取回答
        with st.chat_message("assistant"):
            result = {}  # 防止后续引用时未定义
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-8:]  # 最近4轮
            ]

            # 流式输出：用户看到第一个字 <= 0.5s，整体无等待感
            stream = engine.chat_stream(prompt, history)
            answer_text = st.write_stream(stream)
            result = engine._stream_result or {}

            if "elapsed_total" in result:
                st.caption(f"⏱ {result['elapsed_total']:.2f}s")

            # 来源
            if result.get("source_chunks"):
                with st.expander("📎 参考文档片段"):
                    for i, src in enumerate(result["source_chunks"], 1):
                        st.markdown(f"**片段 {i}** (相似度: {src['score']:.4f})")
                        st.text(src["content"][:200])
                        if i < len(result["source_chunks"]):
                            st.divider()

        # 保存到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "source_chunks": result.get("source_chunks", []),
        })


if __name__ == "__main__":
    main()
