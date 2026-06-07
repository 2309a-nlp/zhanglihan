# -*- coding: utf-8 -*-
"""
工单03 - 智能对话问答系统（Streamlit 界面 | 速度优化 + 水印处理版）
全链路优化目标：首次问答 < 3s
关键优化：
  - HTTP 连接池复用（省 0.5-1.5s TCP 握手）
  - LLM_MAX_TOKENS = 64（省 30-50% 生成时间）
  - 语义缓存（相似问题秒回）
  - PDF水印自动检测与清洗
"""

import os
import sys
import time

# 清理 PYTHONPATH 中不兼容的 Python312 site-packages
_old_paths = [p for p in sys.path if "Python312" in p or "python3.12" in p]
for _p in _old_paths:
    sys.path.remove(_p)

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_engine import QAEngine


# ── 页面配置 ──
st.set_page_config(
    page_title="工单03智能问答系统",
    page_icon="⚡",
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
        st.title("⚡ 工单03 问答系统")
        st.caption("全链路优化 · < 3s · 高并发 · 水印清洗 · bge-small-zh · DeepSeek")

        # 速度优化状态
        st.subheader("🚀 速度优化")
        status = engine.get_system_status()
        warmed = status.get("warmed_up", False)
        cache_size = status.get("cache_size", 0)
        st.metric("连接池", "已预热 ✅" if warmed else "未预热 ⏳")
        st.metric("语义缓存", f"{cache_size} 条")
        st.metric("LLM Tokens", "64（快速模式）")
        st.metric("最大并发", "20 连接")

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
        st.subheader("💧 水印处理")
        wm_enabled = status.get("watermark_enabled", True)
        wm_removed = status.get("watermark_pages_removed", 0)
        st.metric("水印过滤", "启用 ✅" if wm_enabled else "关闭 ⛔")
        if wm_removed > 0:
            st.metric("已过滤水印页", wm_removed)

        st.divider()

        # 系统状态
        st.subheader("⚙️ 系统状态")
        st.metric("文档数", status["total_documents"])
        st.metric("向量数", status["total_vectors"])
        st.caption(f"LLM: {status['llm_model']}")
        st.caption(f"嵌入: {status['embedding_model']}")

        # 清空对话
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_query = None
                st.session_state.last_response = None
                st.rerun()
        with col2:
            if st.button("🔥 预热", use_container_width=True,
                         disabled=warmed):
                engine._warmup()
                st.rerun()

    # ── 主区域：对话 ──
    st.title("🔍 文档智能问答")
    st.caption("目标：首次问答 < 3s | 重复问题 < 0.5s（语义缓存命中）")

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
            result = {}
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-8:]
            ]

            # 流式输出：连接池复用 + 64 tokens → 首字 < 0.3s，总耗时 < 3s
            stream = engine.chat_stream(prompt, history)
            answer_text = st.write_stream(stream)
            result = engine._stream_result or {}

            # 显示计时和来源信息
            elapsed = result.get("elapsed_total", 0)
            llm_source = result.get("llm_source", "api")
            source_icon = {"semantic_cache": "💡", "cache": "💾", "api": "🌐", "error": "❌"}
            icon = source_icon.get(llm_source, "🌐")
            color = "🟢" if elapsed < 3 else ("🟡" if elapsed < 6 else "🔴")
            tags = []
            if llm_source == "semantic_cache":
                tags.append("💡语义缓存")
            elif warmed:
                tags.append("✅连接池复用")
            else:
                tags.append("⚠️首次连接")
            tag_str = " | ".join(tags)
            st.caption(f"{color} ⏱ {elapsed:.2f}s {icon} {tag_str}")

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
