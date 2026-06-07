# -*- coding: utf-8 -*-
"""
工单04 - 智能对话问答系统（高稳定性 · 结构化输出版）
新增特性：
  - 后台健康检查 + 系统状态面板
  - 结构化列表输出（替代段落）
  - 自动降级 + 错误优雅展示
  - 熔断器状态可视化
"""

import sys
# Clean PYTHONPATH to avoid numpy C ABI conflicts from Python312
for _p in list(sys.path):
    if 'Python312' in _p or 'python312' in _p.lower():
        sys.path.remove(_p)

import os
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_engine import QAEngine


# ── 页面配置 ──
st.set_page_config(
    page_title="工单04智能问答系统（高稳定性）",
    page_icon="🛡️",
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
        st.title("🛡️ 工单04 问答系统")
        st.caption("高稳定性 · 自动恢复 · 结构化输出")

        # ── 健康状态面板 ──
        st.subheader("💚 系统健康")
        health = engine.get_detailed_health()

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            db_status = "✅ 正常" if health.get("database") else "❌ 异常"
            st.metric("数据库", db_status)
        with col_h2:
            idx_status = "✅ 正常" if health.get("vector_index") else "❌ 异常"
            st.metric("向量索引", idx_status)

        # LLM 状态
        llm_status = health.get("llm_available", True)
        llm_label = "✅ 可用" if llm_status else "🔴 熔断"
        st.metric("LLM 服务", llm_label)

        # 降级状态
        downgraded = engine._downgraded
        overall_status = "⚠️ 降级中" if downgraded else "✅ 正常"
        st.metric("系统状态", overall_status)

        if health.get("errors"):
            with st.expander("⚠️ 异常记录"):
                for err in health["errors"][-5:]:
                    st.warning(err)

        # 熔断器状态
        cb_state = engine.llm_qa.get_circuit_breaker_state()
        cb_icon = {"CLOSED": "✅", "OPEN": "🔴", "HALF_OPEN": "⚠️"}
        st.caption(f"熔断器: {cb_icon.get(cb_state, '❓')} {cb_state}")

        st.divider()

        # ── 速度优化 ──
        st.subheader("🚀 性能状态")
        status = engine.get_system_status()
        warmed = status.get("warmed_up", False)
        cache_size = status.get("cache_size", 0)
        st.metric("连接池", "已预热 ✅" if warmed else "未预热 ⏳")
        st.metric("语义缓存", f"{cache_size} 条")
        st.metric("LLM Tokens", "128（快速模式）")

        st.divider()

        # ── 文档管理 ──
        st.subheader("📚 文档管理")

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
                st.error("❌ 添加失败，请检查文件格式")

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

        # ── 水印状态 ──
        st.subheader("💧 水印处理")
        wm_enabled = status.get("watermark_enabled", True)
        wm_removed = status.get("watermark_pages_removed", 0)
        st.metric("水印过滤", "启用 ✅" if wm_enabled else "关闭 ⛔")
        if wm_removed > 0:
            st.metric("已过滤水印页", wm_removed)

        st.divider()

        # ── 系统信息 ──
        st.subheader("⚙️ 系统信息")
        st.metric("文档数", status["total_documents"])
        st.metric("向量数", status["total_vectors"])
        st.caption(f"LLM: {status['llm_model']}")
        st.caption(f"嵌入: {status['embedding_model']}")

        # ── 操作按钮 ──
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_query = None
                st.session_state.last_response = None
                st.rerun()
        with col2:
            if st.button("🔄 刷新状态", use_container_width=True):
                st.rerun()

    # ── 主区域：对话 ──
    st.title("🔍 文档智能问答（结构化输出）")
    st.caption("高稳定性 · 自动恢复 · 结构化列表格式 · 异常容错")

    # 聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
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

            # 流式输出（结构化列表格式）
            stream = engine.chat_stream(prompt, history)
            answer_text = st.write_stream(stream)
            result = engine._stream_result or {}

            # 计时和信息
            elapsed = result.get("elapsed_total", 0)
            llm_source = result.get("llm_source", "api")
            downgraded = result.get("downgraded", False)

            source_icon = {
                "semantic_cache": "💡", "cache": "💾",
                "api": "🌐", "error": "❌"
            }
            icon = source_icon.get(llm_source, "🌐")
            color = "🟢" if elapsed < 3 else ("🟡" if elapsed < 6 else "🔴")

            tags = []
            if downgraded:
                tags.append("⚠️ 降级模式")
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
