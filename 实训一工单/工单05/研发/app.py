"""
RAG 智能问答系统 — Streamlit 界面
武汉兴图新科电子股份有限公司 招股说明书分析
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path

st.set_page_config(
    page_title="RAG 智能问答系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))
from rag_engine import (
    init_rag, query as rag_query, reset_conversation, get_db_stats,
)

# Style
st.markdown("""
<style>
    .stApp { background: #0f1117; }
    .main .block-container { padding-top: 1.5rem; }
    .msg-user {
        background: #2d3748; color: #e2e8f0; border-radius: 18px 18px 4px 18px;
        padding: 12px 18px; margin: 8px 0 8px 40px; max-width: 85%;
        line-height: 1.5; border: 1px solid #4a5568; font-size: 0.95rem;
    }
    .msg-user-label { text-align: right; color: #63b3ed; font-size: 0.8rem; margin-bottom: 2px; }
    .msg-assistant {
        background: #1a2332; color: #e2e8f0; border-radius: 18px 18px 18px 4px;
        padding: 12px 18px; margin: 8px 40px 8px 0; max-width: 85%;
        line-height: 1.5; border: 1px solid #2d3748; font-size: 0.95rem;
    }
    .msg-assistant-label { color: #68d391; font-size: 0.8rem; margin-bottom: 2px; }
    .msg-time { color: #718096; font-size: 0.7rem; margin-top: 4px; text-align: right; }
    .data-table { width: 100%; border-collapse: collapse; margin: 8px 0; }
    .data-table th {
        background: #1e3a5f; color: #90cdf4; padding: 8px 12px;
        text-align: right; border: 1px solid #2a4365; font-weight: 600;
    }
    .data-table th:first-child { text-align: left; }
    .data-table td {
        padding: 6px 12px; text-align: right; border: 1px solid #2a4365; color: #e2e8f0;
    }
    .data-table td:first-child { text-align: left; color: #a0aec0; }
    .data-table tr:nth-child(even) td { background: #1a2332; }
    .data-table tr:nth-child(odd) td { background: #1e293b; }
    .sidebar-stat {
        color: #a0aec0; font-size: 0.85rem; padding: 6px 0;
        border-bottom: 1px solid #1a202c;
    }
    .sidebar-stat span { color: #68d391; font-weight: 600; }
    .stTextInput input {
        background: #1a2332 !important; color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important; border-radius: 12px !important;
    }
    .stButton button { background: #2b6cb0; color: white; border-radius: 10px; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-green { background: #22543d; color: #68d391; }
    .badge-blue { background: #1a365d; color: #90cdf4; }
</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "stats" not in st.session_state:
    st.session_state.stats = {}

def _(zh, en):
    return zh if st.session_state.lang == "zh" else en

def initialize_rag():
    with st.spinner(_("正在加载知识库索引...", "Loading knowledge base...")):
        ok = init_rag(force_rebuild=False)
        if ok:
            st.session_state.stats = get_db_stats()
            st.session_state.initialized = True
            return True
    st.error(_("❌ 索引加载失败", "❌ Index load failed"))
    return False

def render_table(text):
    if "|" not in text:
        return text
    lines = text.split("\n")
    html = []
    tbl = []
    in_tbl = False
    for line in lines:
        s = line.strip()
        if s.startswith("**") and s.endswith("**"):
            if in_tbl and tbl:
                html.append(_build_table(tbl))
                tbl = []
                in_tbl = False
            html.append(f"<div style='font-size:1.05rem;font-weight:600;color:#e2e8f0;margin:12px 0 4px 0;'>{s.strip('*')}</div>")
        elif s.startswith("|"):
            in_tbl = True
            tbl.append(line)
        else:
            if in_tbl and tbl:
                html.append(_build_table(tbl))
                tbl = []
                in_tbl = False
            if s:
                html.append(f"<div style='color:#cbd5e1;margin:6px 0;'>{s}</div>")
    if in_tbl and tbl:
        html.append(_build_table(tbl))
    return "".join(html)

def _build_table(lines):
    rows = []
    is_header = True
    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and cells[0].startswith("---"):
            is_header = False
            continue
        tag = "th" if is_header else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        is_header = False
    return "<table class='data-table'>" + "".join(rows) + "</table>"

def display_msg(msg):
    role = msg["role"]
    content = msg["content"]
    elapsed = msg.get("elapsed", 0)
    if role == "user":
        st.markdown(f'<div class="msg-user-label">🧑 💬 {_("你", "You")}</div><div class="msg-user">{content}</div>', unsafe_allow_html=True)
    else:
        rendered = render_table(content)
        time_html = f'<div class="msg-time">⚡ {elapsed:.2f}s</div>' if elapsed > 0 else ""
        st.markdown(f'<div class="msg-assistant-label">🤖 {_("助手", "Assistant")} <span class="badge badge-blue">DeepSeek</span> <span class="badge" style="background:#2d3748;color:#a0aec0;">TF-IDF</span></div><div class="msg-assistant">{rendered}{time_html}</div>', unsafe_allow_html=True)

def send_message(question):
    if not question.strip():
        return
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        t0 = time.time()
        result = rag_query(question)
        elapsed = time.time() - t0
        answer = result.get("answer", "")
        elapsed = result.get("elapsed", elapsed)
    except Exception as e:
        answer = f"❌ {_('查询出错', 'Query error')}: {e}"
        elapsed = 0
    if not answer or answer.startswith("未找到"):
        answer = _("未能找到相关答案，请换一种方式提问。", "No relevant answer found.")
    st.session_state.messages.append({"role": "assistant", "content": answer, "elapsed": elapsed})
    st.session_state.stats["last_query_time"] = elapsed

def handle_upload(uploaded_files):
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    saved = []
    for f in uploaded_files:
        path = data_dir / f.name
        with open(str(path), "wb") as fh:
            fh.write(f.getbuffer())
        saved.append(f.name)
    if saved:
        st.success(_(f"已保存 {len(saved)} 个文件，正在重新构建索引...", f"Saved {len(saved)} files, rebuilding..."))
        ok = init_rag(force_rebuild=True)
        st.session_state.initialized = ok
        st.session_state.stats = get_db_stats() if ok else {}
        if ok:
            st.success(_("✅ 索引构建成功！", "✅ Index rebuilt!"))
        else:
            st.error(_("❌ 索引构建失败", "❌ Index rebuild failed"))
        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;padding:8px 0 16px 0;">📄 RAG 智能问答</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🇨🇳 中文", use_container_width=True, disabled=(st.session_state.lang == "zh")):
            st.session_state.lang = "zh"
            st.rerun()
    with col_b:
        if st.button("🇬🇧 English", use_container_width=True, disabled=(st.session_state.lang == "en")):
            st.session_state.lang = "en"
            st.rerun()
    st.markdown("---")
    st.markdown(f'<div style="color:#a0aec0;font-size:0.9rem;font-weight:500;margin-bottom:4px;">📤 {_("上传数据集", "Upload Data")}</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(_("选择 PDF 文件", "Choose PDF files"), type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        handle_upload(uploaded)
    st.markdown("---")
    st.markdown(f'<div style="color:#a0aec0;font-size:0.9rem;font-weight:500;margin-bottom:8px;">⚙️ {_("系统状态", "System Status")}</div>', unsafe_allow_html=True)
    stats = st.session_state.stats
    if stats:
        st.markdown(f'<div class="sidebar-stat">📚 {_("知识库", "KB")}: <span>{stats.get("chunks", 0)}</span> {_("片段", "chunks")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat">🏷️ {_("词汇量", "Vocab")}: <span>{stats.get("terms", 0):,}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat">💬 {_("对话轮次", "Turns")}: <span>{stats.get("history_turns", 0)}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat">🤖 {_("模型", "Model")}: <span>{stats.get("llm_model", "deepseek-chat")}</span></div>', unsafe_allow_html=True)
        if "last_query_time" in stats:
            cls = "badge-green" if stats["last_query_time"] < 3 else ""
            st.markdown(f'<div class="sidebar-stat">⚡ {_("上次响应", "Last Resp")}: <span class="badge {cls}">{stats["last_query_time"]:.2f}s</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button(_("🔄 重置对话", "🔄 Reset Chat"), use_container_width=True):
        reset_conversation()
        st.session_state.messages = []
        st.session_state.stats = get_db_stats()
        st.rerun()
    if st.button(_("🔁 重新构建索引", "🔁 Rebuild Index"), use_container_width=True):
        with st.spinner(_("正在重新构建索引...", "Rebuilding...")):
            ok = init_rag(force_rebuild=True)
            st.session_state.initialized = ok
            st.session_state.stats = get_db_stats() if ok else {}
        if ok:
            st.success(_("✅ 索引重建完成", "✅ Index rebuilt"))
        st.rerun()
    st.markdown("---")
    st.markdown(f'<div style="color:#4a5568;font-size:0.75rem;text-align:center;">DeepSeek + TF-IDF<br>{_("响应时间 < 3s", "Response < 3s")}</div>', unsafe_allow_html=True)

# Main area
st.markdown(f'<div style="font-size:1.5rem;font-weight:700;color:#e2e8f0;margin-bottom:4px;">📄 RAG {_("智能问答系统", "Q&A System")}</div><div style="color:#718096;font-size:0.85rem;margin-bottom:20px;">{_("TF-IDF + DeepSeek", "TF-IDF + DeepSeek")} | {_("支持多轮对话中英双语", "Multi-turn, CN/EN")}</div>', unsafe_allow_html=True)

if not st.session_state.initialized:
    initialize_rag()

for msg in st.session_state.messages:
    display_msg(msg)

# Suggestions
if not st.session_state.messages:
    suggestions = [
        _("❓ 军用领域收入有多少", "❓ Military revenue"),
        _("❓ 直接军方收入是多少", "❓ Direct military"),
        _("❓ 间接军方收入是多少", "❓ Indirect military"),
        _("❓ 民用领域收入", "❓ Civilian revenue"),
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, use_container_width=True, type="secondary"):
                send_message(s.replace("❓ ", ""))
                st.rerun()

# Input
placeholder = _("输入你的问题...（如：军用领域收入有多少）", "Type your question... (e.g., Military revenue)")
col1, col2 = st.columns([6, 1])
with col1:
    question = st.text_input("q", placeholder=placeholder, label_visibility="collapsed", key="q_input")
with col2:
    send_btn = st.button(_("发送 ➤", "Send ➤"), use_container_width=True, type="primary")
if send_btn and question.strip():
    send_message(question.strip())
    st.rerun()
