"""
RAG 多轮对话问答系统 — Streamlit UI
支持：向量/全文/混合检索、3种重排算法、多嵌入模型、中英文
"""

import streamlit as st
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from main import init_rag, query as rag_query, reset_conversation, get_db_stats
from config import EMBEDDING_MODELS
from rerankers import RERANKER_REGISTRY

st.set_page_config(
    page_title="RAG 智能问答系统 v2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 样式 ====================
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

    .sidebar-stat {
        color: #a0aec0; font-size: 0.85rem; padding: 6px 0;
        border-bottom: 1px solid #1a202c;
    }
    .sidebar-stat span { color: #68d391; font-weight: 600; }

    .stTextInput input {
        background: #1a2332 !important; color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important; border-radius: 12px !important;
    }
    .stSelectbox > div > div {
        background: #1a2332 !important; color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important;
    }
    .stButton button {
        background: #2b6cb0; color: white; border-radius: 10px;
        border: none; font-weight: 600;
    }
    .stButton button:hover { background: #3182ce; }
    .stSpinner > div { border-color: #63b3ed !important; }

    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-green { background: #22543d; color: #68d391; }
    .badge-blue { background: #1a365d; color: #90cdf4; }
    .badge-purple { background: #2a1a5e; color: #b794f4; }
    .badge-orange { background: #5a3a1a; color: #f6ad55; }
    .badge-red { background: #5a1a1a; color: #fc8181; }

    .info-card {
        background: #1a2332; border: 1px solid #2d3748; border-radius: 10px;
        padding: 12px; margin: 8px 0;
    }
    .info-card h4 { color: #90cdf4; margin: 0 0 8px 0; }
    .info-card p { color: #a0aec0; margin: 2px 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ==================== Session State ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "search_type" not in st.session_state:
    st.session_state.search_type = "hybrid"
if "hybrid_strategy" not in st.session_state:
    st.session_state.hybrid_strategy = "weighted"
if "ft_mode" not in st.session_state:
    st.session_state.ft_mode = "boolean"
if "reranker" not in st.session_state:
    st.session_state.reranker = "tfidf"
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = "bge-small-zh-v1.5"
if "top_k" not in st.session_state:
    st.session_state.top_k = 3


def _(zh, en):
    """中英文切换"""
    return zh if st.session_state.lang == "zh" else en


def initialize_rag():
    """初始化 RAG 系统"""
    with st.spinner(_("正在加载知识库索引...", "Loading knowledge base index...")):
        ok = init_rag(
            force_rebuild=False,
            embedding_model=st.session_state.embedding_model,
        )
        if ok:
            st.session_state.stats = get_db_stats()
            st.session_state.initialized = True
            # 预热嵌入模型，避免首次查询慢
            try:
                from embeddings import embed_query
                embed_query("warmup", model_name=st.session_state.embedding_model)
            except Exception:
                pass
            return True
    st.error(_("❌ 索引加载失败", "❌ Index load failed"))
    return False


def handle_query():
    """处理用户查询"""
    if not st.session_state.initialized:
        st.warning(_("请先初始化系统", "Please initialize the system first"))
        return

    question = st.session_state.query_input.strip()
    if not question:
        return

    # 添加到消息列表
    st.session_state.messages.append({"role": "user", "content": question})

    # 执行检索
    with st.spinner(_("🔍 正在检索...", "🔍 Searching...")):
        result = rag_query(
            question=question,
            search_type=st.session_state.search_type,
            hybrid_strategy=st.session_state.hybrid_strategy,
            ft_mode=st.session_state.ft_mode,
            reranker_name=st.session_state.reranker,
            top_k=st.session_state.top_k,
        )

    # 添加到消息列表
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "elapsed": result.get("elapsed", 0),
        "retrieval_time": result.get("retrieval_time", 0),
        "total_results": result.get("total_results", 0),
    })

    # 清空输入
    st.session_state.query_input = ""


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown(f"## 🤖 {_('RAG 智能问答系统', 'RAG QA System')}")

    # 语言切换
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button("🇨🇳 中文", use_container_width=True,
                     type="primary" if st.session_state.lang == "zh" else "secondary"):
            st.session_state.lang = "zh"
            st.rerun()
    with lang_col2:
        if st.button("🇺🇸 English", use_container_width=True,
                     type="primary" if st.session_state.lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()

    st.divider()

    # 初始化按钮
    if not st.session_state.initialized:
        if st.button(_("🚀 初始化系统", "🚀 Initialize System"), use_container_width=True):
            initialize_rag()
            st.rerun()
    else:
        st.markdown(f"<div class='badge badge-green'>✅ {_('已初始化', 'Ready')}</div>",
                    unsafe_allow_html=True)

    st.divider()

    # 检索配置
    st.markdown(f"### ⚙️ {_('检索配置', 'Search Config')}")

    st.session_state.search_type = st.selectbox(
        _("检索方式", "Search Type"),
        options=["hybrid", "vector", "fulltext"],
        index=0,
        format_func=lambda x: {
            "hybrid": _("🔀 混合检索", "🔀 Hybrid"),
            "vector": _("📊 向量检索", "📊 Vector"),
            "fulltext": _("📝 全文检索", "📝 Full-text"),
        }.get(x, x),
        help=_("选择检索方式：混合检索结合向量语义和全文关键词", "Choose search method"),
    )

    # 混合检索策略
    if st.session_state.search_type == "hybrid":
        st.session_state.hybrid_strategy = st.selectbox(
            _("融合策略", "Fusion Strategy"),
            options=["weighted", "rrf", "cascade"],
            format_func=lambda x: {
                "weighted": _("加权融合", "Weighted"),
                "rrf": _("RRF 融合", "RRF"),
                "cascade": _("级联融合", "Cascade"),
            }.get(x, x),
        )

    # 全文检索模式
    if st.session_state.search_type in ["fulltext", "hybrid"]:
        st.session_state.ft_mode = st.selectbox(
            _("全文模式", "Full-text Mode"),
            options=["boolean", "phrase", "fuzzy"],
            format_func=lambda x: {
                "boolean": _("布尔查询", "Boolean"),
                "phrase": _("短语匹配", "Phrase"),
                "fuzzy": _("模糊匹配", "Fuzzy"),
            }.get(x, x),
            help=_("布尔: AND/OR/NOT | 短语: 精确连续短语 | 模糊: 编辑距离匹配", "Search mode"),
        )

    # 重排算法
    st.session_state.reranker = st.selectbox(
        _("重排算法", "Reranker"),
        options=["llm", "tfidf", "adaptive", "none"],
        format_func=lambda x: {
            "llm": _("🤖 LLM 重排器", "🤖 LLM Reranker"),
            "tfidf": _("📊 TF-IDF 重排器", "📊 TF-IDF"),
            "adaptive": _("🎯 自适应重排器", "🎯 Adaptive"),
            "none": _("⏭ 不重排", "⏭ None"),
        }.get(x, x),
        help=_("选择重排算法优化检索结果排序", "Choose reranking algorithm"),
    )

    st.divider()

    # 高级配置
    with st.expander(_("🔧 高级配置", "🔧 Advanced"), expanded=False):
        st.session_state.embedding_model = st.selectbox(
            _("嵌入模型", "Embedding Model"),
            options=list(EMBEDDING_MODELS.keys()),
            index=0,
            format_func=lambda x: f"{x} ({EMBEDDING_MODELS[x]['dim']}d)",
        )

        st.session_state.top_k = st.slider(
            _("返回结果数", "Top-K"),
            min_value=3, max_value=20, value=10,
        )

        if st.button(_("🔄 重建索引", "🔄 Rebuild Index"), use_container_width=True):
            with st.spinner(_("正在重建索引...", "Rebuilding index...")):
                ok = init_rag(
                    force_rebuild=True,
                    embedding_model=st.session_state.embedding_model,
                )
                if ok:
                    st.session_state.stats = get_db_stats()
                    st.success(_("索引重建完成", "Index rebuilt"))
                    st.rerun()

    st.divider()

    # 系统状态
    if st.session_state.initialized:
        st.markdown(f"### 📊 {_('系统状态', 'System Status')}")
        stats = st.session_state.stats
        st.markdown(f"<div class='sidebar-stat'>📄 {_('文档片段', 'Chunks')}: <span>{stats.get('chunks', 0)}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-stat'>📐 {_('向量维度', 'Vector Dim')}: <span>{stats.get('vector_dim', 0)}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-stat'>🏷️ {_('倒排词项', 'Index Terms')}: <span>{stats.get('inverted_index_terms', 0)}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-stat'>🧠 {_('嵌入模型', 'Embedding')}: <span>{stats.get('embedding_model', '-')}</span></div>",
                    unsafe_allow_html=True)

        # 显示支持的模型
        st.markdown(f"### 📦 {_('可用嵌入模型', 'Available Models')}")
        for model_name, model_info in EMBEDDING_MODELS.items():
            active = "✓" if model_name == stats.get("embedding_model") else "○"
            st.markdown(
                f"<div class='sidebar-stat' style='font-size:0.8rem;'>"
                f"{active} {model_name} ({model_info['dim']}d)</div>",
                unsafe_allow_html=True,
            )

    # 重置对话
    st.divider()
    if st.button(_("🗑️ 重置对话", "🗑️ Reset Chat"), use_container_width=True):
        reset_conversation()
        st.session_state.messages = []
        st.rerun()


# ==================== 主界面 ====================
st.markdown(f"# {_('RAG 多轮对话问答系统', 'RAG Multi-turn QA System')}")

# 欢迎信息
if not st.session_state.messages:
    st.markdown(
        f"<div class='info-card'>"
        f"<h4>{_('👋 欢迎使用', '👋 Welcome')}</h4>"
        f"<p>{_('请在左侧边栏配置检索参数，然后在输入框中提问。', 'Configure search parameters in the sidebar, then ask questions below.')}</p>"
        f"<p>{_('当前配置: ', 'Current config: ')}"
        f"<span class='badge badge-blue'>{st.session_state.search_type}</span> "
        f"{_('检索', 'search')} + "
        f"<span class='badge badge-purple'>{st.session_state.reranker}</span> "
        f"{_('重排', 'rerank')} + "
        f"<span class='badge badge-orange'>{st.session_state.embedding_model}</span> "
        f"{_('嵌入', 'embedding')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

# 消息展示
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    elapsed = msg.get("elapsed", 0)

    if role == "user":
        st.markdown(f'<div class="msg-user-label">🧑 {_("你", "You")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-assistant-label">🤖 {_("助手", "Assistant")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-assistant">{content}</div>', unsafe_allow_html=True)

        # 展示元数据
        sources = msg.get("sources", [])
        retrieval_time = msg.get("retrieval_time", 0)
        total_results = msg.get("total_results", 0)

        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"<span class='badge badge-blue'>⚡ {elapsed:.2f}s</span>",
                        unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<span class='badge badge-green'>🔍 {retrieval_time:.3f}s</span>",
                        unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"<span class='badge badge-orange'>📄 {total_results} {_('结果', 'results')}</span>",
                        unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"<span class='badge badge-purple'>🏷️ {st.session_state.search_type}</span>",
                        unsafe_allow_html=True)

        # 来源展开
        if sources:
            with st.expander(_("📚 查看来源文档", "📚 View Source Documents")):
                for i, src in enumerate(sources, 1):
                    text = src.get("text", "")
                    source = src.get("source", "未知")
                    score = src.get("score", 0)
                    st.markdown(
                        f"**{_('来源', 'Source')} {i}**: `{source}` | "
                        f"{_('相关度', 'Relevance')}: {score:.4f}"
                    )
                    st.markdown(f"```\n{text[:300]}...\n```")

    st.markdown("<br>", unsafe_allow_html=True)

# 输入区域
st.markdown("---")
input_col, btn_col = st.columns([6, 1])
with input_col:
    st.text_input(
        _("请输入您的问题...", "Enter your question..."),
        key="query_input",
        placeholder=_("输入问题后按回车...", "Type question and press Enter..."),
        label_visibility="collapsed",
        on_change=handle_query,
    )
with btn_col:
    st.button(_("发送", "Send"), on_click=handle_query, use_container_width=True)
