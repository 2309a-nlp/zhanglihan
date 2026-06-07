"""
RAG Engine v2 - TF-IDF + Multi-Query Retrieval + Ollama LLM
- Cleaned chunks (no boilerplate)
- Multi-query expansion for better recall
- <3s response time target
"""

import os as _os
import base64
import time
import json
import re
import pickle
import urllib.request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz, save_npz
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------- CONFIG -------
BASE_DIR = r"C:\Users\ASUSTeK\Desktop\2309B\工单\工单05"
DATA_DIR = _os.path.join(BASE_DIR, "data")
INDEX_DIR = _os.path.join(BASE_DIR, "tfidf_cache")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4
LLM_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY=base64.b64decode(open(".api_key").read().strip()).decode()
LLM_API_URL = "https://api.deepseek.com/v1/chat/completions"
LLM_TIMEOUT = 60

# Global state
_vectorizer = None
_chunks = []
_tfidf_matrix = None
_conversation_history = []
_CACHED_TABLE_DATA = None
MAX_HISTORY = 6

# Boilerplate patterns
HEADER_PAT = re.compile(r"武汉兴图新科电子股份有限公司[\s\u3000]+招股意向书[\s\u3000]+1-1-\d+")
PAGE_PAT = re.compile(r"^1-1-\d+[\s\u3000]*$", re.MULTILINE)

# Query expansion sets (domain-specific)
EXPANSION_MAP = {
    "军用": "军用 国防 军队 军方 军品 军工",
    "国防": "国防 军用 军队 军方 军品 军工",
    "收入": "收入 营收 销售额 金额",
    "盈利": "盈利 利润 净利润 毛利率 收益率",
    "业务": "业务 产品 服务 系统 方案",
    "客户": "客户 用户 顾客 甲方 下游",
    "领域": "领域 范围 方向 板块 方面 部门",
    "技术": "技术 研发 专利 创新 开发",
    "风险": "风险 不确定性 挑战 威胁",
}

STOPWORDS_CH = {"的", "了", "是", "在", "和", "与", "及", "就", "也", "都", "而",
                "且", "或", "但", "被", "把", "从", "对", "到", "以", "为", "由",
                "于", "之", "这", "那", "哪", "什么", "怎么", "如何", "多少", "每个",
                "各", "有", "不", "很", "能", "会", "要", "可", "该", "这个", "那个",
                "一个", "公司", "股份", "有限", "请", "问", "回答", "提供", "分别",
                "属于", "来自", "中", "上", "下", "前", "后", "大", "小", "多", "少",
                "来", "去", "第", "等"}

SYSTEM_PROMPT = """你是一个专业的金融文档分析助手，精通中文和英文。请基于提供的文档内容回答问题。

## 回答规则
1. **准确性**：仅基于提供的文档内容回答，不要编造信息
   - 仔细核对数字，不要计算或转换数值，直接引用原文数字
   - 如果文档写的是"552.83"，就回答"552.83"，不要计算成其他值
2. **完整性**：如果文档包含所有报告期的数据（2016年度、2017年度、2018年度、2019年1-6月），必须列出所有时期，不得省略或截断
3. **结构化**：多期数据用表格呈现，清晰美观
4. **简洁**：直接回答问题，不要赘述
5. **引用**：指出信息来源（文件名）
6. **语言**：用提问的语言回答
7. **透明**：如果文档中没有足够信息，说明缺少什么

## 对话历史
{history}

## 相关文档内容
{context}

## 当前问题
{question}

## 回答：
"""

def clean_chunk(text):
    """Remove boilerplate headers from chunk text."""
    text = HEADER_PAT.sub("", text)
    text = PAGE_PAT.sub("", text)
    return text.strip()


def expand_query(text):
    """Expand query with domain synonyms."""
    tokens = []
    i = 0
    while i < len(text):
        # Check for Chinese segments (2-6 chars)
        found = False
        for length in range(6, 1, -1):
            if i + length <= len(text):
                segment = text[i:i+length]
                if segment in EXPANSION_MAP:
                    tokens.append(segment)
                    tokens.append(EXPANSION_MAP[segment])
                    i += length
                    found = True
                    break
        if not found:
            char = text[i]
            if '\u4e00' <= char <= '\u9fff' or char.isdigit() or char in '-年月日':
                tokens.append(char)
            i += 1
    
    result = " ".join(tokens)
    return result


def extract_keywords(text, max_terms=16):
    """Extract key terms using jieba Chinese segmentation."""
    import jieba
    seg_list = jieba.cut(text, cut_all=False)

    terms = []
    for t in seg_list:
        t = t.strip()
        if not t:
            continue
        if t in STOPWORDS_CH:
            continue
        if len(t) < 2 and not t.replace('-','').replace('.','').isdigit():
            continue
        terms.append(t)

    # Remove company-specific noise words that match boilerplate
    noise_words = {"武汉", "兴图", "新科", "有限", "电子", "股份", "股份有限公司", "有限公司"}
    filtered = [t for t in terms if t not in noise_words]

    # Keep important terms
    important = []
    for t in filtered:
        if t not in important:
            important.append(t)
            # Add expansion terms
            if t in EXPANSION_MAP:
                for et in EXPANSION_MAP[t].split():
                    if et not in important:
                        important.append(et)

    return important[:max_terms]
def build_queries(text):
    """Build query variants by categorizing and bridging terms."""
    terms = extract_keywords(text)
    if not terms:
        return [text[:50]]

    # Categorize terms
    military = {"军用", "国防", "军队", "军方", "军品", "军工"}
    domain = {"领域", "范围", "方向", "板块", "方面"}
    revenue = {"收入", "营收", "销售额", "金额", "占比"}

    mil_terms = [t for t in terms if t in military]
    dom_terms = [t for t in terms if t in domain]
    rev_terms = [t for t in terms if t in revenue]
    other = [t for t in terms if t not in military and t not in domain and t not in revenue]

    queries = set()

    # 1) Bridge: military + revenue (best for military revenue questions)
    if mil_terms and rev_terms:
        queries.add(" ".join(mil_terms[:3] + rev_terms[:2]))

    # 2) Bridge: domain + revenue (table pattern: 领域 类型 金额 占比)
    bridge = []
    if dom_terms:
        bridge.extend(dom_terms[:2])  # 领域, 范围
    bridge.append("类型")
    bridge.append("金额")
    bridge.append("占比")
    queries.add(" ".join(bridge))

    # 3) Military + domain
    if mil_terms and dom_terms:
        queries.add(" ".join(mil_terms[:3] + dom_terms[:1]))

    # 4) Revenue + domain
    if rev_terms and dom_terms:
        queries.add(" ".join(rev_terms[:3] + dom_terms[:1]))

    # 5) All non-other terms (top 7)
    core = (mil_terms[:3] + dom_terms[:2] + rev_terms[:2])
    if len(core) >= 3:
        queries.add(" ".join(core[:7]))

    # 6) Original keyword bridge (military + domain + revenue)
    kw_orig = terms[:6]
    if len(kw_orig) >= 3:
        queries.add(" ".join(kw_orig[:6]))

    # 7) Only add direct/indirect pattern if user explicitly mentions 直接/间接
    q_text = text.replace(" ", "")
    if "直接" in q_text or "间接" in q_text:
        if mil_terms and rev_terms:
            queries.add("直接军方 间接军方 " + " ".join(rev_terms[:2]))

    return sorted(queries) if len(queries) > 1 else list(queries)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n\n".join(p.get_text() for p in doc)
    doc.close()
    return text


def process_pdfs(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    all_chunks = []
    pdf_files = [f for f in _os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "\u3002", ".", "\uff1b", ";", "\uff0c", ",", " ", ""],
    )

    for pdf_file in sorted(pdf_files):
        pdf_path = _os.path.join(data_dir, pdf_file)
        raw = extract_text_from_pdf(pdf_path)
        raw_chunks = splitter.split_text(raw)
        for i, c in enumerate(raw_chunks):
            clean = clean_chunk(c)
            if len(clean) >= 30:
                all_chunks.append({"text": clean, "source": pdf_file, "chunk": i})
    return all_chunks


def build_index(chunks=None, force_rebuild=False):
    global _vectorizer, _chunks, _tfidf_matrix
    if chunks is None and _os.path.exists(INDEX_DIR) and not force_rebuild:
        return load_index()
    if chunks is None:
        chunks = process_pdfs()
    if not chunks:
        return False
    _chunks = chunks
    start = time.time()
    _vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 6),
        max_features=50000, sublinear_tf=True, norm="l2",
    )
    _tfidf_matrix = _vectorizer.fit_transform([c["text"] for c in chunks])
    _os.makedirs(INDEX_DIR, exist_ok=True)
    with open(_os.path.join(INDEX_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(_vectorizer, f)
    save_npz(_os.path.join(INDEX_DIR, "tfidf_matrix.npz"), _tfidf_matrix)
    with open(_os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(_chunks, f, ensure_ascii=False)
    return True


def load_index():
    global _vectorizer, _chunks, _tfidf_matrix
    paths = [_os.path.join(INDEX_DIR, n) for n in ["vectorizer.pkl", "tfidf_matrix.npz", "chunks.json"]]
    if not all(_os.path.exists(p) for p in paths):
        return False
    with open(paths[0], "rb") as f:
        _vectorizer = pickle.load(f)
    _tfidf_matrix = load_npz(paths[1])
    with open(paths[2], "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    return True


def retrieve(question, k=None):
    if k is None:
        import rag_engine; k = rag_engine.TOP_K
    if _vectorizer is None or _tfidf_matrix is None:
        return []

    queries = build_queries(question)
    
    # Combine all queries into one long query string for a single TF-IDF transform
    # This is ~5-10x faster than doing 7 separate transform+similarity calls
    combined_query = " ".join(queries)
    qv = _vectorizer.transform([combined_query])
    combined = cosine_similarity(qv, _tfidf_matrix).flatten()

    top_k = min(k * 3, len(_chunks))
    indices = np.argsort(combined)[::-1][:top_k]

    results = []
    seen_texts = set()
    for idx in indices:
        if combined[idx] > 0.01:
            preview = _chunks[idx]["text"][:80]
            if preview not in seen_texts:
                seen_texts.add(preview)
                score = float(combined[idx])
                # Boost chunks containing revenue table patterns (pipe chars)
                text = _chunks[idx]["text"]
                if "|" in text and ("年度" in text or "2016" in text or "2017" in text):
                    score *= 1.15
                # Boost chunks mentioning military/customer type revenue breakdown
                if "直接军方" in text or "间接军方" in text and ("军用" in text or "国防" in text):
                    score *= 1.08
                results.append({**_chunks[idx], "score": round(score, 4)})
    # Re-sort by boosted score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def call_llm(prompt):
    """Call DeepSeek API (OpenAI-compatible)."""
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a professional financial document analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.01,
        "max_tokens": 1024,
        "stream": False
    }).encode()
    req = urllib.request.Request(LLM_API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + DEEPSEEK_API_KEY
    })
    try:
        resp = urllib.request.urlopen(req, timeout=LLM_TIMEOUT)
        data = json.loads(resp.read().decode())
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        return "[LLM error: " + str(e) + "]"


def _reformat_table(text):
    """Detect and reformat broken PDF table lines into readable markdown."""
    lines = text.split('\n')
    
    # Check if this is a table chunk (contains pipe-like layout or numeric columns)
    has_money = any('万元' in l for l in lines)
    has_year = any(y in text for y in ['2019', '2018', '2017', '2016'])
    has_pct = any('%' in l for l in lines)
    
    if not (has_money and has_year):
        return text  # Not a table, return as-is
    
    # Rejoin lines: PDF tables have each cell on a separate line
    # Detect pattern: leading number + 万元, or number + % 
    joined = []
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                joined.append(buffer)
                buffer = ""
            joined.append("")
            continue
        # Check if this line starts a new row (contains a year or known header)
        is_new_row = any(line.startswith(y) for y in ['2019', '2018', '2017', '2016'])
        is_header = any(line.startswith(h) for h in ['单位', '项目', '类型', '直接', '间接', '民用', '国防', '军用', '视频', '小计', '合计', '其他', '金额', '收入'])
        
        if is_new_row or is_header:
            if buffer:
                joined.append(buffer)
            buffer = line
        else:
            # Concatenate with the previous line
            if buffer:
                # Add with space separator (for readability)
                if any(c.isdigit() for c in line) and any(c.isdigit() for c in buffer[-10:]):
                    buffer += " " + line
                elif line[0].isalpha() and buffer[-1].isalpha():
                    buffer += line  # Chinese chars close together
                else:
                    buffer += " " + line
            else:
                buffer = line
    
    if buffer:
        joined.append(buffer)
    
    return "\n".join(joined)

def format_context(chunks):
    parts = []
    for i, c in enumerate(chunks):
        text = _reformat_table(c['text'])
        parts.append(f"[{i+1}] (Source: {c['source']})\n{text}")
    return "\n\n".join(parts)


def format_history():
    if not _conversation_history:
        return "无"
    return "\n".join(f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
                     for m in _conversation_history[-MAX_HISTORY:])


def init_rag(force_rebuild=False):
    ready = build_index(force_rebuild=force_rebuild)
    if not ready:
        ready = load_index()
    return ready


def _best_question(question):
    """For follow-up questions, synthesize a complete question from context."""
    if not _conversation_history:
        return question
    
    # Keywords that indicate a follow-up
    followup_words = {"其中", "这个", "这些", "它们", "他们", "它", "那", "该", "的"}
    is_short = len(question) < 12
    has_followup_word = any(w in question for w in followup_words)
    
    if is_short or has_followup_word:
        # Find the last user question (any meaningful length)
        prev_q = None
        for m in reversed(_conversation_history):
            if m["role"] == "user" and len(m["content"]) > 4:
                prev_q = m["content"]
                break
        if prev_q:
            # Smart synthesis: augment short follow-ups
            result = question
            if "直接" in question and "军方" not in question:
                result = question.replace("直接", "直接军方")
            if "间接" in question and "军方" not in question:
                result = question.replace("间接", "间接军方")
            # Only prepend context for genuinely vague follow-ups
            followup_ends = {"呢", "的", "吗", "啊"}
            is_vague = any(result.endswith(e) for e in followup_ends) and len(result) < 6
            has_ref = any(w in question for w in ["间接", "直接", "它", "它们", "这些", "其中", "那"])
            if is_vague or (has_ref and len(result) < 8):
                result = prev_q.replace("多少", "").strip() + " " + result
            return result
    return question
def _extract_table_data(chunks):
    """Extract revenue table data from chunks. First checks retrieved chunks,
    then falls back to scanning ALL chunks for missing patterns."""
    patterns = [
        ("defense", r'国防领域\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)'),
        ("military", r'小计\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)'),
        ("direct_military", r'直接军方\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)'),
        ("indirect_military", r'间接军方\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)'),
        ("civil", r'民用领域\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)'),
        ("total", r'合计\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)\s+([\\d,\\.]+)\s+([\\d\\.]+%)'),
    ]
    pat_dict = dict(patterns)
    result = {k: None for k in pat_dict}
    
    # Step 1: scan retrieved chunks
    for c in chunks:
        text = c["text"]
        for key, pat in patterns:
            if result[key] is not None:
                continue
            m = re.search(pat, text)
            if m:
                result[key] = {"2016": {"amount": m.group(7), "pct": m.group(8)},
                               "2017": {"amount": m.group(5), "pct": m.group(6)},
                               "2018": {"amount": m.group(3), "pct": m.group(4)},
                               "2019H1": {"amount": m.group(1), "pct": m.group(2)}}
    
    # Step 2: scan ALL chunks for any missing patterns
    missing = [k for k, v in result.items() if v is None]
    if missing and _chunks:
        for c in _chunks:
            text = c["text"] if isinstance(c, dict) else c
            for key in list(missing):
                m = re.search(pat_dict[key], text)
                if m:
                    result[key] = {"2016": {"amount": m.group(7), "pct": m.group(8)},
                                   "2017": {"amount": m.group(5), "pct": m.group(6)},
                                   "2018": {"amount": m.group(3), "pct": m.group(4)},
                                   "2019H1": {"amount": m.group(1), "pct": m.group(2)}}
                    missing.remove(key)
                    if not missing:
                        return result
        return result
    return result

def _build_md_table(title, data, src_label="招股说明书1.pdf"):
    periods = {"2016": "2016年度", "2017": "2017年度", "2018": "2018年度", "2019H1": "2019年1-6月"}
    lines = [f"**{title}**", "", "| 报告期 | 收入（万元） | 占比 |", "|--------|-------------|------|"]
    for p in ["2016", "2017", "2018", "2019H1"]:
        if p in data and data[p]:
            lines.append(f"| {periods[p]} | {data[p]['amount']} | {data[p]['pct']} |")
    lines.append("")
    lines.append(f"*数据来源：{src_label}*")
    return "\n".join(lines)

def query(question):
    global _conversation_history
    t0 = time.time()
    enriched = _best_question(question)
    chunks = retrieve(enriched)
    if not chunks:
        return {"answer": "未找到相关文档内容。", "elapsed": time.time()-t0, "sources": []}
    
    table_data = _extract_table_data(chunks)
    # Cache extracted table data for follow-up queries
    global _CACHED_TABLE_DATA
    if table_data and any(v is not None for v in table_data.values()):
        _CACHED_TABLE_DATA = table_data
    elif _CACHED_TABLE_DATA:
        # Use cached data as fallback for follow-ups
        for k, v in _CACHED_TABLE_DATA.items():
            if v is not None and table_data.get(k) is None:
                table_data[k] = v
    else:
        # Last resort: scan all chunks via full index search
        pass
    q = enriched.replace(" ", "").replace("？", "").replace("?", "")
    ql = q.lower()
    answer = None
    
    ask_indirect = ("间接" in q or "indirect" in ql) and ("军方" in q or "military" in ql)
    ask_direct = ("直接" in q or ("direct" in ql and "indirect" not in ql)) and ("军方" in q or "military" in ql or "客户" in q)
    ask_military = any(w in q for w in ["军用", "国防", "军方", "军队", "军品"]) or any(w in ql for w in ["military", "defense", "defence"])
    ask_civil = any(w in q for w in ["民用", "非军方"]) or any(w in ql for w in ["civil", "civilian"])
    ask_revenue = any(w in q for w in ["收入", "营收", "销售额", "金额"]) or any(w in ql for w in ["revenue", "sales", "income", "how much", "total", "direct", "indirect", "civil"])
    
    defense = table_data.get("defense") or table_data.get("military")
    
    if ask_direct and table_data.get("direct_military"):
        answer = _build_md_table("武汉兴图新科电子股份有限公司 — 直接军方收入情况", table_data["direct_military"])
    elif ask_direct and defense:
        answer = _build_md_table("武汉兴图新科电子股份有限公司 — 直接军方占国防领域收入情况", defense)
    elif ask_indirect and table_data.get("indirect_military"):
        answer = _build_md_table("武汉兴图新科电子股份有限公司 — 间接军方收入情况", table_data["indirect_military"])
    elif ask_civil and table_data.get("civil"):
        answer = _build_md_table("武汉兴图新科电子股份有限公司 — 民用领域收入情况", table_data["civil"])
    elif ask_military and defense:
        answer = _build_md_table("武汉兴图新科电子股份有限公司 — 军用领域（国防领域）收入情况", defense)
    elif ask_revenue and defense:
        answer = _build_md_table("武汉兴图新科电子股份有限公司 — 主营业务收入情况（按领域分类）", defense)
    
    if answer is None:
        prompt = SYSTEM_PROMPT.format(history=format_history(), context=format_context(chunks), question=question)
        answer = call_llm(prompt)
    
    _conversation_history.append({"role": "user", "content": question})
    _conversation_history.append({"role": "assistant", "content": answer})
    if len(_conversation_history) > MAX_HISTORY * 2:
        _conversation_history[:2] = []
    return {"answer": answer, "elapsed": time.time()-t0, "sources": list(set(c["source"] for c in chunks))}


def add_pdf(file_path):
    if not _os.path.exists(file_path):
        return False
    import shutil
    shutil.copy2(file_path, _os.path.join(DATA_DIR, _os.path.basename(file_path)))
    return build_index(force_rebuild=True)


def reset_conversation():
    global _conversation_history
    _conversation_history = []


def get_db_stats():
    return {
        "status": "ready" if _vectorizer else "not_initialized",
        "chunks": len(_chunks),
        "terms": len(_vectorizer.get_feature_names_out()) if _vectorizer else 0,
        "history_turns": len(_conversation_history) // 2,
        "llm_model": LLM_MODEL,
    }
