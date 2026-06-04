# -*- coding: utf-8 -*-
import os
import re
import pickle
import jieba
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
ROLES_DIR = os.path.join(_CURRENT_DIR, "roles")
EMBEDDING_MODEL_PATH = os.path.join(_BACKEND_DIR, "bge-m3")
os.makedirs(ROLES_DIR, exist_ok=True)

ALL_ROLES = ["Medical", "Finance", "Law", "Education", "Psychology"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ==================== 文档清洗：去水印、去页眉页脚、去噪声 ====================

NOISE_PATTERNS = [
    r'^\d+\s*$',
    r'^-?\s*\d+\s*-$',
    r'^第\s*\d+\s*页$',
    r'^第\s*\d+\s*节$',
    r'^page\s*\d+$',
    r'^\d+\s*/\s*\d+$',
    r'^\d+\s*of\s*\d+$',
    r'^免责声明',
    r'^版权[所有声明]',
    r'^版权所有',
    r'^本报告由.*?提供',
    r'^仅供内部使用',
    r'^机密[文件]*',
    r'^第\d+页[，,]\s*共\d+页',
    r'^\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?\s*$',
    r'^https?://',
    r'^www\.',
    r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
    r'^[\d-]+\s*(电话|手机|传真|Tel|Fax|Mobile)',
    r'^[【\[]样本[】\]]',
    r'^[【\[]草稿[】\]]',
    r'^[【\[]测试[】\]]',
    r'^[【\[]仅供预览[】\]]',
    r'^[•·\-]\s*$',
    r'^[─━═]+$',
    r'^[_\-]{3,}$',
]

HEADER_FOOTER_KEYWORDS = [
    '免责声明', '版权声明', '版权所有', '侵权必究',
    '研究报告', '行业报告', '公司报告', '深度报告',
    '请阅读最后一页', '投资评级说明', '重要提示',
    '仅供内部使用', '内部资料', '机密',
    '客服电话', '投诉电话', '官方网站',
    '本报告由', '本公司', '本文件',
]

TABLE_PATTERNS = [
    r'[│┃║▏▎▍▌▋▊▉]',
    r'(?:\|[^|]+\|){2,}',
    r'^[\s]*[┌├└┐┤┘┬┼┴┴]',
]


def is_header_or_footer_line(line: str, page_lines: list, line_idx: int) -> bool:
    line_stripped = line.strip()
    if not line_stripped:
        return False
    if line_idx <= 2 and len(line_stripped) < 50:
        for kw in HEADER_FOOTER_KEYWORDS:
            if kw in line_stripped:
                return True
        if len(page_lines) > 5 and line_idx <= 1:
            for j in range(3, min(8, len(page_lines))):
                if line_stripped == page_lines[j].strip() and len(line_stripped) > 5:
                    return True
    if line_idx >= len(page_lines) - 3 and len(line_stripped) < 50:
        for kw in HEADER_FOOTER_KEYWORDS:
            if kw in line_stripped:
                return True
        if re.match(r'^[\d\-/\s]+$', line_stripped):
            return True
        if re.match(r'^(https?://|www\.)', line_stripped):
            return True
    return False


def is_watermark_line(line: str) -> bool:
    line_stripped = line.strip()
    if not line_stripped:
        return False
    watermark_keywords = ['样本', '草稿', '测试版', '预览', '仅供内部',
                          'confidential', 'draft', 'sample', 'preview',
                          '内部文件', '绝密', '机密']
    for kw in watermark_keywords:
        if kw in line_stripped and len(line_stripped) < 30:
            return True
    return False


def is_table_text(text: str) -> bool:
    for pattern in TABLE_PATTERNS:
        if re.search(pattern, text):
            return True
    lines = text.strip().split('\n')
    if len(lines) >= 3:
        separators = set()
        for line in lines:
            parts = [p for p in line.split() if p.strip()]
            if len(parts) >= 3:
                separators.add(len(parts))
        if len(separators) == 1 and len(lines) >= 3:
            sep_count = separators.pop()
            if sep_count >= 3:
                return True
    return False


def format_table_text(text: str) -> str:
    if is_table_text(text):
        lines = text.strip().split('\n')
        formatted = []
        for line in lines:
            cleaned = re.sub(r'[│┃║▏▎▍▌▋▊▉]', '|', line)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                formatted.append(cleaned)
        if formatted:
            return '[表格]\n' + '\n'.join(formatted)
    return text


def clean_document_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    lines = text.split('\n')
    cleaned_lines = []
    page_lines_list = []
    current_page = []
    for line in lines:
        if line.strip() == '' and len(current_page) > 0:
            page_lines_list.append(current_page)
            current_page = []
        else:
            current_page.append(line)
    if current_page:
        page_lines_list.append(current_page)
    if not page_lines_list:
        page_lines_list = [lines]
    for page_lines in page_lines_list:
        page_cleaned = []
        for idx, line in enumerate(page_lines):
            line_stripped = line.strip()
            if not line_stripped:
                page_cleaned.append('')
                continue
            is_noise = any(re.match(p, line_stripped, re.IGNORECASE) for p in NOISE_PATTERNS)
            if is_noise:
                continue
            if is_header_or_footer_line(line, page_lines, idx):
                continue
            if is_watermark_line(line):
                continue
            line_cleaned = re.sub(r'https?://\S+', '', line_stripped)
            line_cleaned = re.sub(r'www\.\S+', '', line_cleaned)
            line_cleaned = re.sub(r'\s+', ' ', line_cleaned).strip()
            if line_cleaned:
                page_cleaned.append(line_cleaned)
        merged = []
        for ln in page_cleaned:
            if ln == '' and merged and merged[-1] == '':
                continue
            merged.append(ln)
        cleaned_lines.extend(merged)
        cleaned_lines.append('')
    result = '\n'.join(cleaned_lines)
    result = result.strip()
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    return result


# ==================== MinerU PDF 解析 ====================

def parse_mineru_pages(data) -> list:
    """解析 MinerU 输出的内容（支持 JSON 和 Markdown）"""
    from langchain_core.documents import Document
    pages_data = []
    if isinstance(data, dict):
        pages = data.get("pages") or data.get("documents") or [data]
    elif isinstance(data, list):
        pages = data
    else:
        pages = [data]
    for pd in pages:
        if not isinstance(pd, dict):
            continue
        page_num = (pd.get("page_idx") or pd.get("page_number") or pd.get("page_num") or 1)
        if isinstance(page_num, str) and page_num.isdigit():
            page_num = int(page_num)
        elif not isinstance(page_num, int):
            page_num = 1
        text = (pd.get("text") or pd.get("content") or pd.get("markdown") or "")
        # 检查是否有表格（Markdown 表格行）
        has_table = bool(re.search(r'^\|.+\|$', text, re.MULTILINE))
        # 提取图片描述
        image_descs = re.findall(r'!\[([^\]]*)\]\([^)]+\)', text)
        pages_data.append({
            "page_num": page_num,
            "text": text.strip(),
            "has_table": has_table,
            "has_image": len(image_descs) > 0,
            "image_descriptions": image_descs,
        })
    return pages_data


def load_pdf_with_mineru(fpath: str) -> list:
    """使用 MinerU 解析 PDF 文件，优先使用生成的 Markdown 文件"""
    from click.testing import CliRunner
    from langchain_core.documents import Document
    from mineru.cli.client import main
    import tempfile, json, time
    start = time.time()
    # 保存 MinerU 输出到数据目录，保留 .md 和图片供后续使用
    mineru_out_dir = os.path.join(os.path.dirname(fpath), "mineru_output_" + os.path.splitext(os.path.basename(fpath))[0])
    os.makedirs(mineru_out_dir, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(main, ["-p", fpath, "-o", mineru_out_dir, "-b", "pipeline", "-m", "auto", "-f", "False", "-t", "True"])
    if result.exit_code != 0:
        logger.warning("MinerU fail: " + result.output[:200])
        return []

    docs = []
    basename = os.path.splitext(os.path.basename(fpath))[0]
    output_subdir = os.path.join(mineru_out_dir, basename)

    if os.path.isdir(output_subdir):
        # 1. 优先使用 MinerU 生成的 Markdown 文件（含表格和图片引用）
        md_path = None
        for root, dirs, files in os.walk(output_subdir):
            for fn in sorted(files):
                if fn.endswith(".md"):
                    md_path = os.path.join(root, fn)
                    break
            if md_path:
                break

        if md_path:
            try:
                md_text = open(md_path, "r", encoding="utf-8").read()
                if md_text.strip():
                    # 提取图片描述：MinerU 在图片下方生成 "图xxx" 描述
                    md_lines = md_text.split(chr(10))
                    new_md_lines = []
                    i = 0
                    while i < len(md_lines):
                        line = md_lines[i]
                        if re.match(r'^\s*!\[.*\]\([^)]+\)', line):
                            if i + 1 < len(md_lines):
                                next_line = md_lines[i + 1].strip()
                                if next_line and len(next_line) < 100 and not next_line.startswith("#"):
                                    new_md_lines.append("[图片: " + next_line + "]")
                                    i += 2
                                    continue
                            alt_match = re.search(r'!\[([^\]]*)\]\([^)]+\)', line)
                            alt_text = alt_match.group(1) if alt_match else ""
                            if alt_text.strip():
                                new_md_lines.append("[图片: " + alt_text.strip() + "]")
                            else:
                                new_md_lines.append("[图片]")
                            i += 1
                        else:
                            new_md_lines.append(line)
                            i += 1
                    text_with_images = chr(10).join(new_md_lines)

                    cleaned = clean_document_text(text_with_images)
                    if cleaned:
                        has_table = bool(re.search(r'^\|.+\|$', cleaned, re.MULTILINE))
                        docs.append(Document(
                            page_content=cleaned,
                            metadata={
                                "page": 1,
                                "source": os.path.basename(fpath),
                                "has_table": has_table,
                                "has_image": True,
                            }
                        ))
                        tbl_str = "是" if has_table else "否"
                        logger.info(
                            "  MinerU Markdown: %s (%d chars, 表格=%s, 图片=是)"
                            % (os.path.basename(md_path), len(cleaned), tbl_str)
                        )
            except Exception as e:
                logger.warning("  Markdown 读取失败: %s" % str(e))

        # 2. Fallback 到 JSON
        if not docs:
            json_path = None
            for fn in sorted(os.listdir(output_subdir)):
                if fn.endswith(".json"):
                    json_path = os.path.join(output_subdir, fn)
                    break
            if json_path:
                try:
                    data = json.load(open(json_path, "r", encoding="utf-8"))
                    pages_data = parse_mineru_pages(data)
                    for pd_info in pages_data:
                        if pd_info["text"]:
                            cleaned = clean_document_text(pd_info["text"])
                            if cleaned:
                                docs.append(Document(
                                    page_content=cleaned,
                                    metadata={
                                        "page": pd_info["page_num"],
                                        "source": os.path.basename(fpath),
                                        "has_table": pd_info["has_table"],
                                    }
                                ))
                except Exception as e:
                    logger.warning(f"  JSON 解析失败: {e}")

    logger.info("MinerU: %.1fs %d pages (已清洗去噪)" % (time.time()-start, len(docs)))
    return docs

def load_documents_for_role(role_name: str):
    role_data_dir = os.path.join(DATA_DIR, role_name)
    if not os.path.isdir(role_data_dir):
        logger.warning(f"角色目录不存在: {role_data_dir}")
        return []

    docs = []
    for fname in os.listdir(role_data_dir):
        if fname.startswith('mineru') and os.path.isdir(os.path.join(role_data_dir, fname)):
            continue
        fpath = os.path.join(role_data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith(".pdf"):
            try:
                pdf_docs = load_pdf_with_mineru(fpath)
                if pdf_docs:
                    for d in pdf_docs:
                        d.metadata["source"] = f"{role_name}/{fname}"
                        d.metadata["domain"] = role_name
                    docs.extend(pdf_docs)
                    logger.info(f"PDF(MinerU): {fname} - {len(pdf_docs)} items")
                else:
                    logger.warning(f"MinerU empty, fallback: {fname}")
                    from langchain_community.document_loaders import PDFPlumberLoader
                    loader = PDFPlumberLoader(fpath)
                    pdf_docs = loader.load()
                    for d in pdf_docs:
                        d.metadata["source"] = f"{role_name}/{fname}"
                        d.metadata["domain"] = role_name
                        d.page_content = clean_document_text(d.page_content)
                    docs.extend(pdf_docs)
                    logger.info(f"PDF(fallback): {fname} - {len(pdf_docs)} pages")
            except Exception as e:
                logger.warning(f"MinerU fail {fname}: {e}, fallback")
                try:
                    from langchain_community.document_loaders import PDFPlumberLoader
                    loader = PDFPlumberLoader(fpath)
                    pdf_docs = loader.load()
                    for d in pdf_docs:
                        d.metadata["source"] = f"{role_name}/{fname}"
                        d.metadata["domain"] = role_name
                        d.page_content = clean_document_text(d.page_content)
                    docs.extend(pdf_docs)
                    logger.info(f"PDF(fallback): {fname} - {len(pdf_docs)} pages")
                except Exception as e2:
                    logger.warning(f"PDF load fail {fname}: {e2}")
        elif fname.lower().endswith(".docx"):
            try:
                loader = Docx2txtLoader(fpath)
                docx_docs = loader.load()
                for d in docx_docs:
                    d.metadata["source"] = f"{role_name}/{fname}"
                    d.metadata["domain"] = role_name
                    d.page_content = clean_document_text(d.page_content)
                docs.extend(docx_docs)
                logger.info(f"  DOCX: {fname} ({len(docx_docs)} 页)")
            except Exception as e:
                logger.warning(f"  DOCX 加载失败 {fname}: {e}")
    return docs


def build_role_index(role_name: str):
    logger.info(f"\n==== 开始构建 [{role_name}] 索引 ====")
    documents = load_documents_for_role(role_name)
    if not documents:
        logger.warning(f"[{role_name}] 没有加载到任何文档，跳过")
        return False

    logger.info(f"共加载 {len(documents)} 个文档片段")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"分块完成: {len(chunks)} 个文本块")

    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    chunk_texts = [c.page_content for c in chunks]

    vectorstore = FAISS.from_texts(
        texts=chunk_texts,
        embedding=embeddings_model,
        metadatas=[c.metadata for c in chunks]
    )

    tokenized_corpus = [list(jieba.cut(t)) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    role_index_dir = os.path.join(ROLES_DIR, role_name)
    os.makedirs(role_index_dir, exist_ok=True)

    vectorstore.save_local(role_index_dir)
    with open(os.path.join(role_index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(role_index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    logger.info(f"[{role_name}] 索引构建完成! ({len(chunks)} chunks)")
    return True


def main():
    logger.info("开始为所有角色构建知识库索引...")
    built, failed = [], []
    for role in ALL_ROLES:
        if build_role_index(role):
            built.append(role)
        else:
            failed.append(role)
    logger.info(f"成功: {built}")
    if failed:
        logger.warning(f"失败: {failed}")


if __name__ == "__main__":
    main()
