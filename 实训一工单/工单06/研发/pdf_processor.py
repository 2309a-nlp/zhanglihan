"""
PDF 文档处理模块
支持：PDF 文本提取、分块、清洗
"""

import os
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取纯文本"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        logger.warning("PyMuPDF 未安装，尝试 pdfplumber...")
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
            return text
        except ImportError:
            raise ImportError("请安装 PyMuPDF 或 pdfplumber: pip install pymupdf")


# 模板文本（boilerplate）过滤模式
BOILERPLATE_PATTERNS = [
    re.compile(r"武汉兴图新科电子股份有限公司[\s\u3000]+招股意向书[\s\u3000]+1-1-\d+"),
    re.compile(r"^1-1-\d+[\s\u3000]*$", re.MULTILINE),
    re.compile(r"[\s\u3000]{10,}"),  # 连续空白
    re.compile(r"[·•◆▶▷●○]+"),       # 特殊符号行
]


def clean_text(text: str) -> str:
    """清洗文本，去除模板文本和无用内容"""
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    # 合并多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    """
    将文本分块
    使用递归字符分割，保留段落完整性
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        # 手动分块
        chunks = []
        paragraphs = text.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += "\n\n" + para
            else:
                if current:
                    chunks.append(current.strip())
                current = para
        if current:
            chunks.append(current.strip())
        return chunks


def process_pdfs(data_dir: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[dict]:
    """
    处理目录下所有 PDF 文件

    返回:
        [{"text": "...", "source": "file.pdf", "chunk_id": 0}, ...]
    """
    all_chunks = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"目录中没有 PDF 文件: {data_dir}")
        return []

    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(data_dir, pdf_file)
        logger.info(f"处理PDF: {pdf_file}")

        try:
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(raw_text)
            raw_chunks = chunk_text(cleaned, chunk_size, chunk_overlap)

            for i, chunk in enumerate(raw_chunks):
                if len(chunk.strip()) >= 30:  # 过滤过短的块
                    all_chunks.append({
                        "text": chunk.strip(),
                        "source": pdf_file,
                        "chunk_id": i,
                    })

            logger.info(f"  -> {len(raw_chunks)} 个片段（过滤后 {len([c for c in raw_chunks if len(c.strip())>=30])} 个）")
        except Exception as e:
            logger.error(f"处理 {pdf_file} 失败: {e}")

    return all_chunks
