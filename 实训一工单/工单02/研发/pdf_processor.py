# -*- coding: utf-8 -*-
"""
PDF 文档处理模块
负责 PDF 文件的解析、文本提取和分块处理
使用 pdfplumber 进行 PDF 文本提取
支持表格感知分块：检测财务数据表格并保持其完整性
"""

import os
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
import time

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _table_to_text(table: List[List[Optional[str]]],
                   page_num: int, table_idx: int) -> str:
    """
    将 pdfplumber 表格转换为连续文本块（保持完整表格结构）

    例：
        （表格 第129页）
        公司主营业务收入按客户列示（万元）：
        国防领域: 2018年度 18,780.67万元(94.84%) | 2017年度 14,414.16万元(97.31%) | 2016年度 6,464.51万元(82.10%)
        民用领域: 2018年度 1,021.81万元(5.16%) | 2017年度 398.56万元(2.69%) | 2016年度 1,409.12万元(17.90%)
        合计: 2018年度 19,802.48万元(100.00%) | 2017年度 14,812.72万元(100.00%) | 2016年度 7,873.63万元(100.00%)
    """
    if not table or len(table) < 2:
        return ""

    lines = [f"【表格 {table_idx + 1} 第{page_num}页】"]

    # 第一行是表头
    header = table[0]
    header_clean = [str(c).strip().replace("\n", " ") if c else "" for c in header]
    # 跳过空表头
    header_clean = [h for h in header_clean if h]
    if header_clean:
        lines.append(" | ".join(header_clean))

    # 数据行
    for row in table[1:]:
        cells = [str(c).strip().replace("\n", " ") if c else "" for c in row]
        # 跳过全空行
        if not any(c for c in cells):
            continue
        # 如果这一行只有标签（如 "类型"），将其作为表头的一部分
        non_empty = [c for c in cells if c]
        if len(non_empty) == 1 and len(cells) > 3:
            lines.append(f"{non_empty[0]}:")
            continue
        lines.append(" | ".join(cells))

    return "\n".join(lines)



class PDFProcessor:
    """PDF 文档处理器（表格感知版）"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        初始化 PDF 处理器

        Args:
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 文本块重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 不含中文逗号（避免砍断财务数字串）
            # 不含"\\n"（避免切断财务表格行）
            separators=["\n\n", "。", "！", "？", "；", " ", ""],
            length_function=len,
        )

    # ── 表格感知提取 ──

    def extract_text_with_tables(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        从 PDF 提取文本 + 独立表格块

        返回:
            (常规文本, 表格块列表)
            表格块以独立 chunk 形式返回，不经过文本分割器
        """
        start_time = time.time()
        text_parts = []
        table_chunks = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                # 1. 尝试提取表格
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = None

                if tables:
                    for t_idx, table in enumerate(tables):
                        tbl_text = _table_to_text(table, page_num + 1, t_idx)
                        if tbl_text and len(tbl_text) > 50:
                            table_chunks.append(tbl_text)

                # 2. 提取页面文本（含表格中的文字）
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(f"【第{page_num + 1}页】\n{page_text}")

        full_text = "\n\n".join(text_parts)
        elapsed = time.time() - start_time
        print(f"  [PDF提取] 共 {total_pages} 页, 提取文本 {len(full_text)} 字符, "
              f"独立表格 {len(table_chunks)} 个, 耗时 {elapsed:.2f}s")
        return full_text, table_chunks

    def split_text_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本分割成块（仅文本部分，不含独立表格）

        Args:
            text: 要分割的文本（不含表格区域）

        Returns:
            文本块列表
        """
        start_time = time.time()
        chunks = self.text_splitter.create_documents([text])
        chunk_data = []

        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "chunk_id": i,
                "content": chunk.page_content,
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "type": "text",
                }
            }
            chunk_data.append(chunk_dict)

        elapsed = time.time() - start_time
        print(f"  [文本分块] 共生成 {len(chunk_data)} 个文本块, "
              f"耗时 {elapsed:.3f}s")
        return chunk_data

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        完整处理一个 PDF：表格感知提取 → 文本分块 + 独立表格块

        Args:
            pdf_path: PDF 文件路径

        Returns:
            完整文本块列表（含独立表格 chunks）
        """
        print(f"[PDF处理] 开始处理: {os.path.basename(pdf_path)}")
        full_text, table_chunks = self.extract_text_with_tables(pdf_path)

        if not full_text.strip():
            raise ValueError(f"PDF 文件 {pdf_path} 未能提取到文本内容")

        # 1. 常规文本分块
        text_chunks = self.split_text_into_chunks(full_text)

        # 2. 独立表格块（保留完整结构，不分块）
        next_id = len(text_chunks)
        for tbl_text in table_chunks:
            text_chunks.append({
                "chunk_id": next_id,
                "content": tbl_text,
                "metadata": {
                    "chunk_index": next_id,
                    "total_chunks": next_id + 1,
                    "type": "table",
                }
            })
            next_id += 1

        print(f"[PDF处理] 完成: {os.path.basename(pdf_path)}, "
              f"生成 {len(text_chunks)} 个文本块 "
              f"(其中表格块 {len(table_chunks)} 个)")
        return text_chunks

    @staticmethod
    def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
        """
        获取 PDF 文件元数据（使用 pdfplumber）

        Args:
            pdf_path: PDF 文件路径

        Returns:
            元数据字典
        """
        file_size = os.path.getsize(pdf_path)
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata
            total_pages = len(pdf.pages)

        return {
            "filename": os.path.basename(pdf_path),
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "total_pages": total_pages,
            "title": metadata.get("Title", "未知") if metadata else "未知",
            "author": metadata.get("Author", "未知") if metadata else "未知",
            "file_hash": hashlib.md5(open(pdf_path, "rb").read()).hexdigest(),
        }
