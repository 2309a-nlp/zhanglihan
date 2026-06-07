# -*- coding: utf-8 -*-
"""
PDF 文档处理模块（水印处理版）
负责 PDF 文件的解析、水印清洗、文本提取和分块处理
使用 pdfplumber 进行 PDF 文本提取
新增特性：
  - PDF 水印自动检测与清洗
  - 页眉/页脚/背景重复文字过滤
  - 水印页自动过滤
  - 支持表格感知分块
"""

import os
import hashlib
import re
import time
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    WATERMARK_REMOVAL_ENABLED,
    WATERMARK_CLEAN_MODE,
    WATERMARK_PAGE_THRESHOLD,
    WATERMARK_KEYWORDS,
)


# ==================== 水印检测与清洗 ====================

# 已知的招股说明书页眉模式（正则）
_HEADER_PATTERNS = [
    r'武汉[^，。]*电子[^，。]*招股[意说]明书',       # "武汉兴图新科电子股份有限公司 招股意向书"
    r'武汉[^，。]*信息[^，。]*招股[意说]明书',        # "武汉力源信息技术股份有限公司 招股意向书"
    r'招股[意说]明书\s*（?申报稿）?',                  # "招股说明书（申报稿）"
    r'声明',                                           # "声 明" 标题
    r'本次发行概况',                                   # 表格标题
    r'发行人声明',                                     # "发行人声明"
    r'创业板风险提示',                                  # "创业板风险提示"
]

# 页脚数字模式（页码）
_FOOTER_PATTERN = re.compile(r'^\s*\d+\s*[-–—]\s*\d+\s*$|^\s*\d+\s*$')


def _detect_page_header_lines(pages_text: List[str]) -> set:
    """
    在所有页面中检测重复出现的行（页眉/页脚/背景水印）
    
    策略：统计每一行文本在所有页面中的出现次数，
    如果某行在超过 80% 的页面中都出现，则视为水印。
    但跳过常识性短文本（如单个数字、页码）。
    
    Returns:
        需要过滤的水印行集合
    """
    line_counter = Counter()
    total_pages = len(pages_text)
    if total_pages < 3:
        return set()

    for text in pages_text:
        seen_lines = set()
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 过滤过短的行（<=2个字符）
            if len(line) <= 2:
                continue
            # 过滤纯数字（页码）
            if line.strip().isdigit():
                continue
            # 过滤纯标点
            if all(c in " []（）().,-—–/\\|·" for c in line):
                continue
            seen_lines.add(line)
        
        for line in seen_lines:
            line_counter[line] += 1

    # 出现在 80%+ 页面的行视为水印
    threshold = max(3, int(total_pages * 0.8))
    watermark_lines = {
        line for line, count in line_counter.items()
        if count >= threshold
    }
    return watermark_lines


def _clean_page_content(text: str, header_lines: set) -> str:
    """
    清洗单页文本：移除水印行 + 清理空白
    
    Args:
        text: 单页原始文本
        header_lines: 需要过滤的水印行集合
    
    Returns:
        清洗后的文本
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 跳过空行
        if not stripped:
            continue
        # 跳过页脚页码
        if _FOOTER_PATTERN.match(stripped):
            continue
        # 跳过水印行
        if stripped in header_lines:
            continue
        cleaned.append(stripped)
    
    return "\n".join(cleaned)


def _has_watermark_keyword(text: str) -> bool:
    """检查文本是否包含水印关键词"""
    lower_text = text.lower()
    for kw in WATERMARK_KEYWORDS:
        if kw.lower() in lower_text:
            return True
    return False


def _is_watermark_only_page(text: str, header_lines: set) -> bool:
    """
    判断某页是否只有水印内容（无实际文档内容）
    
    判定条件：
    1. 清洗后文本为空或极短
    2. 或清洗后文本长度不足原始文本 30%
    3. 或内容只有页眉页码数字
    """
    cleaned = _clean_page_content(text, header_lines)
    if not cleaned.strip():
        return True
    
    # 如果清洗后内容量不足原始文本的 WATERMARK_PAGE_THRESHOLD，视为水印页
    raw_len = len(text.strip())
    clean_len = len(cleaned.strip())
    if raw_len > 0 and clean_len > 0:
        ratio = clean_len / raw_len
        if ratio < WATERMARK_PAGE_THRESHOLD:
            return True
    
    return False


def _get_known_watermark_lines() -> set:
    """返回已知的招股说明书水印/页眉行（基于正则匹配检测）"""
    # 这部分在运行时动态检测，不需要硬编码
    return set()


def _table_to_text(table: List[List[Optional[str]]],
                   page_num: int, table_idx: int) -> str:
    """
    将 pdfplumber 表格转换为连续文本块（保持完整表格结构）
    """
    if not table or len(table) < 2:
        return ""

    lines = [f"【表格 {table_idx + 1} 第{page_num}页】"]

    header = table[0]
    header_clean = [str(c).strip().replace("\n", " ") if c else "" for c in header]
    header_clean = [h for h in header_clean if h]
    if header_clean:
        lines.append(" | ".join(header_clean))

    for row in table[1:]:
        cells = [str(c).strip().replace("\n", " ") if c else "" for c in row]
        if not any(c for c in cells):
            continue
        non_empty = [c for c in cells if c]
        if len(non_empty) == 1 and len(cells) > 3:
            lines.append(f"{non_empty[0]}:")
            continue
        lines.append(" | ".join(cells))

    return "\n".join(lines)


class PDFProcessor:
    """PDF 文档处理器（水印处理 + 表格感知版）"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "。", "！", "？", "；", " ", ""],
            length_function=len,
        )
        # 水印配置
        self.watermark_enabled = WATERMARK_REMOVAL_ENABLED
        self.watermark_mode = WATERMARK_CLEAN_MODE
        self._header_lines = set()  # 自动检测到的水印行（跨页分析后填充）
        self._watermark_pages_removed = 0  # 累计移除的水印页数（供界面显示）

    # ── 水印检测与分析 ──

    def _analyze_watermarks(self, pages_text: List[Dict]) -> set:
        """
        全页分析后检测水印行
        
        pages_text: [{"page_num": int, "text": str}, ...]
        Returns: 检测到的水印行集合
        """
        texts = [p["raw_text"] for p in pages_text]
        self._header_lines = _detect_page_header_lines(texts)
        
        if self._header_lines:
            print(f"  [水印检测] 发现 {len(self._header_lines)} 个跨页重复行（页眉/页脚/背景水印）")
            if self.watermark_mode == "aggressive":
                for line in sorted(list(self._header_lines))[:5]:
                    print(f"    → {line[:80]}")
        
        return self._header_lines

    def _remove_watermarks_from_page(self, text: str) -> str:
        """
        从单页文本中移除水印
        
        分步处理：
        1. 已知页眉/页脚模式清洗
        2. 跨页检测的重复行移除
        3. 水印关键词行移除
        """
        if not text:
            return text

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 1. 移除页脚页码
            if _FOOTER_PATTERN.match(stripped):
                continue

            # 2. 移除检测到的跨页重复行
            if stripped in self._header_lines:
                continue

            # 3. 移除水印关键词行（保守模式下仅移除明确的水印行）
            if self.watermark_mode == "aggressive":
                if _has_watermark_keyword(stripped):
                    continue

            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def _filter_watermark_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        过滤只有水印的页面
        
        Returns:
            保留的页面列表
        """
        if not self.watermark_enabled or not pages:
            return pages

        kept = []
        removed = 0
        for page in pages:
            if _is_watermark_only_page(page["raw_text"], self._header_lines):
                removed += 1
            else:
                kept.append(page)

        if removed > 0:
            print(f"  [水印过滤] 移除了 {removed} 个水印页（无实质内容），保留 {len(kept)} 页")
        self._watermark_pages_removed = removed
        return kept

    # ── 表格感知提取 ──

    def extract_text_with_tables(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        从 PDF 提取文本 + 独立表格块（集成水印清洗）
        
        Returns:
            (清洗后的常规文本, 表格块列表)
        """
        start_time = time.time()
        text_parts = []
        table_chunks = []
        raw_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                # 记录原始文本（用于水印分析）
                page_text = page.extract_text() or ""
                raw_pages.append({
                    "page_num": page_num + 1,
                    "raw_text": page_text.strip(),
                })

                # 1. 提取表格
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = None

                if tables:
                    for t_idx, table in enumerate(tables):
                        tbl_text = _table_to_text(table, page_num + 1, t_idx)
                        if tbl_text and len(tbl_text) > 50:
                            table_chunks.append(tbl_text)

                # 2. 提取页面文本
                if page_text and page_text.strip():
                    text_parts.append({
                        "page_num": page_num + 1,
                        "text": page_text.strip(),
                    })

        # ── 水印检测与清洗 ──
        if self.watermark_enabled:
            # 第一步：跨页分析水印模式
            watermark_start = time.time()
            self._analyze_watermarks(raw_pages)

            # 第二步：过滤水印页
            text_parts = self._filter_watermark_pages(text_parts)

            # 第三步：每页清洗水印行
            for p in text_parts:
                p["text"] = self._remove_watermarks_from_page(p["text"])

            wm_elapsed = time.time() - watermark_start
            print(f"  [水印处理] 耗时 {wm_elapsed:.3f}s")

        # 拼接最终文本
        full_text = "\n\n".join(
            f"【第{p['page_num']}页】\n{p['text']}"
            for p in text_parts if p["text"].strip()
        )

        elapsed = time.time() - start_time
        page_count = sum(1 for p in text_parts if p["text"].strip())
        print(f"  [PDF提取] 原始 {total_pages} 页, "
              f"保留 {page_count} 页有效内容, "
              f"提取文本 {len(full_text)} 字符, "
              f"独立表格 {len(table_chunks)} 个, "
              f"总耗时 {elapsed:.2f}s")
        return full_text, table_chunks

    def split_text_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """将文本分割成块（仅文本部分，不含独立表格）"""
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
        完整处理一个 PDF：水印清洗 → 表格感知提取 → 文本分块
        
        Args:
            pdf_path: PDF 文件路径
        
        Returns:
            完整文本块列表（含独立表格 chunks）
        """
        print(f"[PDF处理] 开始处理: {os.path.basename(pdf_path)}"
              f"{' [水印清洗启用]' if self.watermark_enabled else ''}")
        
        full_text, table_chunks = self.extract_text_with_tables(pdf_path)

        if not full_text.strip():
            raise ValueError(
                f"PDF 文件 {pdf_path} 未能提取到文本内容。"
                f"可能该 PDF 为纯扫描件，需要 OCR 处理。"
            )

        # 1. 常规文本分块
        text_chunks = self.split_text_into_chunks(full_text)

        # 2. 独立表格块
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
        """获取 PDF 文件元数据"""
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
