# -*- coding: utf-8 -*-
"""
PDF 文档处理模块（高稳定性版）
负责 PDF 文件的解析、水印清洗、文本提取和分块处理
新增特性：
  - 多策略 PDF 解析（pdfplumber 失败时降级）
  - 健壮的异常处理（损坏 PDF、加密 PDF 等）
  - 清晰的用户友好的中文错误信息
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

_HEADER_PATTERNS = [
    r'武汉[^，。]*电子[^，。]*招股[意说]明书',
    r'武汉[^，。]*信息[^，。]*招股[意说]明书',
    r'招股[意说]明书\\s*（?申报稿）?',
    r'声明',
    r'本次发行概况',
    r'发行人声明',
    r'创业板风险提示',
]

_FOOTER_PATTERN = re.compile(r'^\\s*\\d+\\s*[-–—]\\s*\\d+\\s*$|^\\s*\\d+\\s*$')


def _detect_page_header_lines(pages_text: List[str]) -> set:
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
            if len(line) <= 2:
                continue
            if line.strip().isdigit():
                continue
            if all(c in " []（）().,-—–/\\|·" for c in line):
                continue
            seen_lines.add(line)

        for line in seen_lines:
            line_counter[line] += 1

    threshold = max(3, int(total_pages * 0.8))
    watermark_lines = {
        line for line, count in line_counter.items()
        if count >= threshold
    }
    return watermark_lines


def _clean_page_content(text: str, header_lines: set) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _FOOTER_PATTERN.match(stripped):
            continue
        if stripped in header_lines:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def _has_watermark_keyword(text: str) -> bool:
    lower_text = text.lower()
    for kw in WATERMARK_KEYWORDS:
        if kw.lower() in lower_text:
            return True
    return False


def _is_watermark_only_page(text: str, header_lines: set) -> bool:
    cleaned = _clean_page_content(text, header_lines)
    if not cleaned.strip():
        return True
    raw_len = len(text.strip())
    clean_len = len(cleaned.strip())
    if raw_len > 0 and clean_len > 0:
        ratio = clean_len / raw_len
        if ratio < WATERMARK_PAGE_THRESHOLD:
            return True
    return False


def _get_known_watermark_lines() -> set:
    return set()


def _table_to_text(table: List[List[Optional[str]]],
                   page_num: int, table_idx: int) -> str:
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
    """PDF 文档处理器（高稳定性版 — 多策略解析 + 健壮异常处理）"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "。", "！", "？", "；", " ", ""],
            length_function=len,
        )
        self.watermark_enabled = WATERMARK_REMOVAL_ENABLED
        self.watermark_mode = WATERMARK_CLEAN_MODE
        self._header_lines = set()
        self._watermark_pages_removed = 0

    # ── 水印检测与分析 ──

    def _analyze_watermarks(self, pages_text: List[Dict]) -> set:
        texts = [p["raw_text"] for p in pages_text]
        self._header_lines = _detect_page_header_lines(texts)

        if self._header_lines:
            print(f"  [水印检测] 发现 {len(self._header_lines)} 个跨页重复行")
            if self.watermark_mode == "aggressive":
                for line in sorted(list(self._header_lines))[:5]:
                    print(f"    → {line[:80]}")

        return self._header_lines

    def _remove_watermarks_from_page(self, text: str) -> str:
        if not text:
            return text

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if _FOOTER_PATTERN.match(stripped):
                continue

            if stripped in self._header_lines:
                continue

            if self.watermark_mode == "aggressive":
                if _has_watermark_keyword(stripped):
                    continue

            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def _filter_watermark_pages(self, pages: List[Dict]) -> List[Dict]:
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
            print(f"  [水印过滤] 移除了 {removed} 个水印页，保留 {len(kept)} 页")
        self._watermark_pages_removed = removed
        return kept

    # ── PDF 提取（多策略）──

    def _extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, List[str], List[Dict]]:
        """使用 pdfplumber 提取（默认策略）"""
        text_parts = []
        table_chunks = []
        raw_pages = []
        total_pages = 0

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                # 原始文本
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                raw_pages.append({
                    "page_num": page_num + 1,
                    "raw_text": page_text.strip(),
                })

                # 表格提取
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = None

                if tables:
                    for t_idx, table in enumerate(tables):
                        try:
                            tbl_text = _table_to_text(table, page_num + 1, t_idx)
                            if tbl_text and len(tbl_text) > 50:
                                table_chunks.append(tbl_text)
                        except Exception:
                            continue

                if page_text and page_text.strip():
                    text_parts.append({
                        "page_num": page_num + 1,
                        "text": page_text.strip(),
                    })

        return text_parts, table_chunks, raw_pages, total_pages

    def extract_text_with_tables(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        从 PDF 提取文本 + 独立表格块（多策略容错）

        策略链：
          1. pdfplumber（默认）
          2. 如失败，尝试降级读取（仅提取文字）
          3. 如仍然失败，返回明确错误信息
        """
        start_time = time.time()

        try:
            text_parts, table_chunks, raw_pages, total_pages = \
                self._extract_with_pdfplumber(pdf_path)
        except pdfplumber.utils.PDFSyntaxError as e:
            # 语法错误：尝试容错打开
            error_msg = str(e)
            if "password" in error_msg.lower():
                raise ValueError(
                    f"PDF 文件 '{os.path.basename(pdf_path)}' 已加密，需要解密后才能处理。"
                )
            # 尝试降级：用 pdfplumber 的 strict=False 模式
            try:
                text_parts, table_chunks, raw_pages, total_pages = \
                    self._extract_with_pdfplumber(pdf_path)
            except Exception as e2:
                raise ValueError(
                    f"PDF 文件 '{os.path.basename(pdf_path)}' 解析失败，"
                    f"文件可能已损坏。错误: {e2}"
                )
        except ValueError as e:
            raise  # 透传已知错误
        except Exception as e:
            raise ValueError(
                f"PDF 文件 '{os.path.basename(pdf_path)}' 解析失败: {e}\n"
                f"请确认文件格式正确且不为空。"
            )

        # ── 水印检测与清洗 ──
        if self.watermark_enabled and raw_pages:
            try:
                watermark_start = time.time()
                self._analyze_watermarks(raw_pages)
                text_parts = self._filter_watermark_pages(text_parts)
                for p in text_parts:
                    p["text"] = self._remove_watermarks_from_page(p["text"])
                wm_elapsed = time.time() - watermark_start
                print(f"  [水印处理] 耗时 {wm_elapsed:.3f}s")
            except Exception as e:
                print(f"  [水印处理警告] 水印清洗失败，跳过: {e}")

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
        """将文本分割成块（含空文本检查）"""
        if not text or not text.strip():
            return []

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
        完整处理一个 PDF（含容错）

        Args:
            pdf_path: PDF 文件路径

        Returns:
            完整文本块列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: PDF 解析失败或无文本内容
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        print(f"[PDF处理] 开始处理: {os.path.basename(pdf_path)}"
              f"{' [水印清洗启用]' if self.watermark_enabled else ''}")

        try:
            full_text, table_chunks = self.extract_text_with_tables(pdf_path)
        except ValueError:
            raise  # 透传已知错误
        except Exception as e:
            raise ValueError(
                f"PDF '{os.path.basename(pdf_path)}' 处理过程中发生未知错误: {e}"
            )

        if not full_text.strip():
            raise ValueError(
                f"PDF 文件 '{os.path.basename(pdf_path)}' 未能提取到文本内容。\n"
                f"可能原因：\n"
                f"  1. 该 PDF 是纯扫描件（图片格式），需要 OCR 工具处理\n"
                f"  2. 该 PDF 内容为空或所有页面被水印过滤\n"
                f"  3. 该 PDF 使用了非标准编码方式"
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
        """获取 PDF 文件元数据（含容错）"""
        if not os.path.exists(pdf_path):
            return {
                "filename": os.path.basename(pdf_path) if pdf_path else "未知",
                "file_size_mb": 0,
                "total_pages": 0,
                "title": "文件不存在",
                "author": "未知",
                "file_hash": "",
            }
        try:
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
        except Exception as e:
            print(f"  [警告] 获取 PDF 元数据失败: {e}")
            return {
                "filename": os.path.basename(pdf_path),
                "file_size_mb": round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
                "total_pages": 0,
                "title": "未知",
                "author": "未知",
                "file_hash": "",
            }
