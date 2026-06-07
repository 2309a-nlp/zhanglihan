# -*- coding: utf-8 -*-
"""
问答引擎（线程安全 + 预热版）
整合 PDF 处理、MySQL 存储、向量检索、LLM 对话问答
核心优化：
  1. 系统启动时预热 LLM 连接池和向量检索
  2. 线程安全的全文搜索和缓存访问
  3. 提前缓存 chunk 文本到内存，避免反复读取 MySQL
  4. 语义缓存（相似问题秒回）
"""

import os
import time
import threading
from typing import List, Dict, Any, Optional, Generator

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_qa import LLMQA
from db.document_store import DocumentStore
from config import (
    DOCS_DIR, VECTOR_STORE_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
)


class QAEngine:
    """问答引擎 - 系统核心（线程安全 + 预热版）"""

    def __init__(self):
        self.pdf_processor = PDFProcessor(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        )
        self.vector_store = VectorStore()
        self.llm_qa = LLMQA()
        self.doc_store = DocumentStore()
        self.is_ready = False
        self._warmed_up = False

        # ── 线程安全 ──
        self._cache_lock = threading.Lock()
        self._history_lock = threading.Lock()

        # ── 内存缓存 ──
        self.loaded_documents = []
        self._chunk_cache = {}   # {chunk_id: content}
        self._conversation_history = []

    # ==================== 初始化 + 预热 ====================

    def initialize(self):
        """初始化：建表 + 加载索引 + 预热 LLM 连接池"""
        print("=" * 60)
        print("  问答引擎初始化")
        print("=" * 60)

        self.doc_store.init_tables()

        if self.vector_store.load_index(VECTOR_STORE_PATH):
            self.is_ready = True
            docs = self.doc_store.get_all_documents()
            self.loaded_documents = docs
            info = self.vector_store.get_index_info()
            total_chunks = self.doc_store.get_total_chunk_count()
            print(f"  [引擎] 就绪: {info['total_vectors']} 个向量, "
                  f"{len(docs)} 个文档, {total_chunks} 个文本块")
            self._load_chunk_cache()
        else:
            print("  [引擎] 未找到索引，请先添加 PDF 文档")
            self.is_ready = False

        # ── LLM 预连接预热（首次提问省 0.5-1.5s）──
        if self.is_ready:
            self._warmup()

        print("=" * 60)
        return self.is_ready

    def _warmup(self):
        """预热 LLM 连接池（启动时执行一次）"""
        if self._warmed_up:
            return
        start = time.time()
        self.llm_qa.warmup()
        self._warmed_up = True
        elapsed = time.time() - start
        if elapsed > 0.1:
            print(f"  [引擎预热] 完成, 耗时 {elapsed:.2f}s")

    # ==================== 文档管理 ====================

    def add_pdf(self, pdf_path: str) -> bool:
        """添加 PDF 文档：解析 → 存 MySQL → 重建 FAISS 索引"""
        if not os.path.exists(pdf_path):
            print(f"[错误] 文件不存在: {pdf_path}")
            return False

        filename = os.path.basename(pdf_path)

        import hashlib
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        existing = self.doc_store.get_document_by_hash(file_hash)
        if existing:
            print(f"[跳过] 文档已存在: {filename} (doc_id={existing['id']})")
            return True

        try:
            metadata = self.pdf_processor.get_pdf_metadata(pdf_path)
            chunks = self.pdf_processor.process_pdf(pdf_path)
        except Exception as e:
            print(f"[错误] PDF 解析失败: {e}")
            return False

        try:
            doc_id = self.doc_store.add_document(
                filename=filename,
                file_hash=file_hash,
                file_size_kb=int(metadata.get("file_size_mb", 0) * 1024),
                total_pages=metadata.get("total_pages", 0),
                author=metadata.get("author", ""),
                title=metadata.get("title", ""),
                source_path=pdf_path,
                chunks=chunks,
            )
        except Exception as e:
            print(f"[错误] MySQL 存储失败: {e}")
            return False

        self._rebuild_faiss()
        doc = self.doc_store.get_document(doc_id)
        self.loaded_documents.append(doc)
        return True

    def _load_chunk_cache(self):
        """从 MySQL 加载文本块到内存缓存，同时重建 BM25 索引"""
        chunks = self.doc_store.get_chunks()
        with self._cache_lock:
            self._chunk_cache = {c["id"]: c["content"] for c in chunks}
        # 重建 BM25 关键词索引
        if chunks:
            self.vector_store.rebuild_bm25(self._chunk_cache)
        print(f"  [缓存] 已加载 {len(self._chunk_cache)} 个文本块到内存")

    def _rebuild_faiss(self):
        """从 MySQL 读取所有 chunks 重建 FAISS + BM25 双索引"""
        chunks = self.doc_store.get_chunks()
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        texts = [c["content"] for c in chunks]
        self.vector_store.build_index(ids, texts)
        self.vector_store.save_index(VECTOR_STORE_PATH)
        with self._cache_lock:
            self._chunk_cache = {c["id"]: c["content"] for c in chunks}
        self.is_ready = True

    def remove_doc(self, doc_id: int) -> bool:
        """删除文档并重建索引"""
        doc = self.doc_store.get_document(doc_id)
        if not doc:
            return False
        self.doc_store.delete_document(doc_id)
        self.loaded_documents = [
            d for d in self.loaded_documents if d["id"] != doc_id
        ]
        remaining = self.doc_store.get_total_chunk_count()
        if remaining > 0:
            self._rebuild_faiss()
        else:
            self.vector_store.index = None
            self.vector_store.total_vectors = 0
            with self._cache_lock:
                self._chunk_cache.clear()
            self.is_ready = False
            faiss_path = f"{VECTOR_STORE_PATH}.faiss"
            if os.path.exists(faiss_path):
                os.remove(faiss_path)
        print(f"[删除] 已移除: {doc['filename']}")
        return True

    def list_docs(self) -> List[Dict]:
        return self.doc_store.get_all_documents()

    # ==================== 对话问答（RAG only）====================

    def chat_stream(self, query: str,
                    history: Optional[List[Dict]] = None):
        """
        流式对话式 RAG 问答（逐 token 产出）

        Args:
            query: 用户输入的问题
            history: 对话历史

        Yields:
            文本 token（逐 token 产出，在 Streamlit 中用 st.write_stream() 消费）

        迭代结束后读取 self._stream_result:
            { "answer": str, "elapsed_total": float, "retrieval_time": float,
              "llm_time": float, "source_chunks": [...] }
        """
        self._stream_result = None
        overall_start = time.time()

        if not self.is_ready:
            self._stream_result = {
                "answer": "系统尚未加载文档，请先上传 PDF 文件。",
                "elapsed_total": 0,
                "retrieval_time": 0,
                "llm_time": 0,
                "source_chunks": [],
            }
            yield ""
            return

        # 1. 语义检索（~120ms）
        r_start = time.time()
        search_result = self.vector_store.search(query, top_k=TOP_K_RESULTS)
        retrieval_time = time.time() - r_start

        # 2. 从内存缓存获取文本内容（无需数据库 IO）
        retrieved_chunks = []
        for r in search_result:
            with self._cache_lock:
                content = self._chunk_cache.get(r["chunk_id"], "")
            retrieved_chunks.append({
                "content": content,
                "score": r["score"],
            })

        # 3. 构建 source_chunks 元数据
        source_chunks = [
            {
                "content": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
                "score": c["score"],
            }
            for c in retrieved_chunks
        ]

        # 4. 流式 RAG 回答
        for token in self.llm_qa.chat_answer_stream(query, retrieved_chunks, history):
            yield token

        # 5. 合并结果
        llm_res = self.llm_qa._stream_result
        overall_time = time.time() - overall_start
        print(f"[流式] {overall_time:.3f}s (检索 {retrieval_time:.3f}s, "
              f"LLM {llm_res.get('elapsed_time', 0):.3f}s, "
              f"源 {llm_res.get('source', '?')})")

        self._stream_result = {
            "answer": llm_res["answer"],
            "elapsed_total": round(overall_time, 3),
            "retrieval_time": round(retrieval_time, 3),
            "llm_time": llm_res["elapsed_time"],
            "llm_source": llm_res.get("source", "api"),
            "source_chunks": source_chunks,
        }

    # ==================== 系统状态 ====================

    def get_system_status(self) -> Dict[str, Any]:
        docs = self.list_docs()
        return {
            "ready": self.is_ready,
            "warmed_up": self._warmed_up,
            "total_documents": len(docs),
            "cache_size": self.llm_qa._semantic_cache.size(),
            "documents": [
                {
                    "id": d["id"],
                    "filename": d["filename"],
                    "pages": d["total_pages"],
                    "chunks": d["total_chunks"],
                    "created": str(d["created_at"]),
                }
                for d in docs
            ],
            "total_vectors": self.vector_store.total_vectors,
            "llm_model": self.llm_qa.model,
            "embedding_model": self.vector_store.model_name,
        }

    def close(self):
        self.llm_qa.close()
        self.doc_store.close()
