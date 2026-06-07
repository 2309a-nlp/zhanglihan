# -*- coding: utf-8 -*-
"""
问答引擎（高稳定性版）
整合 PDF 处理、MySQL 存储、向量检索、LLM 对话问答
新增：
  1. 后台健康检查线程（综合检测各组件状态）
  2. 自动恢复逻辑（索引修复、数据库重连）
  3. 优雅降级（LLM 不可用时可返回检索结果）
  4. 全面的异常处理（不因单组件故障导致系统崩溃）
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
    """问答引擎 - 系统核心（高稳定性版）"""

    def __init__(self):
        from config import WATERMARK_REMOVAL_ENABLED
        self.watermark_enabled = WATERMARK_REMOVAL_ENABLED
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
        self._engine_lock = threading.Lock()  # 引擎级锁

        # ── 内存缓存 ──
        self.loaded_documents = []
        self._chunk_cache = {}   # {chunk_id: content}
        self._conversation_history = []

        # ── 健康检查 ──
        self._health_thread = None
        self._health_running = False
        self._health_interval = 60  # 秒
        self._health_status = {
            "database": False,
            "vector_index": False,
            "llm_available": True,
            "overall": False,
            "last_check": 0,
            "errors": [],
        }

        # ── 降级状态 ──
        self._downgraded = False
        self._downgrade_reason = ""

    # ==================== 初始化 + 预热 ====================

    def initialize(self):
        """初始化：建表 + 加载索引 + 预热 + 启动健康检查"""
        print("=" * 60)
        print("  问答引擎初始化（高稳定性版）")
        print("=" * 60)

        try:
            self.doc_store.init_tables()
            self._health_status["database"] = True
        except Exception as e:
            print(f"  [错误] 数据库初始化失败: {e}")
            self._health_status["database"] = False
            self._health_status["errors"].append(f"数据库: {e}")

        if self.vector_store.load_index(VECTOR_STORE_PATH):
            self.is_ready = True
            try:
                docs = self.doc_store.get_all_documents()
                self.loaded_documents = docs
            except Exception:
                self.loaded_documents = []
            info = self.vector_store.get_index_info()
            print(f"  [引擎] 就绪: {info['total_vectors']} 个向量, "
                  f"{len(self.loaded_documents)} 个文档")
            self._health_status["vector_index"] = True
            self._load_chunk_cache()
        else:
            print("  [引擎] 未找到索引，请先添加 PDF 文档")
            self.is_ready = False

        # ── 重置熔断器（确保全新启动后不残留旧状态）──
        self.llm_qa.reset_circuit_breaker()

        # ── LLM 预热 ──
        if self.is_ready:
            self._warmup()

        # ── 启动健康检查线程 ──
        self._start_health_check()

        print("=" * 60)
        return self.is_ready

    def _warmup(self):
        """预热 LLM 连接池"""
        if self._warmed_up:
            return
        start = time.time()
        try:
            self.llm_qa.warmup()
        except Exception:
            pass
        self._warmed_up = True
        elapsed = time.time() - start
        if elapsed > 0.1:
            print(f"  [引擎预热] 完成, 耗时 {elapsed:.2f}s")

    # ==================== 健康检查 ====================

    def _start_health_check(self):
        """启动后台健康检查线程"""
        if self._health_thread and self._health_thread.is_alive():
            return

        self._health_running = True

        def _check_loop():
            while self._health_running:
                try:
                    self._run_health_check()
                    self._auto_recovery()
                except Exception as e:
                    print(f"  [健康检查] 异常: {e}")
                time.sleep(self._health_interval)

        self._health_thread = threading.Thread(target=_check_loop,
                                                daemon=True,
                                                name="health-check")
        self._health_thread.start()
        print(f"  [健康检查] 已启动（每 {self._health_interval}s）")

    def _stop_health_check(self):
        """停止健康检查线程"""
        self._health_running = False
        if self._health_thread:
            self._health_thread.join(timeout=5)

    def _run_health_check(self):
        """执行一次综合健康检查"""
        errors = []

        # 1. 数据库检查
        try:
            self.doc_store.is_healthy()
            self._health_status["database"] = True
        except Exception as e:
            self._health_status["database"] = False
            errors.append(f"数据库不可用: {e}")
            # 尝试重连
            try:
                self.doc_store.reconnect()
                self._health_status["database"] = True
                errors.pop()
                print("  [健康检查] 数据库已重连 ✅")
            except Exception as e2:
                errors.append(f"数据库重连失败: {e2}")

        # 2. 向量索引检查
        try:
            idx_health = self.vector_store.check_health()
            self._health_status["vector_index"] = idx_health["healthy"]
            if not idx_health["healthy"]:
                for err in idx_health.get("errors", []):
                    errors.append(f"向量索引: {err}")
                for warn in idx_health.get("warnings", []):
                    print(f"  [健康检查] 警告: {warn}")
        except Exception as e:
            self._health_status["vector_index"] = False
            errors.append(f"向量索引检查失败: {e}")

        # 3. LLM 熔断器状态 + 自动恢复
        cb_state = self.llm_qa.get_circuit_breaker_state()
        self._health_status["llm_available"] = (cb_state != "OPEN")
        if cb_state == "OPEN":
            print(f"  [健康检查] ⚠ LLM 熔断保护中，尝试 API 自检恢复...")
            try:
                self.llm_qa.warmup()
                self.llm_qa.reset_circuit_breaker()
                self._health_status["llm_available"] = True
                print(f"  [健康检查] ✅ LLM 熔断器已自动重置（API 自检通过）")
            except Exception as api_e:
                print(f"  [健康检查] LLM 自检仍失败: {api_e}")

        # 4. 综合状态
        essential_ok = (self._health_status["database"] or
                        self._health_status["vector_index"])
        self._health_status["overall"] = essential_ok
        self._health_status["last_check"] = time.time()
        self._health_status["errors"] = errors

    def _auto_recovery(self):
        """自动恢复逻辑"""
        # 向量索引修复
        if not self._health_status["vector_index"]:
            try:
                chunks = dict(self._chunk_cache)
                if chunks:
                    self.vector_store.auto_repair(chunks)
            except Exception as e:
                print(f"  [自动恢复] 向量索引修复失败: {e}")

        # 降级撤销（如果组件恢复）
        if self._downgraded:
            db_ok = self._health_status.get("database", False)
            idx_ok = self._health_status.get("vector_index", False)
            if db_ok and idx_ok:
                self._downgraded = False
                self._downgrade_reason = ""
                self.is_ready = True
                print("  [自动恢复] ✅ 组件恢复，降级状态已撤销")

    def get_detailed_health(self) -> Dict[str, Any]:
        """获取详细的健康状态报告"""
        return dict(self._health_status)

    # ==================== 文档管理 ====================

    def add_pdf(self, pdf_path: str) -> bool:
        """添加 PDF 文档（含异常处理）"""
        if not os.path.exists(pdf_path):
            print(f"[错误] 文件不存在: {pdf_path}")
            return False

        filename = os.path.basename(pdf_path)

        import hashlib
        try:
            with open(pdf_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"[错误] 读取文件失败: {e}")
            return False

        try:
            existing = self.doc_store.get_document_by_hash(file_hash)
            if existing:
                print(f"[跳过] 文档已存在: {filename} (doc_id={existing['id']})")
                return True
        except Exception as e:
            print(f"[警告] 查重失败（数据库可能不可用）: {e}")
            existing = None

        # PDF 解析（含错误处理）
        try:
            metadata = self.pdf_processor.get_pdf_metadata(pdf_path)
            chunks = self.pdf_processor.process_pdf(pdf_path)
        except Exception as e:
            print(f"[错误] PDF 解析失败: {e}")
            return False

        if not chunks:
            print(f"[错误] PDF 解析结果为空: {filename}")
            return False

        # MySQL 存储（含错误处理）
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

        # 重建索引
        try:
            self._rebuild_faiss()
        except Exception as e:
            print(f"[错误] 索引重建失败: {e}")
            return False

        try:
            doc = self.doc_store.get_document(doc_id)
            self.loaded_documents.append(doc)
        except Exception:
            pass

        return True

    def _load_chunk_cache(self):
        """从 MySQL 加载文本块到内存缓存"""
        try:
            chunks = self.doc_store.get_chunks()
            with self._cache_lock:
                self._chunk_cache = {c["id"]: c["content"] for c in chunks}
            if chunks:
                self.vector_store.rebuild_bm25(self._chunk_cache)
            print(f"  [缓存] 已加载 {len(self._chunk_cache)} 个文本块到内存")
        except Exception as e:
            print(f"  [缓存] 加载失败: {e}")

    def _rebuild_faiss(self):
        """从 MySQL 读取所有 chunks 重建 FAISS + BM25 双索引"""
        try:
            chunks = self.doc_store.get_chunks()
        except Exception as e:
            print(f"  [错误] 数据库读取失败: {e}")
            return

        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        texts = [c["content"] for c in chunks]

        try:
            self.vector_store.build_index(ids, texts)
            self.vector_store.save_index(VECTOR_STORE_PATH)
            with self._cache_lock:
                self._chunk_cache = {c["id"]: c["content"] for c in chunks}
            self.is_ready = True
        except Exception as e:
            print(f"  [错误] 索引构建失败: {e}")

    def remove_doc(self, doc_id: int) -> bool:
        """删除文档并重建索引（含异常处理）"""
        try:
            doc = self.doc_store.get_document(doc_id)
            if not doc:
                return False
            with self._engine_lock:
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
                print(f"[删除] 已移除: {doc['filename']}")
            return True
        except Exception as e:
            print(f"[错误] 删除文档失败: {e}")
            return False

    def list_docs(self) -> List[Dict]:
        try:
            return self.doc_store.get_all_documents()
        except Exception:
            return self.loaded_documents

    # ==================== 对话问答（含降级）====================

    def _validate_query(self, query: str) -> Optional[str]:
        """校验用户输入"""
        return self.llm_qa.validate_input(query)

    def chat_stream(self, query: str,
                    history: Optional[List[Dict]] = None):
        """
        流式对话式 RAG 问答（含优雅降级）

        降级策略：
        - LLM 不可用（熔断）→ 返回检索的原始段落
        - 检索失败 → 返回错误提示
        - 数据库不可用 → 使用内存缓存
        """
        self._stream_result = None
        overall_start = time.time()

        # 输入校验
        validation_error = self._validate_query(query)
        if validation_error:
            self._stream_result = {
                "answer": validation_error,
                "elapsed_total": 0,
                "retrieval_time": 0,
                "llm_time": 0,
                "source_chunks": [],
                "downgraded": False,
            }
            yield validation_error
            return

        # 系统未就绪
        if not self.is_ready:
            msg = "系统尚未加载文档，请先上传 PDF 文件。"
            self._stream_result = {
                "answer": msg,
                "elapsed_total": 0,
                "retrieval_time": 0,
                "llm_time": 0,
                "source_chunks": [],
                "downgraded": False,
            }
            yield msg
            return

        # 1. 语义检索
        r_start = time.time()
        retrieved_chunks = []
        source_chunks = []

        try:
            if not self._health_status.get("vector_index", True):
                # 索引不可用，尝试自动修复
                chunks_data = dict(self._chunk_cache)
                if chunks_data:
                    self.vector_store.auto_repair(chunks_data)

            search_result = self.vector_store.search(query, top_k=TOP_K_RESULTS)

            for r in search_result:
                with self._cache_lock:
                    content = self._chunk_cache.get(r["chunk_id"], "")
                retrieved_chunks.append({
                    "content": content,
                    "score": r["score"],
                })
                source_chunks.append({
                    "content": content[:200] + ("..." if len(content) > 200 else ""),
                    "score": r["score"],
                })
        except ValueError as e:
            # 索引为空
            msg = f"⚠️ {e}"
            self._stream_result = {
                "answer": msg,
                "elapsed_total": round(time.time() - overall_start, 3),
                "retrieval_time": 0,
                "llm_time": 0,
                "source_chunks": [],
                "downgraded": False,
            }
            yield msg
            return
        except Exception as e:
            # 检索失败
            msg = f"⚠️ 文档检索失败: {e}。请稍后重试。"
            self._stream_result = {
                "answer": msg,
                "elapsed_total": round(time.time() - overall_start, 3),
                "retrieval_time": round(time.time() - r_start, 3),
                "llm_time": 0,
                "source_chunks": [],
                "downgraded": True,
            }
            yield msg
            return

        retrieval_time = time.time() - r_start

        # 2. 检查 LLM 可用性 — 熔断降级
        cb_state = self.llm_qa.get_circuit_breaker_state()
        if cb_state == "OPEN":
            # 降级：返回检索结果
            return self._downgraded_response(
                query, retrieved_chunks, source_chunks,
                retrieval_time, overall_start
            )

        # 3. 流式 LLM 回答
        try:
            for token in self.llm_qa.chat_answer_stream(
                    query, retrieved_chunks, history):
                yield token
        except Exception as e:
            # LLM 调用异常，降级
            print(f"  [降级] LLM 调用异常: {e}")
            yield from self._downgraded_response(
                query, retrieved_chunks, source_chunks,
                retrieval_time, overall_start
            )
            return

        # 4. 合并结果
        llm_res = getattr(self.llm_qa, '_stream_result', None)
        if not llm_res:
            yield from self._downgraded_response(
                query, retrieved_chunks, source_chunks,
                retrieval_time, overall_start
            )
            return

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
            "downgraded": False,
        }

    def _downgraded_response(self, query: str,
                              retrieved_chunks: List[Dict],
                              source_chunks: List[Dict],
                              retrieval_time: float,
                              overall_start: float) -> Generator[str, None, None]:
        """
        降级回答：LLM 不可用时直接返回检索的原始文本
        """
        if not retrieved_chunks:
            msg = ("⚠️ 未找到相关文档内容，请尝试其他关键词。")
            self._stream_result = {
                "answer": msg,
                "elapsed_total": round(time.time() - overall_start, 3),
                "retrieval_time": round(retrieval_time, 3),
                "llm_time": 0,
                "source_chunks": source_chunks,
                "downgraded": True,
                "downgrade_reason": "LLM 不可用",
            }
            yield msg
            return

        # 构建结构化列表格式的降级输出
        msg = f"（降级模式）基于以下文档片段回答您的问题：\n\n"
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            content = chunk["content"].strip()[:200]
            msg += f"· {content}\n"

        msg += ("\n---\n"
                "⚠️ LLM 当前不可用，以上为文档原始内容。"
                "系统将自动恢复，请稍后重试。")

        self._stream_result = {
            "answer": msg,
            "elapsed_total": round(time.time() - overall_start, 3),
            "retrieval_time": round(retrieval_time, 3),
            "llm_time": 0,
            "source_chunks": source_chunks,
            "downgraded": True,
            "downgrade_reason": "LLM 不可用",
        }
        yield msg

    # ==================== 系统状态 ====================

    def get_system_status(self) -> Dict[str, Any]:
        docs = self.list_docs()
        wm_removed = 0
        if hasattr(self, 'pdf_processor') and hasattr(self.pdf_processor, '_watermark_pages_removed'):
            wm_removed = self.pdf_processor._watermark_pages_removed
        return {
            "ready": self.is_ready,
            "warmed_up": self._warmed_up,
            "total_documents": len(docs),
            "cache_size": self.llm_qa._semantic_cache.size(),
            "watermark_enabled": getattr(self, 'watermark_enabled', True),
            "watermark_pages_removed": wm_removed,
            "downgraded": self._downgraded,
            "downgrade_reason": self._downgrade_reason,
            "health": dict(self._health_status),
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
            "circuit_breaker_state": self.llm_qa.get_circuit_breaker_state(),
            "embedding_model": self.vector_store.model_name,
        }

    def close(self):
        self._stop_health_check()
        self.llm_qa.close()
        self.doc_store.close()
