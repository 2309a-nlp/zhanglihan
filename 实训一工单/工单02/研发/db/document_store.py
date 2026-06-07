# -*- coding: utf-8 -*-
"""
MySQL 文档存储（连接池版）
负责文档元数据与文本块的持久化存储管理
支持连接池复用，避免高并发下频繁创建/销毁连接
"""

import json
import threading
from queue import Queue, Empty, Full
from typing import List, Dict, Any, Optional

import pymysql
import pymysql.cursors

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    MYSQL_POOL_MIN, MYSQL_POOL_MAX,
)


class _ConnectionPool:
    """简单的 MySQL 连接池（线程安全）"""

    def __init__(self, min_conn: int = 2, max_conn: int = 5):
        self._host = MYSQL_HOST
        self._port = MYSQL_PORT
        self._user = MYSQL_USER
        self._password = MYSQL_PASSWORD
        self._database = MYSQL_DATABASE
        self._max = max_conn
        self._pool = Queue(maxsize=max_conn)
        self._created = 0
        self._lock = threading.Lock()
        # 预创建最小连接
        for _ in range(min_conn):
            self._pool.put(self._new_conn())
            with self._lock:
                self._created += 1

    def _new_conn(self):
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def acquire(self):
        try:
            conn = self._pool.get_nowait()
        except Empty:
            with self._lock:
                if self._created < self._max:
                    conn = self._new_conn()
                    self._created += 1
                else:
                    # 达到上限，阻塞等待
                    conn = self._pool.get()
        # 断线重连
        try:
            conn.ping(reconnect=True)
        except Exception:
            conn = self._new_conn()
        return conn

    def release(self, conn):
        if conn is None:
            return
        try:
            self._pool.put_nowait(conn)
        except Full:
            conn.close()
            with self._lock:
                self._created -= 1

    def close_all(self):
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
            except Exception:
                pass
        self._created = 0


class DocumentStore:
    """MySQL 文档存储管理器（连接池版）"""

    def __init__(self):
        self._pool = _ConnectionPool(
            min_conn=MYSQL_POOL_MIN,
            max_conn=MYSQL_POOL_MAX,
        )

    def _execute(self, sql: str, params: tuple = ()) -> Any:
        """执行 SQL（从连接池获取连接）"""
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur
        finally:
            self._pool.release(conn)

    def _execute_many(self, sql: str, rows: List[tuple]):
        """批量执行（需要事务）"""
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.release(conn)

    def init_tables(self):
        """初始化表结构（幂等）"""
        sqls = [
            """CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                file_size_kb INT DEFAULT 0,
                total_pages INT DEFAULT 0,
                total_chunks INT DEFAULT 0,
                author VARCHAR(255) DEFAULT '',
                title VARCHAR(255) DEFAULT '',
                source_path VARCHAR(512) DEFAULT '',
                status VARCHAR(32) DEFAULT 'ready',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_file_hash (file_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS chunks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doc_id INT NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
                    ON DELETE CASCADE,
                INDEX idx_doc_id (doc_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
        ]
        for sql in sqls:
            self._execute(sql)
        print("[DB] 表结构已就绪")

    # ---- 文档 CRUD ----

    def add_document(self, filename: str, file_hash: str,
                     file_size_kb: int = 0, total_pages: int = 0,
                     author: str = "", title: str = "",
                     source_path: str = "",
                     chunks: Optional[List[Dict]] = None
                     ) -> int:
        """添加文档及其文本块（事务）Returns: doc_id"""
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                total = len(chunks) if chunks else 0
                cur.execute(
                    """INSERT INTO documents
                       (filename, file_hash, file_size_kb, total_pages,
                        total_chunks, author, title, source_path)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (filename, file_hash, file_size_kb, total_pages,
                     total, author, title, source_path),
                )
                doc_id = cur.lastrowid

                if chunks:
                    chunk_sql = """INSERT INTO chunks
                                   (doc_id, chunk_index, content, metadata)
                                   VALUES (%s, %s, %s, %s)"""
                    rows = [
                        (doc_id, c.get("chunk_index", c["chunk_id"]),
                         c["content"],
                         json.dumps(c.get("metadata", {}), ensure_ascii=False))
                        for c in chunks
                    ]
                    cur.executemany(chunk_sql, rows)

                conn.commit()
                return doc_id
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.release(conn)

    def get_document(self, doc_id: int) -> Optional[Dict]:
        cur = self._execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        return cur.fetchone()

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        cur = self._execute(
            "SELECT * FROM documents WHERE file_hash = %s", (file_hash,)
        )
        return cur.fetchone()

    def get_all_documents(self) -> List[Dict]:
        cur = self._execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
        return cur.fetchall()

    def delete_document(self, doc_id: int) -> bool:
        cur = self._execute(
            "DELETE FROM documents WHERE id = %s", (doc_id,)
        )
        return cur.rowcount > 0

    # ---- 文本块操作 ----

    def get_chunks(self, doc_id: Optional[int] = None) -> List[Dict]:
        if doc_id is not None:
            cur = self._execute(
                "SELECT * FROM chunks WHERE doc_id = %s ORDER BY chunk_index",
                (doc_id,)
            )
        else:
            cur = self._execute(
                "SELECT * FROM chunks ORDER BY doc_id, chunk_index"
            )
        rows = cur.fetchall()
        for r in rows:
            if isinstance(r.get("metadata"), str):
                r["metadata"] = json.loads(r["metadata"])
        return rows

    def get_total_chunk_count(self) -> int:
        cur = self._execute("SELECT COUNT(*) as cnt FROM chunks")
        return cur.fetchone()["cnt"]

    def close(self):
        self._pool.close_all()
