# -*- coding: utf-8 -*-
"""
MySQL 文档存储
负责文档元数据与文本块的持久化存储管理
"""

import json
import pymysql
import pymysql.cursors
from typing import List, Dict, Any, Optional

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
)


class DocumentStore:
    """MySQL 文档存储管理器"""

    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        """建立数据库连接"""
        self.conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def _ensure_conn(self):
        """确保连接未断开，断线重连"""
        try:
            self.conn.ping(reconnect=True)
        except Exception:
            self._connect()

    def _execute(self, sql: str, params: tuple = ()) -> Any:
        """执行 SQL（自动重连）"""
        self._ensure_conn()
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur

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
        """
        添加文档及其文本块（事务）
        Returns: doc_id
        """
        sql = """INSERT INTO documents
                 (filename, file_hash, file_size_kb, total_pages,
                  total_chunks, author, title, source_path)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        total_chunks = len(chunks) if chunks else 0

        self._ensure_conn()
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (
                    filename, file_hash, file_size_kb, total_pages,
                    total_chunks, author, title, source_path,
                ))
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

                self.conn.commit()
                return doc_id
        except Exception:
            self.conn.rollback()
            raise

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """获取单个文档"""
        cur = self._execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        return cur.fetchone()

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """通过文件 hash 获取文档"""
        cur = self._execute(
            "SELECT * FROM documents WHERE file_hash = %s", (file_hash,)
        )
        return cur.fetchone()

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档（按创建时间降序）"""
        cur = self._execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
        return cur.fetchall()

    def delete_document(self, doc_id: int) -> bool:
        """删除文档及其所有文本块（级联）"""
        cur = self._execute(
            "DELETE FROM documents WHERE id = %s", (doc_id,)
        )
        return cur.rowcount > 0

    # ---- 文本块操作 ----

    def get_chunks(self, doc_id: Optional[int] = None) -> List[Dict]:
        """
        获取文本块（全部或指定文档）
        Returns: list of {id, doc_id, chunk_index, content, metadata}
        """
        if doc_id is not None:
            cur = self._execute(
                "SELECT * FROM chunks WHERE doc_id = %s "
                "ORDER BY chunk_index", (doc_id,)
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
        """获取文本块总数"""
        cur = self._execute("SELECT COUNT(*) as cnt FROM chunks")
        return cur.fetchone()["cnt"]

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
