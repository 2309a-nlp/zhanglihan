# -*- coding: utf-8 -*-
"""
MySQL 文档存储（高稳定性版）
负责文档元数据与文本块的持久化存储管理
新增特性：
  - 数据库自动重连（断线后自动恢复）
  - 连接健康检查（防止废弃连接）
  - 重试机制（瞬态错误自动恢复）
"""

import json
import time
import threading
from queue import Queue, Empty, Full
from typing import List, Dict, Any, Optional

import pymysql
import pymysql.cursors

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    MYSQL_POOL_MIN, MYSQL_POOL_MAX,
    DB_RECONNECT_MAX_ATTEMPTS, DB_RECONNECT_INTERVAL,
)


class _ConnectionPool:
    """MySQL 连接池（高稳定性版 — 自动重连 + 健康检查）"""

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
        self._healthy = True
        self._last_health_check = 0
        self._health_check_interval = 15  # 每 15 秒检查一次连接健康
        for _ in range(min_conn):
            self._pool.put(self._new_conn())
            with self._lock:
                self._created += 1

    def _new_conn(self):
        """创建新连接，重试支持"""
        last_error = None
        for attempt in range(DB_RECONNECT_MAX_ATTEMPTS):
            try:
                conn = pymysql.connect(
                    host=self._host,
                    port=self._port,
                    user=self._user,
                    password=self._password,
                    database=self._database,
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=True,
                    connect_timeout=5,
                    read_timeout=30,
                )
                return conn
            except pymysql.Error as e:
                last_error = e
                if attempt < DB_RECONNECT_MAX_ATTEMPTS - 1:
                    time.sleep(DB_RECONNECT_INTERVAL * (attempt + 1))
        raise ConnectionError(
            f"数据库连接失败（已重试 {DB_RECONNECT_MAX_ATTEMPTS} 次）: {last_error}"
        )

    def _check_conn_health(self, conn) -> bool:
        """检查连接是否健康"""
        try:
            conn.ping(reconnect=True)
            return True
        except Exception:
            return False

    def acquire(self):
        """获取连接（含健康检查和自动修复）"""
        now = time.time()

        # 定时健康检查
        if now - self._last_health_check > self._health_check_interval:
            self._last_health_check = now
            self._health_check()

        try:
            conn = self._pool.get_nowait()
        except Empty:
            with self._lock:
                if self._created < self._max:
                    try:
                        conn = self._new_conn()
                        self._created += 1
                    except ConnectionError as e:
                        self._healthy = False
                        raise
                else:
                    conn = self._pool.get()

        # 连接健康检查
        if not self._check_conn_health(conn):
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created -= 1
            try:
                conn = self._new_conn()
                self._created += 1
            except ConnectionError as e:
                self._healthy = False
                raise

        self._healthy = True
        return conn

    def _health_check(self):
        """池级健康检查：移除废弃连接"""
        checked = []
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
            except Empty:
                break
            if self._check_conn_health(conn):
                checked.append(conn)
            else:
                try:
                    conn.close()
                except Exception:
                    pass
                with self._lock:
                    self._created -= 1
        # 归还健康连接
        for conn in checked:
            try:
                self._pool.put_nowait(conn)
            except Full:
                try:
                    conn.close()
                except Exception:
                    pass
                with self._lock:
                    self._created -= 1

        # 如果池中连接不足最小值，补充
        if self._created < MYSQL_POOL_MIN:
            needed = MYSQL_POOL_MIN - self._created
            for _ in range(needed):
                try:
                    conn = self._new_conn()
                    self._pool.put_nowait(conn)
                    self._created += 1
                except (Full, ConnectionError):
                    break

    def is_healthy(self) -> bool:
        """返回连接池健康状态"""
        return self._healthy

    def release(self, conn):
        if conn is None:
            return
        try:
            self._pool.put_nowait(conn)
        except Full:
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created -= 1

    def close_all(self):
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                try:
                    conn.close()
                except Exception:
                    pass
            except Empty:
                break
            except Exception:
                pass
        self._created = 0


class DocumentStore:
    """MySQL 文档存储管理器（高稳定性版）"""

    def __init__(self):
        self._pool = _ConnectionPool(
            min_conn=MYSQL_POOL_MIN,
            max_conn=MYSQL_POOL_MAX,
        )

    def is_healthy(self) -> bool:
        """检查数据库连接是否健康"""
        return self._pool.is_healthy()

    def _execute(self, sql: str, params: tuple = ()) -> Any:
        """执行 SQL（含自动重试）"""
        last_error = None
        for attempt in range(DB_RECONNECT_MAX_ATTEMPTS):
            conn = None
            try:
                conn = self._pool.acquire()
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur
            except (pymysql.Error, ConnectionError) as e:
                last_error = e
                if conn:
                    try:
                        self._pool.release(conn)
                    except Exception:
                        pass
                    conn = None
                if attempt < DB_RECONNECT_MAX_ATTEMPTS - 1:
                    time.sleep(DB_RECONNECT_INTERVAL * (attempt + 1))
            finally:
                if conn:
                    self._pool.release(conn)

        raise ConnectionError(
            f"数据库操作失败（已重试 {DB_RECONNECT_MAX_ATTEMPTS} 次）: {last_error}"
        )

    def _execute_many(self, sql: str, rows: List[tuple]):
        """批量执行（含自动重试）"""
        last_error = None
        for attempt in range(DB_RECONNECT_MAX_ATTEMPTS):
            conn = None
            try:
                conn = self._pool.acquire()
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)
                conn.commit()
                return
            except (pymysql.Error, ConnectionError) as e:
                last_error = e
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        self._pool.release(conn)
                    except Exception:
                        pass
                    conn = None
                if attempt < DB_RECONNECT_MAX_ATTEMPTS - 1:
                    time.sleep(DB_RECONNECT_INTERVAL * (attempt + 1))
            finally:
                if conn:
                    self._pool.release(conn)

        raise ConnectionError(
            f"批量数据库操作失败（已重试 {DB_RECONNECT_MAX_ATTEMPTS} 次）: {last_error}"
        )

    def init_tables(self):
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
        last_error = None
        for attempt in range(DB_RECONNECT_MAX_ATTEMPTS):
            conn = None
            try:
                conn = self._pool.acquire()
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
            except (pymysql.Error, ConnectionError) as e:
                last_error = e
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        self._pool.release(conn)
                    except Exception:
                        pass
                    conn = None
                if attempt < DB_RECONNECT_MAX_ATTEMPTS - 1:
                    time.sleep(DB_RECONNECT_INTERVAL * (attempt + 1))
            finally:
                if conn:
                    self._pool.release(conn)

        raise ConnectionError(
            f"添加文档失败（已重试 {DB_RECONNECT_MAX_ATTEMPTS} 次）: {last_error}"
        )

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


    def reconnect(self):
        """重新连接数据库连接池"""
        self.close()
        self.__init__()
        return True

    def close(self):
        self._pool.close_all()
