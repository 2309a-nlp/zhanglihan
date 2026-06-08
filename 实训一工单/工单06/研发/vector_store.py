"""
向量存储模块 — 基于 FAISS
支持：多种嵌入模型、增量添加、语义检索、归一化评分
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS 向量存储
    支持多种嵌入模型切换，余弦相似度检索
    """

    def __init__(self):
        self.index = None
        self.documents = []      # [{text, source, chunk_id, embedding_model}, ...]
        self.embeddings = None   # numpy array
        self.embedding_model = None
        self.dimension = None
        self._built = False

    def build(self, documents: List[dict], embeddings: np.ndarray):
        """
        构建向量索引

        参数:
            documents: [{"text": "...", "source": "file.pdf", "chunk_id": 0}, ...]
            embeddings: numpy array, shape=(n, dim)
        """
        import faiss

        self.documents = documents
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]
        self.embedding_model = self._detect_model()

        # 构建 FAISS 索引 (Inner Product ≈ 归一化后的余弦相似度)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self._built = True
        logger.info(f"向量索引构建完成: {len(documents)} 文档, dim={self.dimension}")

    def _detect_model(self) -> str:
        """检测使用的嵌入模型"""
        from config import EMBEDDING_MODELS
        if self.embeddings is not None:
            dim = self.embeddings.shape[1]
            for name, cfg in EMBEDDING_MODELS.items():
                if cfg["dim"] == dim:
                    return name
        return "unknown"

    def add_documents(self, documents: List[dict], embeddings: np.ndarray):
        """增量添加文档"""
        if not self._built:
            self.build(documents, embeddings)
            return

        import faiss
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)
        self.embeddings = np.vstack([self.embeddings, embeddings]) if self.embeddings is not None else embeddings
        logger.info(f"增量添加 {len(documents)} 文档")

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[tuple]:
        """
        向量检索

        参数:
            query_embedding: 查询向量
            top_k: 返回数

        返回:
            [(document, score), ...]
        """
        if not self._built or self.index is None:
            logger.warning("向量索引尚未构建")
            return []

        if len(self.documents) == 0:
            return []

        actual_k = min(top_k, len(self.documents))
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)

        distances, indices = self.index.search(query_vec, actual_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            score = float(distances[0][i])
            doc = self.documents[idx]
            results.append((doc, score))

        return results

    def save(self, path: str):
        """保存向量索引到磁盘"""
        import faiss, tempfile
        os.makedirs(path, exist_ok=True)

        # faiss C++ 不支持含中文的路径，先用临时 ASCII 路径写入再复制
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
        os.close(tmp_fd)
        try:
            faiss.write_index(self.index, tmp_path)
            import shutil
            shutil.copy2(tmp_path, os.path.join(path, "faiss_index.faiss"))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        meta = {
            "documents": self.documents,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"向量索引已保存: {path}")

    def load(self, path: str) -> bool:
        """从磁盘加载向量索引"""
        import faiss, tempfile

        index_path = os.path.join(path, "faiss_index.faiss")
        meta_path = os.path.join(path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        # faiss C++ 不支持含中文的路径，复制到临时 ASCII 路径再读取
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
        os.close(tmp_fd)
        try:
            import shutil
            shutil.copy2(index_path, tmp_path)
            self.index = faiss.read_index(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.documents = meta["documents"]
        self.embedding_model = meta.get("embedding_model")
        self.dimension = meta.get("dimension")
        self._built = True

        logger.info(f"向量索引已加载: {len(self.documents)} 文档, dim={self.dimension}")
        return True
