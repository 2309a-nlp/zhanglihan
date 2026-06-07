# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer, util

# 加载微调后的模型
model = SentenceTransformer("./bge-finetuned-final")

# 测试查询
query = "如何使用榅桲果？"
documents = [
    "榅桲果可以煎汁内服，也可如膏剂服用。",
    "苹果可以直接生吃。",
    "服药时要遵循医嘱，不要自行增加或减少药量。"
]

# 计算相似度
query_emb = model.encode(query)
doc_embs = model.encode(documents)

for doc, emb in zip(documents, doc_embs):
    score = util.cos_sim(query_emb, emb)
    print(f"相似度 {score.item():.4f} - {doc}")

# 找到最相关的文档
scores = util.cos_sim(query_emb, doc_embs)[0]
best_idx = scores.argmax().item()
print(f"\n最佳匹配：{documents[best_idx]}")