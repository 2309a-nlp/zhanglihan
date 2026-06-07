# -*- coding: utf-8 -*-
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset

# 加载之前保存的数据集
print("正在加载预处理的数据集...")
train_dataset = Dataset.load_from_disk("my_train_dataset")
eval_dataset = Dataset.load_from_disk("my_eval_dataset")

print(f"训练集大小: {len(train_dataset)}")
print(f"评估集大小: {len(eval_dataset)}")
print(f"数据样例:\n{train_dataset[0]}")

# 1. 加载模型
model_id = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
print("模型加载完成。")

# 2. 定义损失函数 (套娃损失)
matryoshka_dimensions = [768, 512, 256, 128, 64]
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dimensions)
print("损失函数定义完成。")

# 3. 创建评估器
corpus = dict(zip(eval_dataset['id'], eval_dataset['positive']))
queries = dict(zip(eval_dataset['id'], eval_dataset['anchor']))
relevant_docs = {qid: [qid] for qid in queries.keys()}

matryoshka_evaluators = []
for dim in matryoshka_dimensions:
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"dim_{dim}",
        truncate_dim=dim,
        score_functions={"cosine": cos_sim},
    )
    matryoshka_evaluators.append(ir_evaluator)

sequential_evaluator = SequentialEvaluator(matryoshka_evaluators)
print("评估器创建完成。")

# 4. 定义训练参数
args = SentenceTransformerTrainingArguments(
    output_dir="bge-finetuned-matryoshka",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    fp16=True,  # 使用fp16混合精度
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_dim_128_cosine_ndcg@10",
    logging_steps=50,
    report_to="none",
)

# 5. 创建训练器
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset.select_columns(["positive", "anchor"]),
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=sequential_evaluator,
)

# 6. 开始训练
print("\n--- 开始微调 ---")
trainer.train()

# 7. 保存最终模型
final_model_path = "bge-finetuned-final"
trainer.save_model(final_model_path)
print(f"微调完成，最终模型已保存至: {final_model_path}")

# 8. 微调后评估
print("\n--- 微调后评估 ---")
final_model = SentenceTransformer(final_model_path, device="cuda" if torch.cuda.is_available() else "cpu")
final_results = sequential_evaluator(final_model)
for dim in matryoshka_dimensions:
    key = f"dim_{dim}_cosine_ndcg@10"
    print(f"{key}: {final_results[key]}")