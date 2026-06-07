import pandas as pd
from datasets import Dataset

# 1. 直接从CSV加载数据
df = pd.read_csv("data.csv")

# 2. 检查数据格式，打印前几行确认
print("数据预览：")
print(df.head())
print(f"\n总数据量：{len(df)}条")

# 3. 重命名列名为模型需要的格式
df = df.rename(columns={
    "question": "anchor",
    "answer": "positive"
})

# 4. 添加ID列（评估时需要）
df['id'] = df.index.astype(str)

# 5. 检查是否有空值
print(f"\n空值检查：")
print(f"anchor列空值数：{df['anchor'].isna().sum()}")
print(f"positive列空值数：{df['positive'].isna().sum()}")

# 如果有空值，删除它们
df = df.dropna(subset=['anchor', 'positive'])

# 6. 转换为 HuggingFace Dataset 格式
dataset = Dataset.from_pandas(df[["id", "anchor", "positive"]])

# 7. 分割训练集和评估集（90%训练，10%评估）
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"\n训练集大小: {len(train_dataset)}")
print(f"评估集大小: {len(eval_dataset)}")

# 8. 保存处理好的数据集（修正：使用 save_to_disk 而不是 to_disk）
train_dataset.save_to_disk("my_train_dataset")
eval_dataset.save_to_disk("my_eval_dataset")

print("\n数据集已保存！")
print("训练集样例：")
print(train_dataset[0])