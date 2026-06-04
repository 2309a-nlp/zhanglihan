# -*- coding: utf-8 -*-
# 1. 导入必要的库
import os
import logging
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# ==================== 日志系统配置 ====================
# 配置日志格式：时间 - 级别 - 具体信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==================== 1. 连接本地 Milvus 向量数据库 ====================
try:
    logging.info("正在连接 Milvus...")
    connections.connect("default", host="localhost", port="19530")
    logging.info("✅ 成功连接到 Milvus!")
except Exception as e:
    logging.warning(f"Milvus 连接失败（服务未启动）: {e}")
    logging.warning("对话存储到 Milvus 功能将不可用")

try:
    logging.info("正在加载向量化模型...")
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _model_path = os.path.join(_current_dir, "bge-m3")
    model = SentenceTransformer(_model_path)
    logging.info("✅ 模型加载完成！")
except Exception as e:
    logging.warning(f"向量模型加载失败: {e}")
    model = None

# ==================== 3. 定义并创建集合（相当于关系型数据库中的“表”） ====================
collection_name = "chat_history_v3"

# 检查该集合是否已经存在
if utility.has_collection(collection_name):
    collection = Collection(collection_name)  # 如果存在，直接获取
    logging.info(f"✅ 找到已存在的集合: {collection_name}")
else:
    # 如果不存在，则定义表的结构（字段）
    fields = [
        # 主键 ID，设为自增（auto_id=True），不需要手动传值
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="dialog_id", dtype=DataType.VARCHAR, max_length=100),  # 会话/对话ID
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),  # 用户ID
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=65535),  # 用户提问的内容
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),  # AI 回答的内容
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),  # 核心：存储文本对应的 1024 维向量
        FieldSchema(name="timestamp", dtype=DataType.INT64)  # 记录时间戳
    ]
    # 将字段组合成完整的表结构描述
    schema = CollectionSchema(fields, description="Expert RAG Chat History")
    # 在 Milvus 中创建这个新集合
    collection = Collection(collection_name, schema, consistency_level="Strong")
    logging.info(f"✅ 成功创建新集合: {collection_name}")

    # 4. 为向量字段创建索引（IVF_FLAT 是一种常用的向量索引，用于加速后续的检索速度）
    index_params = {
        "metric_type": "L2",  # 使用 L2 距离（欧氏距离）来衡量向量相似度
        "index_type": "IVF_FLAT",  # 索引类型
        "params": {"nlist": 128}  # 将数据分成 128 个聚类中心
    }
    collection.create_index(field_name="vector", index_params=index_params)
    logging.info("✅ 成功建立索引")


# ==================== 5. 封装函数：保存对话记录到 Milvus ====================
def save_chat_to_milvus(dialog_id, user_id, question, answer):
    try:
        # 将“问题”和“回答”拼接成一段完整的文本
        text_to_embed = question + " " + answer
                # 使用加载好的模型将这段文本转化为 1024 维的向量
        vector = model.encode([text_to_embed]).tolist()[0]

        # 准备要插入的数据（注意：Milvus 原生接口要求数据以列（字段）的形式传入列表）
        # 顺序必须和上面 fields 定义的顺序（除自增主键外）保持一致
        data = [
            [dialog_id],
            [user_id],
            [question],
            [answer],
            [vector],
            [int(time.time())]  # 获取当前时间戳
        ]

        collection.insert(data)  # 执行插入操作
        collection.load()  # 插入后加载到内存，Attu 才能刷新看到数据
        logging.info(f"💾 成功将对话存入 Milvus (用户: {user_id})")
        return True
    except Exception as e:
        logging.error(f"❌ Milvus 插入失败: {e}")
        return False


# ==================== 6. 封装函数：根据用户 ID 查询历史聊天记录 ====================
def get_chat_history(user_id, limit=10):
    try:
        collection.load()  # 查询前，先将集合加载到内存中
        # 根据 user_id 进行过滤查询，获取最近的 limit 条记录
        results = collection.query(
            expr=f"user_id == '{user_id}'",  # 过滤表达式：只查当前用户的
            output_fields=["question", "answer", "timestamp"],  # 指定需要返回的字段
            limit=limit  # 限制返回的条数
        )
        return results
    except Exception as e:
        logging.error(f"❌ Milvus 查询失败: {e}")
        return []


# ==================== 7. 本地测试代码 ====================
# 只有当直接运行这个 .py 文件时，下面的代码才会执行
if __name__ == "__main__":
    logging.info("\n--- 开始测试 Milvus 存储功能 ---")

    # 模拟保存一条对话记录
    save_chat_to_milvus("test_session_001", "user_zhangsan", "你好，我想了解一下RAG技术",
                        "RAG是检索增强生成，可以让大模型结合外部知识库来回答问题。")

    # 模拟查询刚才保存的历史记录
    history = get_chat_history("user_zhangsan")
    logging.info(f"\n--- 查询到 {len(history)} 条历史记录 ---")
    for item in history:
        logging.info(f"问: {item['question']}")
        logging.info(f"答: {item['answer']}")
        # 将时间戳转换回人类可读的格式
        logging.info(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['timestamp']))}\n")