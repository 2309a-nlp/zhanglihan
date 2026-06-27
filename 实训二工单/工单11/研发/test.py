"""
医疗挂号管理 Agent - 端到端测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import database
from agent import MedicalAgent

# 初始化
database.init_db()
database.seed_data()
agent = MedicalAgent(db_path=database.DB_PATH)

# 测试用例
test_queries = [
    "帮我查下张建国医生下周的坐诊时间",
    "牙科最近的号哪天的？",
    "帮我大宝挂一个今天下午儿科专家的号",
    "我之前挂过眼科的专家，帮我再约一次",
    "取消我上周挂的消化内科普通号",
]

print("=" * 60)
print("医疗挂号 Agent - LLM 驱动端到端测试")
print("=" * 60)

for i, query in enumerate(test_queries, 1):
    print(f"\n[测试 {i}] 用户: {query}")
    print("-" * 40)

    try:
        result = agent.chat(query)
        print(f"Agent: {result}")
    except Exception as e:
        print(f"错误: {e}")

    # 清空历史以避免上下文干扰
    agent.messages = [{"role": "system", "content": agent.messages[0]["content"]}]
    print()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
