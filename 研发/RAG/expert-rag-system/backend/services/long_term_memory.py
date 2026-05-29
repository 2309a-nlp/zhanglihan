# -*- coding: utf-8 -*-
"""
长期记忆模块
使用 MySQL 存储用户的核心信息（名字、偏好、关键事实等）
每次对话后提取并更新，每次对话前注入到 system prompt
"""
import os
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# MySQL 连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "rag",
}

# 尝试导入 pymysql
try:
    import pymysql
    pymysql.install_as_MySQLdb()
except ImportError:
    logger.error("pymysql 未安装，请执行: pip install pymysql")
    raise


def _get_connection():
    """获取 MySQL 数据库连接"""
    conn = pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    return conn


def _init_table():
    """初始化用户记忆表（如果不存在则创建）"""
    conn = _get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id VARCHAR(100) PRIMARY KEY,
                    memory_summary JSON NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
        logger.info("✅ 用户长期记忆表已就绪")
    except Exception as e:
        logger.error(f"❌ 初始化用户记忆表失败: {e}")
        raise
    finally:
        conn.close()


def get_memory(user_id: str) -> Dict:
    """获取用户的长期记忆摘要"""
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT memory_summary FROM user_memory WHERE user_id = %s",
                    (user_id,)
                )
                row = cursor.fetchone()
                if row:
                    summary = row["memory_summary"]
                    if isinstance(summary, str):
                        return json.loads(summary)
                    return summary  # 已经是 dict
                return {}
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"获取记忆失败: {e}")
        return {}


def save_memory(user_id: str, memory_dict: Dict):
    """保存用户的长期记忆摘要"""
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cursor:
                memory_json = json.dumps(memory_dict, ensure_ascii=False)
                cursor.execute(
                    """INSERT INTO user_memory (user_id, memory_summary)
                       VALUES (%s, %s)
                       ON DUPLICATE KEY UPDATE memory_summary = %s""",
                    (user_id, memory_json, memory_json)
                )
            conn.commit()
            logger.info(f"长期记忆已更新: {user_id}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"保存记忆失败: {e}")


def update_memory_from_dialog(user_id: str, question: str, answer: str, llm) -> Dict:
    """
    从对话中提取关键信息并更新用户记忆
    使用大模型分析对话，提取用户的关键个人信息
    """
    from langchain_core.prompts import ChatPromptTemplate

    # 获取现有记忆
    current_memory = get_memory(user_id)
    current_summary = json.dumps(current_memory, ensure_ascii=False)

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能记忆提取器。分析用户的对话，提取关于用户的重要个人信息。
这些信息应该包括：
1. 用户的姓名、称呼、身份
2. 用户提到的个人情况（年龄、职业、健康状态等）
3. 用户的偏好和需求
4. 其他对后续对话有帮助的关键信息

当前已有的用户记忆：{current_memory}

用户最新说的话：{user_message}
AI的回复：{ai_response}

请判断是否有新的重要信息需要记忆。如果有，返回完整的记忆JSON对象（包含新旧信息合并）；
如果没有新信息需要添加，返回 NO_UPDATE。

要求：
- 只记忆事实性信息（用户明确提到的），不要推断猜测
- 每条信息简短扼要
- 用 JSON 格式输出，key 是信息类别（如 name, occupation, preference 等）"""),
        ("human", "请分析并返回结果。")
    ])

    try:
        chain = extract_prompt | llm
        response = chain.invoke({
            "current_memory": current_summary,
            "user_message": question,
            "ai_response": answer
        })

        content = response.content.strip()

        if content == "NO_UPDATE":
            return current_memory

        # 清理可能的 markdown 代码块标记
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        new_memory = json.loads(content)

        # 合并新旧记忆（新信息覆盖旧信息）
        merged = {**current_memory, **new_memory}
        save_memory(user_id, merged)
        logger.info(f"用户 {user_id} 记忆已更新: {json.dumps(merged, ensure_ascii=False)}")
        return merged

    except Exception as e:
        logger.warning(f"记忆提取解析失败: {e}")
        return current_memory


def build_memory_context(user_id: str) -> str:
    """
    构建用户记忆上下文字符串（用于注入 system prompt）
    如果没有记忆信息，返回空字符串
    """
    memory = get_memory(user_id)
    if not memory:
        return ""

    # 格式化为自然语言
    parts = []
    for key, value in memory.items():
        if key in ("name", "称呼"):
            parts.append(f"用户的{key}是{value}")
        elif key in ("age", "年龄"):
            parts.append(f"用户{value}岁")
        elif key in ("occupation", "职业"):
            parts.append(f"用户的职业是{value}")
        elif key in ("preference", "偏好"):
            parts.append(f"用户偏好{value}")
        else:
            parts.append(f"用户信息 - {key}: {value}")

    if parts:
        return "【用户信息备忘】" + "；".join(parts)
    return ""


# 模块加载时自动初始化表
_init_table()
