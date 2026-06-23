"""
日程提醒智能体 - Schedule Reminder Agent
基于 Flask + LangChain + DeepSeek API 实现自然语言日程管理
支持：添加日程 / 查询日程 / 取消日程 / 到时提醒
"""

import sys
import os
import json
import sqlite3
import random
from datetime import datetime, date, timedelta
from flask import Flask, render_template, request, jsonify

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

# 修复 Windows 终端编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 对话历史存储（内存中，保留最近 6 轮对话）
# ============================================================
_chat_histories = {}
MAX_HISTORY = 12


def get_history(session_id: str):
    return _chat_histories.get(session_id, [])


def add_to_history(session_id: str, user_msg: str, ai_msg: str):
    if session_id not in _chat_histories:
        _chat_histories[session_id] = []
    history = _chat_histories[session_id]
    history.append(HumanMessage(content=user_msg))
    history.append(AIMessage(content=ai_msg))
    if len(history) > MAX_HISTORY:
        _chat_histories[session_id] = history[-MAX_HISTORY:]


# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)

# ============================================================
# 数据库
# ============================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schedules.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT,
            status TEXT DEFAULT 'pending',
            notified INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()
    print("✅ 数据库初始化完成")


# ============================================================
# 加载 .env 文件
# ============================================================
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip(chr(34)).strip(chr(39))
                if key and key not in os.environ:
                    os.environ[key] = value


_load_env()

# ============================================================
# DeepSeek API 配置
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

if not DEEPSEEK_API_KEY:
    print("=" * 60)
    print("⚠️  未配置 DEEPSEEK_API_KEY！")
    print("   1. 复制 .env.example 为 .env")
    print("   2. 在 .env 中填入你的 API Key")
    print("=" * 60)

# ============================================================
# 提醒语句库
# ============================================================
REMINDER_TEMPLATES = [
    "温馨提醒：（{content}）的时间到啦，主人！",
    "主人！是时候{content}了喔~",
    "亲爱的主人，现在是{content}的时候啦！",
    "嘿，主人，该{content}了哦~",
]


def get_reminder_message(content: str) -> str:
    return random.choice(REMINDER_TEMPLATES).format(content=content)


def get_today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def get_today_cn() -> str:
    return date.today().strftime("%Y年%m月%d日")


def get_tomorrow_str() -> str:
    return (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")


def get_yesterday_str() -> str:
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


# ============================================================
# LangChain 工具定义
# ============================================================

@tool
def add_schedule(content: str, date: str, time: str = None) -> str:
    """
    添加一条日程提醒到数据库。
    当用户想要记录、添加、提醒某个事项时调用此工具。
    调用前必须先询问用户确认，用户确认后才调用。

    Args:
        content: 事项内容，如"买咖啡""开会""去医院"
        date: 日程日期，格式 YYYY-MM-DD
        time: 日程时间，格式 HH:MM，可选（全天日程可不传）
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO schedules (content, date, time, status) VALUES (?, ?, ?, 'pending')",
        (content, date, time),
    )
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    result = {"status": "success", "id": record_id, "content": content, "date": date, "time": time}
    print(f"✅ 日程已添加: {json.dumps(result, ensure_ascii=False)}")
    return json.dumps(result, ensure_ascii=False)


@tool
def list_schedules(date: str = None, status: str = None) -> str:
    """
    查询日程列表。按日期查询某天的所有日程，也可按状态筛选。
    当用户想查看、列出日程安排时调用此工具。

    Args:
        date: 查询的日期，格式 YYYY-MM-DD，不传则默认为今天
        status: 状态筛选，"pending"=待办,"completed"=已完成,"cancelled"=已取消，不传则查全部
    """
    if not date:
        date = get_today_str()

    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM schedules WHERE date = ?"
    params = [date]

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY time IS NULL, time ASC, id ASC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    records = []
    for row in rows:
        records.append({
            "id": row["id"],
            "content": row["content"],
            "date": row["date"],
            "time": row["time"],
            "status": row["status"],
        })

    result = {"count": len(records), "date": date, "records": records}
    print(f"📋 查询日程: {date} 共 {len(records)} 条")
    return json.dumps(result, ensure_ascii=False)


@tool
def delete_schedule(schedule_id: int) -> str:
    """
    删除一条日程。按 ID 从数据库中删除指定的日程记录。
    调用前必须先调用 list_schedules 查出该日程，向用户展示确认后执行。

    Args:
        schedule_id: 要删除的日程 ID（整数）
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 先查是否存在
    cursor.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return json.dumps({"status": "error", "message": f"未找到ID为 {schedule_id} 的日程"}, ensure_ascii=False)

    record = {"id": row["id"], "content": row["content"], "date": row["date"], "time": row["time"]}

    cursor.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
    conn.commit()
    conn.close()

    result = {"status": "success", "message": "日程已删除", "deleted": record}
    print(f"🗑️ 日程已删除: {json.dumps(result, ensure_ascii=False)}")
    return json.dumps(result, ensure_ascii=False)


@tool
def get_pending_reminders() -> str:
    """
    获取当前时间需要提醒的日程。
    检查数据库中状态为 pending、未被通知过、且时间已到的日程。
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")

    conn = get_db_connection()
    cursor = conn.cursor()

    # 查询需要提醒的日程（时间已到且未通知）
    cursor.execute("""
        SELECT * FROM schedules
        WHERE status = 'pending'
          AND notified = 0
          AND date <= ?
          AND (time IS NULL OR time <= ?)
        ORDER BY date ASC, time ASC
    """, (today_str, current_time))

    rows = cursor.fetchall()
    reminders = []
    for row in rows:
        reminders.append({
            "id": row["id"],
            "content": row["content"],
            "date": row["date"],
            "time": row["time"],
        })
        # 标记为已通知
        cursor.execute("UPDATE schedules SET notified = 1 WHERE id = ?", (row["id"],))

    conn.commit()
    conn.close()

    result = {"count": len(reminders), "reminders": reminders, "checked_at": now.strftime("%H:%M:%S")}
    if reminders:
        print(f"⏰ 有待提醒日程: {len(reminders)} 条")
    return json.dumps(result, ensure_ascii=False)


tools = [add_schedule, list_schedules, delete_schedule, get_pending_reminders]

# ============================================================
# 系统提示词
# ============================================================
def get_system_prompt() -> str:
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    today_cn = today.strftime("%Y年%m月%d日")
    tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    return f"""你是一位可爱贴心的日程提醒管家，帮助用户管理日程安排。你的任务是准确理解用户意图，调用正确的数据库工具完成操作。

# 当前时间
- 今天：{today_cn}（{today_str}）
- 明天：{tomorrow_str}
- 昨天：{yesterday_str}
- 当前时间：{datetime.now().strftime('%H:%M')}

# 核心能力

## 1. 添加日程 → add_schedule
**触发词**：提醒、记着、记得、安排、添加、加个、帮我记、别忘了
**操作流程**：
  1. 提取事项内容、日期、时间
  2. 如果信息不完整 → 主动追问（缺什么问什么）
  3. 展示完整信息让用户确认 → "我将添加日程：【内容】，日期：【日期】，时间：【时间】，确认吗？"
  4. 用户确认后才调用 add_schedule
  5. 添加成功后回复："✅ 已添加日程：【内容】 📅 【日期】 ⏰ 【时间】"

## 2. 查询日程 → list_schedules
**触发词**：查、看看、哪些、有什么安排、日程、计划、今天有事吗
**操作**：调用 list_schedules(date=目标日期) → 展示结果
**无日程时**："今天还没有日程安排哦，需要我帮你记一个吗？😊"
**有日程时**：列出编号、事项、时间、状态

## 3. 取消/删除日程 → delete_schedule
**触发词**：取消、删除、去掉、不要了、移除
**操作流程**：
  1. 用户说"取消日程N" → 先调用 list_schedules(date=今天) 查出所有日程
  2. 找到列表中第N条对应的 id
  3. 展示该日程给用户确认 → "你确定要取消【事项内容】(【日期】 【时间】)吗？"
  4. 用户确认后调用 delete_schedule(schedule_id=对应的ID)
  5. 删除成功后回复："🗑️ 已取消日程：【事项内容】"

## 4. 查看待提醒 → get_pending_reminders
**触发词**：到点提醒我、有什么要提醒的、看看提醒
**操作**：调用 get_pending_reminders()
**有提醒时**：从以下格式中随机选一条输出：
  - "温馨提醒：（{{事项内容}}）的时间到啦，主人！"
  - "主人！是时候{{事项内容}}了喔~"
  - "亲爱的主人，现在是{{事项内容}}的时候啦！"
  - "嘿，主人，该{{事项内容}}了哦~"

# 完整性引导（极其重要）
当用户输入信息不完整时，必须主动询问补充：
- 用户说"提醒我买咖啡" → 问："好的！请问买咖啡安排在哪天、什么时间呢？"
- 用户说"明天开会" → 问："好的！明天开会具体什么时间呢？"
- 用户说"下午3点有事" → 问："下午3点具体是什么事项呢？"
- 用户说"后天" → 问："后天具体是什么事、什么时间呢？"

# 口语化理解
- "提醒我xxx" → 添加日程
- "明天xxx" / "明天下午xxx" → 日期=明天，解析时间
- "下午3点xxx" → 时间=15:00
- "后天xxx" → 日期=后天
- "上午/下午/晚上" → 分别解析为 上午=09:00, 中午=12:00, 下午=14:00, 晚上=19:00

# 数据库调用强制规则
在任何情况下，必须调用数据库工具来获取或存储数据。
禁止凭空编造信息。
即使用户说"简单查一下就行"，也必须走数据库工具。
每次响应前先想：这次需要调用工具吗？需要 → 必须调。

# 先确认后操作
- 添加日程：先展示全部信息 → 用户确认 → 调工具
- 删除日程：先查出来展示 → 用户确认 → 调工具
- 用户说"确认""是的""对""好""可以" → 执行之前展示的操作
- 用户说"不用了""算了""取消" → 不执行

# 对话风格
- 亲切可爱，像贴心的管家
- 善用 emoji（📅 ✅ ❌ 🗑️ ⏰ 🔔 📋）
- 主动询问缺失信息
- 查询结果清晰列出：编号、事项、时间、状态
- 使用"主人""亲爱的主人"称呼用户"""


# ============================================================
# 创建 Agent
# ============================================================
def create_schedule_agent():
    if not DEEPSEEK_API_KEY:
        return None

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.7,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=get_system_prompt(),
    )


# ============================================================
# Flask 路由
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请提供有效的JSON数据"}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    try:
        agent = create_schedule_agent()
        if agent is None:
            return jsonify({"error": "API Key 未配置", "response": "⚠️ 请先配置 DeepSeek API Key"}), 500

        session_id = data.get("session_id", "default")
        history = get_history(session_id)
        input_messages = history + [HumanMessage(content=user_message)]

        result = agent.invoke({"messages": input_messages})

        result_messages = result.get("messages", [])
        response_text = "抱歉，我无法处理您的请求，请稍后再试。"

        for msg in reversed(result_messages):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                response_text = msg.content
                break

        add_to_history(session_id, user_message, response_text)

        # 同时检查是否有待提醒
        reminder_text = check_and_get_reminders()
        if reminder_text:
            response_text = reminder_text + "\n\n" + response_text

        return jsonify({"response": response_text, "session_id": session_id})

    except Exception as e:
        error_msg = str(e)
        print(f"❌ 错误: {error_msg}")
        friendly_msg = f"⚠️ 处理请求时出现错误，请稍后再试。\n\n错误详情：{error_msg}"
        return jsonify({"response": friendly_msg, "error": error_msg}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    api_status = "configured" if DEEPSEEK_API_KEY else "missing"
    return jsonify({
        "status": "ok",
        "api_key": api_status,
        "database": "connected",
    })


@app.route("/api/check_reminders", methods=["GET"])
def check_reminders():
    """检查待提醒日程（供前端轮询）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")

    cursor.execute("""
        SELECT * FROM schedules
        WHERE status = 'pending'
          AND notified = 0
          AND date <= ?
          AND (time IS NULL OR time <= ?)
        ORDER BY date ASC, time ASC
    """, (today_str, current_time))

    rows = cursor.fetchall()
    reminders = []
    for row in rows:
        content = row["content"]
        reminder_msg = get_reminder_message(content)
        reminders.append({
            "id": row["id"],
            "content": content,
            "date": row["date"],
            "time": row["time"],
            "message": reminder_msg,
        })
        cursor.execute("UPDATE schedules SET notified = 1 WHERE id = ?", (row["id"],))

    conn.commit()
    conn.close()
    return jsonify({"count": len(reminders), "reminders": reminders})


def check_and_get_reminders() -> str:
    """内部检查提醒，返回提醒文本"""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")

    cursor.execute("""
        SELECT * FROM schedules
        WHERE status = 'pending'
          AND notified = 0
          AND date <= ?
          AND (time IS NULL OR time <= ?)
        ORDER BY date ASC, time ASC
    """, (today_str, current_time))

    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return ""

    messages = []
    for row in rows:
        messages.append(get_reminder_message(row["content"]))
        cursor.execute("UPDATE schedules SET notified = 1 WHERE id = ?", (row["id"],))

    conn.commit()
    conn.close()
    return "\n".join(messages)


@app.route("/api/schedules", methods=["GET"])
def get_schedules():
    """REST: 获取日程列表"""
    date_param = request.args.get("date", get_today_str())
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM schedules WHERE date = ? ORDER BY time IS NULL, time ASC, id ASC",
        (date_param,),
    )
    rows = cursor.fetchall()
    conn.close()
    records = [{"id": r["id"], "content": r["content"], "date": r["date"],
                 "time": r["time"], "status": r["status"], "notified": r["notified"]}
               for r in rows]
    return jsonify({"count": len(records), "date": date_param, "records": records})


@app.route("/api/schedules", methods=["POST"])
def add_schedule_rest():
    """REST: 添加日程"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请提供JSON数据"}), 400

    content = data.get("content", "").strip()
    date_val = data.get("date", get_today_str()).strip()
    time_val = data.get("time")
    confirm = data.get("confirm", False)

    if not content:
        return jsonify({"error": "事项内容不能为空"}), 400

    if not confirm:
        return jsonify({
            "status": "need_confirm",
            "message": f"即将添加日程：【{content}】日期：【{date_val}】时间：【{time_val or '全天'}】",
            "preview": {"content": content, "date": date_val, "time": time_val}
        })

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO schedules (content, date, time, status) VALUES (?, ?, ?, 'pending')",
        (content, date_val, time_val),
    )
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return jsonify({"status": "success", "id": record_id, "content": content, "date": date_val, "time": time_val})


@app.route("/api/schedules/<int:schedule_id>", methods=["DELETE"])
def delete_schedule_rest(schedule_id):
    """REST: 删除日程"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"status": "error", "message": f"日程 {schedule_id} 不存在"}), 404

    cursor.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": f"日程 {schedule_id} 已删除"})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    if session_id in _chat_histories:
        del _chat_histories[session_id]
    return jsonify({"status": "success", "message": "对话历史已清空"})


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("📅 日程提醒智能体 启动中...")
    print("   支持：添加日程 | 查询日程 | 取消日程 | 到时提醒")
    print("=" * 60)
    print(f"🌐 访问地址: http://127.0.0.1:5000")
    print(f"🔑 API Key: {'✅ 已配置' if DEEPSEEK_API_KEY else '❌ 未配置'}")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
