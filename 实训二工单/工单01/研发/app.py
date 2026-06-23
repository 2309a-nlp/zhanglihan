"""
智能记账本 - Smart Expense Tracker
使用 LangChain + DeepSeek API 实现自然语言记账和查询
支持：支出记录 / 收入记录 / 查询 / 删除
"""

import sys
import os
import json
import sqlite3
from datetime import timedelta, date
from flask import Flask, render_template, request, jsonify

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

# 修复 Windows 终端中文/Emoji 编码问题
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

MAX_HISTORY = 12  # 最多保留 12 条历史消息（6 轮对话）


def get_history(session_id: str):
    """获取对话历史"""
    return _chat_histories.get(session_id, [])


def add_to_history(session_id: str, user_msg: str, ai_msg: str):
    """添加一轮对话到历史"""
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
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expenses.db")


def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row):
    """将数据库行转为字典 —— 消除 search_transactions 与 get_all_records 的重复"""
    return {
        "id": row["id"],
        "type": row["type"],
        "date": row["date"],
        "category": row["category"],
        "project": row["project"],
        "amount": row["amount"],
        "owner": row["owner"],
        "note": row["note"],
        "created_at": row["created_at"],
    }


def _record_transaction(type_: str, date: str, category: str, project: str,
                        amount: float, owner: str, note: str) -> dict:
    """记录一笔收支的公共逻辑 —— 消除 record_expense / record_income 重复"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transactions (type, date, category, project, amount, owner, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (type_, date, category, project, amount, owner, note),
    )
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return {
        "status": "success", "id": record_id, "type": type_,
        "date": date, "category": category, "project": project,
        "amount": amount, "owner": owner, "note": note,
    }


def _get_record_by_id(record_id: int):
    """按 ID 查询一条记录，不存在返回 None —— 消除 delete 相关重复"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def _delete_by_id(record_id: int):
    """按 ID 删除记录"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def init_db():
    """初始化数据库表（兼容旧表升级）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL DEFAULT 'expense',
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            project TEXT NOT NULL,
            amount REAL NOT NULL,
            owner TEXT DEFAULT '本人',
            note TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # 兼容旧表迁移
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expenses'")
    if cursor.fetchone():
        try:
            cursor.execute("""
                INSERT INTO transactions (type, date, category, project, amount, owner, note, created_at)
                SELECT 'expense', date, category, project, amount, owner, note, created_at
                FROM expenses
            """)
            cursor.execute("DROP TABLE expenses")
            conn.commit()
            print("📦 旧数据已迁移到新表")
        except Exception as e:
            print(f"⚠️ 数据迁移跳过: {e}")

    conn.commit()
    conn.close()
    print("✅ 数据库初始化完成")


# ============================================================
# 加载 .env 文件（本地开发用，生产环境直接设环境变量）
# ============================================================
def _load_env():
    """从项目根目录的 .env 文件加载环境变量（如果文件存在）"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


_load_env()

# ============================================================
# DeepSeek API 配置
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

if not DEEPSEEK_API_KEY:
    print("=" * 60)
    print("⚠️  未配置 DEEPSEEK_API_KEY！")
    print("   1. 复制 .env.example 为 .env")
    print("   2. 在 .env 中填入你的 API Key")
    print("=" * 60)

# ============================================================
# LangChain 工具定义
# ============================================================

@tool
def record_expense(
    date: str, category: str, project: str, amount: float,
    owner: str = "本人", note: str = ""
) -> str:
    """
    记录一笔支出一花钱/消费/付款到数据库。
    当用户说"买了"、"花了"、"消费"、"支付"、"用了...钱"、"转了"等表示花钱的关键词时调用此工具。

    Args:
        date: 消费日期，格式 YYYY-MM-DD
        category: 消费类别。可选值：书籍、餐饮、日用品、交通、娱乐、服饰、教育、医疗、住房、通讯、人情、其他
        project: 具体的消费项目名称，如：三体、咖啡、地铁票
        amount: 消费金额（正数，表示支出了多少钱）
        owner: 归属人（谁花的钱），默认为"本人"
        note: 备注信息，可选
    """
    result = _record_transaction("expense", date, category, project, amount, owner, note)
    print(f"✅ 支出记账成功: {json.dumps(result, ensure_ascii=False)}")
    return json.dumps(result, ensure_ascii=False)


@tool
def record_income(
    date: str, category: str, project: str, amount: float,
    owner: str = "本人", note: str = ""
) -> str:
    """
    记录一笔收入/进账到数据库。
    当用户说"赚了"、"收入"、"发了工资"、"收到"、"进账"、"到账"、"奖金"、"报销"等表示进钱的关键词时调用此工具。

    Args:
        date: 收入日期，格式 YYYY-MM-DD
        category: 收入类别。可选值：工资、奖金、理财收益、兼职、报销、红包收入、二手出售、其他收入
        project: 具体的收入来源名称，如：6月工资、项目奖金、基金收益
        amount: 收入金额（正数，表示收入了多少钱）
        owner: 归属人（谁的收入），默认为"本人"
        note: 备注信息，可选
    """
    result = _record_transaction("income", date, category, project, amount, owner, note)
    print(f"✅ 收入记账成功: {json.dumps(result, ensure_ascii=False)}")
    return json.dumps(result, ensure_ascii=False)


@tool
def search_transactions(
    type: str = None,
    start_date: str = None,
    end_date: str = None,
    category: str = None,
    owner: str = None,
    keyword: str = None
) -> str:
    """
    查询数据库中的收支记录，支持按类型（收入/支出）、时间范围、类别、归属人、关键词筛选。
    当用户想要查询、查看、统计收支记录时调用此工具。
    查完后可根据用户需求进一步删除某条记录（调用 delete_transaction）。

    Args:
        type: 记录类型筛选，"expense"=支出，"income"=收入，不传则查询全部
        start_date: 查询起始日期，格式 YYYY-MM-DD，可选
        end_date: 查询结束日期，格式 YYYY-MM-DD，可选
        category: 类别筛选，可选
        owner: 归属人筛选，可选
        keyword: 项目名称关键词搜索，可选
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM transactions WHERE 1=1"
    params = []

    if type:
        query += " AND type = ?"
        params.append(type)
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if category:
        query += " AND category = ?"
        params.append(category)
    if owner:
        query += " AND owner LIKE ?"
        params.append(f"%{owner}%")
    if keyword:
        query += " AND (project LIKE ? OR note LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])

    query += " ORDER BY date DESC, id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    records = []
    total_expense = 0
    total_income = 0
    for row in rows:
        records.append(_row_to_dict(row))
        if row["type"] == "expense":
            total_expense += row["amount"]
        else:
            total_income += row["amount"]

    result = {
        "total_expense": round(total_expense, 2),
        "total_income": round(total_income, 2),
        "net": round(total_income - total_expense, 2),
        "count": len(records),
        "records": records,
    }
    print(f"🔍 查询结果: {len(records)} 条, 支出{total_expense}元, 收入{total_income}元")
    return json.dumps(result, ensure_ascii=False)


@tool
def delete_transaction(record_id: int) -> str:
    """
    删除数据库中的一条收支记录。
    当用户说"删除"、"去掉"、"取消"、"删掉"某条记录时，先调用 search_transactions 找到目标记录的 id，
    然后调用此工具进行删除。

    Args:
        record_id: 要删除的记录ID（整数），从 search_transactions 返回结果中获取
    """
    record = _get_record_by_id(record_id)
    if not record:
        return json.dumps({"status": "error", "message": f"未找到ID为 {record_id} 的记录"}, ensure_ascii=False)

    _delete_by_id(record_id)

    deleted = {k: v for k, v in record.items() if k != "created_at"}
    result = {"status": "success", "message": "记录已删除", "deleted_record": deleted}
    print(f"🗑️ 删除成功: {json.dumps(result, ensure_ascii=False)}")
    return json.dumps(result, ensure_ascii=False)


tools = [record_expense, record_income, search_transactions, delete_transaction]

# ============================================================
# 系统提示词
# ============================================================
def get_system_prompt() -> str:
    """获取带有当前日期的系统提示词"""
    today = date.today()
    current_month_start = today.replace(day=1)
    yesterday = today - timedelta(days=1)
    day_before_yesterday = today - timedelta(days=2)

    return f"""你是一位专业的智能记账助手，帮助用户记录和管理个人财务收支。你需要准确理解用户意图，并调用正确的工具完成操作。

# 当前时间信息
- 今天是：{today.strftime('%Y-%m-%d')}（{today.strftime('%Y年%m月%d日')}）
- 昨天是：{yesterday.strftime('%Y-%m-%d')}
- 前天是：{day_before_yesterday.strftime('%Y-%m-%d')}
- 本月范围：{current_month_start.strftime('%Y-%m-%d')} 至 {today.strftime('%Y-%m-%d')}

# 核心能力
1. **记支出**：当用户说花了钱 → 调用 record_expense 保存
2. **记收入**：当用户说有进账 → 调用 record_income 保存
3. **查账单**：当用户想查看记录 → 调用 search_transactions 检索后回复
4. **删记录**：当用户想删除某条 → 先 search_transactions 找到ID，再 delete_transaction 删除

# 意图判断规则

## 1. 支出记账 → 调用 record_expense
**触发词**：买了、花了、支付、消费、支出、用了、转了、付款、买单、掏钱
**操作**：提取日期/类别/项目/金额/归属人，调用 record_expense
**回复格式**："✅ 已记录支出：[日期] [类别] [项目] [金额]元 [归属人]"

## 2. 收入记账 → 调用 record_income
**触发词**：赚了、收入、进账、到账、发了工资、收到、领了、奖金、报销、退款、卖二手
**操作**：提取日期/类别/项目/金额/归属人，调用 record_income
**回复格式**："💰 已记录收入：[日期] [类别] [项目] +[金额]元 [归属人]"

## 3. 查询账单 → 调用 search_transactions
**触发词**：查、哪天、什么时候、多少、花了多少、赚了多少、买了什么、看看、统计、汇总、账单、明细、还剩多少
**操作**：解析查询条件 → 调用 search_transactions → 基于结果清晰回答
**回复格式**：先总结，再列明细。收支都要展示清楚。

## 4. 删除记录 → 查到就删，不用确认！
**触发词**：删除、去掉、取消、删掉、不要了、移除、确认删除
**核心原则**：查到就删，绝对不要问"确认删除吗？"！用户说删除就是确认！

**操作流程**：
  1. 调用 search_transactions 找到匹配记录
  2. 如果只有1条匹配 → 立即调用 delete_transaction(record_id=xxx) 删除，回复删除成功
  3. 如果有多条匹配 → 列出编号让用户选一条，用户回复后立即删，别再确认
  4. 用户说"确认"、"对"、"是"、"删吧" = 上一轮已经选了，你需要在本轮搜索结果基础上直接删

**回复格式**："🗑️ 已删除：[日期] [类型] [项目] [金额]元"

**错误示例（禁止）**：
- ❌ "找到三体记录，确认删除吗？" → 直接删！
- ❌ "你确定要删除吗？" → 绝对不要问！
- ❌ 用户说"确认"后还不动手 → 立刻调 delete_transaction

# 分类推断规则

## 支出类别（12类）
- 书/小说/漫画/杂志 → 书籍
- 咖啡/奶茶/吃饭/餐厅/外卖/烧烤/火锅/早餐/午餐/晚餐/零食/饮料/水果/买菜 → 餐饮
- 淘宝/京东/超市/日用/纸巾/洗发水/洗衣液/牙膏 → 日用品
- 打车/地铁/公交/机票/火车票/高铁/加油/停车/共享单车 → 交通
- 电影/游戏/演唱会/KTV/旅游/景点/门票 → 娱乐
- 衣服/鞋/包/裤子/裙子/帽子 → 服饰
- 课/培训/教材/学费/考试 → 教育
- 医院/药/挂号/体检/诊所 → 医疗
- 房租/房贷/物业/水电/煤气 → 住房
- 话费/网费/手机 → 通讯
- 红包/礼物/结婚/随礼/请客 → 人情
- 无法归类 → 其他

## 收入类别（8类）
- 工资/发工资/薪水/月薪 → 工资
- 奖金/年终奖/绩效 → 奖金
- 理财/基金/股票/利息/收益 → 理财收益
- 兼职/副业/接单/外快 → 兼职
- 报销/公司报销 → 报销
- 红包/收到红包/压岁钱 → 红包收入
- 卖二手/闲鱼/卖掉 → 二手出售
- 其他收入 → 其他收入

# 归属人推断
- 女儿/孩子/宝宝/闺女 → 女儿
- 老婆/媳妇/妻子 → 妻子
- 老公/丈夫/先生 → 丈夫
- 妈妈/爸爸/父亲/母亲 → 家人
- 无明确归属 → 本人

# 时间解析
- 今天 → {today.strftime('%Y-%m-%d')}
- 昨天 → {yesterday.strftime('%Y-%m-%d')}
- 前天 → {day_before_yesterday.strftime('%Y-%m-%d')}
- 这个月 → {current_month_start.strftime('%Y-%m-%d')} 至 {today.strftime('%Y-%m-%d')}
- 上个月 → 上个月第一天到最后一天
- X月X号 → 今年该日期

# 对话风格
- 简洁直接，善用 emoji（💰 收入 / 💸 支出 / 📊 汇总 / 🗑️ 删除 / ✅ 成功）
- 信息不完整时主动询问缺什么（金额？项目？类别？）
- 查询结果要有汇总（总收支、结余）+ 明细
- 财务状况要算净额：收入 - 支出 = 结余

# 重要
- 金额始终为正数
- 必须先调用工具获取真实数据，不能凭空编造
- 删除操作必须先查再删，不能直接猜 ID"""


# ============================================================
# 创建 LangChain Agent
# ============================================================
def create_expense_agent():
    """创建一个新的 agent 实例。如果 API Key 未配置，返回 None。"""
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
    """主页"""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """聊天接口 - 处理用户的自然语言输入"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请提供有效的JSON数据"}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    try:
        agent = create_expense_agent()
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
        return jsonify({"response": response_text, "session_id": session_id})

    except Exception as e:
        error_msg = str(e)
        print(f"❌ 错误: {error_msg}")

        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "401" in error_msg:
            friendly_msg = "⚠️ DeepSeek API Key 无效，请检查您的 API Key 是否正确。"
        elif "rate" in error_msg.lower() or "429" in error_msg:
            friendly_msg = "⚠️ API 请求过于频繁，请稍后再试。"
        elif "timeout" in error_msg.lower():
            friendly_msg = "⚠️ 请求超时，请稍后再试。"
        elif "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
            friendly_msg = "⚠️ API 额度不足，请检查您的账户余额。"
        else:
            friendly_msg = f"⚠️ 处理请求时出现错误，请稍后再试。\n\n错误详情：{error_msg}"

        return jsonify({"response": friendly_msg, "error": error_msg}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    api_status = "configured" if DEEPSEEK_API_KEY else "missing"
    return jsonify({
        "status": "ok",
        "api_key": api_status,
        "database": "connected",
    })


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """清空对话历史"""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    if session_id in _chat_histories:
        del _chat_histories[session_id]
    return jsonify({"status": "success", "message": "对话历史已清空"})


@app.route("/api/records", methods=["GET"])
def get_all_records():
    """获取所有记录"""
    type_filter = request.args.get("type", "")
    conn = get_db_connection()
    cursor = conn.cursor()

    if type_filter in ("expense", "income"):
        cursor.execute(
            "SELECT * FROM transactions WHERE type = ? ORDER BY date DESC, id DESC",
            (type_filter,),
        )
    else:
        cursor.execute("SELECT * FROM transactions ORDER BY date DESC, id DESC")

    rows = cursor.fetchall()
    conn.close()

    records = [_row_to_dict(row) for row in rows]
    return jsonify({"count": len(records), "records": records})


@app.route("/api/records/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    """删除指定记录（REST API）"""
    record = _get_record_by_id(record_id)
    if not record:
        return jsonify({"status": "error", "message": f"记录 {record_id} 不存在"}), 404

    _delete_by_id(record_id)
    return jsonify({"status": "success", "message": f"记录 {record_id} 已删除"})


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("📒 智能记账本 启动中...")
    print("   支持：支出 | 收入 | 查询 | 删除")
    print("=" * 60)
    print(f"🌐 访问地址: http://127.0.0.1:5000")
    print(f"🔑 API Key: {'✅ 已配置' if DEEPSEEK_API_KEY else '❌ 未配置'}")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
