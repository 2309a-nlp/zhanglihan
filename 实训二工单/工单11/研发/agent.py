"""
医疗挂号管理 Agent - LLM 驱动版本（JSON 工具调用模式）
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-挂号管理任务
"""

import json
import os
import time
import re
from typing import Optional

from openai import OpenAI
import database

import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# 从 .env 读取 Key
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# ============================================================
# 工具定义 (Tools for LLM Function Calling)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": "根据姓名查询患者信息（包括本人及家属，如大宝、二宝等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "患者姓名，如 '张三', '大宝'"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_patient",
            "description": "新建患者档案（当挂号对象不在系统中时调用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "患者姓名"},
                    "sex": {"type": "string", "description": "性别，默认'男'"},
                    "birth": {"type": "string", "description": "出生日期，YYYY-MM-DD"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_department",
            "description": "根据科室名称查询科室信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "科室名称，如 '儿科', '牙科', '眼科'"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_doctors",
            "description": "查询医生信息，可按科室或姓名筛选",
            "parameters": {
                "type": "object",
                "properties": {
                    "dep_id": {"type": "integer", "description": "科室ID"},
                    "name": {"type": "string", "description": "医生姓名，如 '张建国'"},
                    "profession": {"type": "string", "description": "职称筛选，如 '专家', '普通'"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_registers",
            "description": "查询某患者的挂号记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "p_id": {"type": "integer", "description": "患者ID"},
                    "status": {"type": "integer", "description": "状态筛选：1-正常, 0-已取消"}
                },
                "required": ["p_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_register",
            "description": "创建挂号记录。注意：需提供 dep_id, p_id, reg_time, fee",
            "parameters": {
                "type": "object",
                "properties": {
                    "dep_id": {"type": "integer", "description": "科室ID"},
                    "p_id": {"type": "integer", "description": "患者ID"},
                    "reg_time": {"type": "string", "description": "挂号/就诊时间，YYYY-MM-DD HH:MM"},
                    "fee": {"type": "integer", "description": "挂号费，默认10"}
                },
                "required": ["dep_id", "p_id", "reg_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_register",
            "description": "取消某条挂号记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "reg_id": {"type": "integer", "description": "挂号记录ID"}
                },
                "required": ["reg_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_schedule",
            "description": "查询医生的坐诊时间安排。可按医生姓名、科室或日期查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "医生姓名，如 '张建国'"},
                    "department": {"type": "string", "description": "科室名称，如 '儿科', '牙科'"},
                    "date": {"type": "string", "description": "查询日期，YYYY-MM-DD 格式，如 '2026-06-27'。省略则查所有日期"}
                }
            }
        }
    }
]

# ============================================================
# 工具执行器
# ============================================================

def execute_tool(func_name: str, args: dict, db_path: str = None) -> str:
    """执行数据库工具函数"""
    try:
        if func_name == "get_patient_info":
            result = database.get_patient_by_name(args.get("name"), db_path)
        elif func_name == "create_patient":
            pid = database.create_patient(
                name=args.get("name"),
                sex=args.get("sex", "男"),
                birth=args.get("birth"),
                db_path=db_path
            )
            result = {"p_ID": pid, "p_Name": args.get("name")}
        elif func_name == "query_department":
            result = database.get_department_by_name(args.get("name"), db_path)
        elif func_name == "query_doctors":
            dep_id = args.get("dep_id")
            name = args.get("name")
            if name:
                result = database.get_doctor_by_name(name, db_path)
            elif dep_id:
                result = database.get_doctors_by_dept(dep_id, args.get("profession"), db_path)
            else:
                result = "请提供科室ID或医生姓名"
        elif func_name == "query_registers":
            result = database.get_registers_by_patient(args.get("p_id"), args.get("status"), db_path)
        elif func_name == "create_register":
            rid = database.create_register(
                dep_id=args.get("dep_id"),
                p_id=args.get("p_id"),
                reg_time=args.get("reg_time"),
                fee=args.get("fee", 10),
                db_path=db_path
            )
            result = {"reg_ID": rid, "status": "success"}
        elif func_name == "cancel_register":
            success = database.cancel_register(args.get("reg_id"), db_path)
            result = {"reg_ID": args.get("reg_id"), "cancelled": success}
        elif func_name == "query_schedule":
            doctor_name = args.get("doctor_name")
            department = args.get("department")
            date = args.get("date")
            if doctor_name:
                doc = database.get_doctor_by_name(doctor_name, db_path)
                if doc:
                    result = database.get_schedule_by_doctor(doc["d_ID"], db_path)
                else:
                    result = {"error": f"未找到医生: {doctor_name}"}
            elif department:
                dept = database.get_department_by_name(department, db_path)
                if dept:
                    result = database.get_schedule_by_department(dept["dep_ID"], db_path)
                else:
                    result = {"error": f"未找到科室: {department}"}
            else:
                result = {"error": "请提供医生姓名或科室名称"}
        else:
            result = f"Unknown tool: {func_name}"
        
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """你是医院智能挂号助手。请严格使用提供的工具完成用户的挂号、查询、取消等需求。
【核心规则】
1. 必须使用工具：任何数据查询或写入操作都必须调用对应工具，禁止凭空编造。
2. 身份识别：当用户提到"大宝"、"二宝"等家属时，先调用 get_patient_info 查找该患者。若不存在，需调用 create_patient 建档。
3. 科室匹配：用户说"儿科专家"时，先 query_department 找科室，再 query_doctors 筛选 profession='专家' 的医生。
4. 挂号逻辑：create_register 需要 dep_id, p_id, reg_time。成功后返回 reg_ID 和次序。
5. 取消逻辑：先 query_registers 找到对应记录，获取 reg_ID，再调用 cancel_register。
6. 输出格式：工具调用后，用简洁的中文向用户汇报结果。
7. 坐诊时间：当用户询问某医生的坐诊时间、哪天出诊、什么时候有号时，使用 query_schedule 工具。可按医生姓名或科室查询。
"""


class MedicalAgent:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or database.DB_PATH
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
        )
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        max_iterations = 5
        final_reply = ""

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )
            choice = response.choices[0]
            message = choice.message

            if choice.finish_reason == "tool_calls":
                tool_calls = message.tool_calls
                self.messages.append(message)  # 保留 assistant 的工具调用消息

                for tc in tool_calls:
                    func_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"  [Tool Call] {func_name}({args})")
                    result = execute_tool(func_name, args, self.db_path)
                    print(f"  [Tool Result] {result[:100]}...")
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                final_reply = message.content or ""
                self.messages.append({"role": "assistant", "content": final_reply})
                break

        return final_reply

    def stream_chat(self, user_input: str):
        """流式生成器：yield (event_type, data)"""
        self.messages.append({"role": "user", "content": user_input})
        
        # 1. 思考中
        yield ("status", "正在理解您的需求...")
        
        max_iterations = 5
        for _ in range(max_iterations):
            yield ("status", "正在查询号源数据...")
            
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                stream=True,
            )

            tool_calls_buffer = []
            current_text = ""
            is_tool_call = False

            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                
                if delta.tool_calls:
                    is_tool_call = True
                    for tc in delta.tool_calls:
                        tool_calls_buffer.append(tc)
                
                if delta.content:
                    current_text += delta.content
                    yield ("text", delta.content)

            if is_tool_call:
                # 解析工具调用
                # 由于流式返回的 tool_calls 是分片的，需要合并
                # 这里简化处理：重新发一次非流式请求获取完整 tool_calls，或从 buffer 拼凑
                # 为了稳定性，我们直接用非流式获取 tool_calls，然后流式生成最终回复
                yield ("status", "正在处理挂号请求...")
                
                # 重新获取完整 tool_calls (非流式)
                full_response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                )
                
                msg = full_response.choices[0].message
                self.messages.append(msg)
                
                for tc in msg.tool_calls or []:
                    func_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    yield ("tool", f"调用 {func_name}...")
                    result = execute_tool(func_name, args, self.db_path)
                    yield ("tool_result", result[:150])
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                # 最终回复
                if current_text:
                    self.messages.append({"role": "assistant", "content": current_text})
                break
        
        yield ("done", "")


if __name__ == "__main__":
    database.init_db()
    database.seed_data()
    agent = MedicalAgent()
    
    test_cases = [
        "帮我查下张建国医生下周的坐诊时间",
        "帮我大宝挂一个今天下午2点儿科专家的号",
        "牙科最近的号哪天的？",
    ]
    
    for q in test_cases:
        print(f"\nQ: {q}")
        reply = agent.chat(q)
        print(f"A: {reply}")

