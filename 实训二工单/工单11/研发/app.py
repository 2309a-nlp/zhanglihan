"""
医疗挂号管理 Agent - Streamlit 前端界面（流式输出）
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-挂号管理任务
"""

import sys
import os
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

import database
from agent import MedicalAgent

st.set_page_config(
    page_title="智能挂号助手",
    page_icon="🏥",
    layout="centered",
)

# 会话状态初始化
if "agent" not in st.session_state:
    database.init_db()
    database.seed_data()
    st.session_state.agent = MedicalAgent(db_path=database.DB_PATH)
    st.session_state.messages = []

with st.sidebar:
    st.header("🏥 智能挂号助手")
    st.caption("基于 DeepSeek LLM 的医疗挂号管理系统")
    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.agent.messages = [{"role": "system", "content": st.session_state.agent.messages[0]["content"] if st.session_state.agent.messages else ""}]
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**快捷功能**")
    quick_actions = {
        "📋 挂号咨询": "我想挂一个儿科的号",
        "🔍 医生查询": "帮我查下张建国医生的信息",
        "🚫 取消挂号": "取消我上周挂的消化内科普通号",
        "👦 家属挂号": "帮我大宝挂一个今天下午2点儿科专家的号",
        "📦 历史复诊": "我之前挂过眼科的专家，帮我再约一次",
        "🦷 号源查询": "牙科最近的号哪天的？",
    }
    for label, query in quick_actions.items():
        if st.button(label, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})

            # 显示状态折叠面板
            with st.status("🤔 正在分析您的需求...", expanded=True) as status:
                response_text = ""
                tool_events = []
                
                for event_type, data in st.session_state.agent.stream_chat(query):
                    if event_type == "status":
                        status.update(label=data)
                    elif event_type == "tool":
                        tool_events.append(data)
                        status.update(label=f"🔧 执行: {data}")
                    elif event_type == "tool_result":
                        status.update(label="✅ 数据获取成功")
                    elif event_type == "text":
                        response_text += data
                
                status.update(label="✅ 处理完成", state="complete")

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

# 主界面
st.title("🏥 医院智能挂号系统")
st.caption("支持语音意图识别、多患者管理、智能分诊与实时挂号")

# 显示历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("请输入您的挂号需求，例如：帮我挂一个儿科专家的号"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.status("🤔 正在分析您的需求...", expanded=True) as status:
            response_text = ""
            tool_events = []
            
            for event_type, data in st.session_state.agent.stream_chat(prompt):
                if event_type == "status":
                    status.update(label=data)
                elif event_type == "tool":
                    tool_events.append(data)
                    status.update(label=f"🔧 正在调用: {data}")
                elif event_type == "tool_result":
                    status.update(label="✅ 数据获取完成")
                elif event_type == "text":
                    response_text += data
            
            status.update(label="✅ 处理完成", state="complete")
        
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()

