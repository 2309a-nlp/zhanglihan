# 📅 日程提醒智能体

一个基于 Flask + LangChain + DeepSeek API 的日程管理智能体，通过自然语言对话管理日程安排。

## 功能

- **📋 查询日程** - "我今天的日程有哪些？"
- **➕ 添加日程** - "提醒我买咖啡"
- **❌ 取消日程** - "取消日程1"
- **⏰ 到时提醒** - 到时间自动弹窗提醒

## 快速开始

### 1. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-your_key_here
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
python app.py
```

访问 http://127.0.0.1:5000

## 技术栈

- **后端**: Flask + LangChain Agent
- **AI**: DeepSeek API (deepseek-chat)
- **数据库**: SQLite
- **前端**: 原生 HTML/CSS/JS

## 项目结构

```
工单02/
├── app.py              # Flask 主程序
├── templates/index.html # 聊天界面
├── static/style.css     # 样式
├── static/script.js     # 前端逻辑
├── .env                # API Key（不提交）
├── .env.example        # 环境变量示例
├── .gitignore          # Git 忽略规则
└── requirements.txt    # 依赖
```
