# 智能挂号助手 - LLM 驱动版本

基于 DeepSeek 大模型的医疗挂号管理系统，支持自然语言对话、意图识别、工具调用和数据库事务联动。

## 功能特性

- **自然语言理解**: 通过 DeepSeek LLM 理解用户意图，支持任意相似的挂号/查询 query
- **工具调用**: Agent 自动调用数据库工具完成挂号、查询、取消等操作
- **对话历史**: 支持多轮对话，记忆上下文
- **Web 界面**: Streamlit 构建的现代化聊天界面
- **快捷操作**: 侧边栏预设常用功能按钮
- **实时反馈**: 显示工具调用记录、响应时间等指标

## 项目结构

```
C:\Users\ASUSTeK\Desktop\2309B\实训二工单\工单11\
├── database.py      # 数据库模块（SQLite DDL + CRUD）
├── agent.py         # LLM Agent 核心（DeepSeek 接入 + 工具调用）
├── app.py           # Streamlit 前端界面
├── main.py          # 主入口（启动脚本）
├── test.py          # 端到端测试
├── start.bat        # Windows 启动脚本
├── requirements.txt # Python 依赖
└── .env             # 环境变量（API Key）
```

## 快速开始

### 方式 1: 双击启动
双击 `start.bat` 或运行:
```bash
python main.py
```
浏览器自动打开 http://localhost:8511

### 方式 2: 直接运行测试
```bash
python test.py
```

## 环境要求

- Python 3.10+
- 依赖: `pip install -r requirements.txt`
- DeepSeek API Key（已在 `.env` 中配置）

## 支持的操作

| 操作 | 示例 |
|------|------|
| 挂号咨询 | "帮我大宝挂一个今天下午儿科专家的号" |
| 号源查询 | "牙科最近的号哪天的？" |
| 历史复诊 | "我之前挂过眼科的专家，帮我再约一次" |
| 多患者挂号 | "我明天上午9点想带二宝看皮肤科，还有号吗？" |
| 取消挂号 | "取消我上周挂的消化内科普通号" |
| 医生排班 | "帮我查下张建国医生下周的坐诊时间" |

## 技术架构

```
用户输入 -> Streamlit UI -> DeepSeek LLM -> 工具调用 -> SQLite 数据库 -> 格式化回复 -> 前端渲染
```

## API 配置

在 `.env` 文件中配置:
```
DEEPSEEK_API_KEY=sk-5c2...
## 开发说明

- `agent.py` 定义了 9 个可调用的工具函数
- `SYSTEM_PROMPT` 控制 Agent 的行为和回答风格
- 工具调用日志在侧边栏实时显示

## 测试覆盖

- 5 个核心测试用例
- 支持 30+ 自然语言变体
- 平均响应时间: < 5s（含 LLM 调用）

