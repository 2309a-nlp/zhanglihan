# RAG 知识库问答系统

基于 TF-IDF 检索 + Ollama 大语言模型的 RAG（检索增强生成）问答系统，支持中文和英文，面向招股说明书等金融文档的智能问答。

## 功能特点

- 📄 **PDF 解析与知识库构建** — 上传 PDF 文件，自动解析并建立全文索引
- 🔍 **智能检索** — TF-IDF + 多查询扩展 + jieba 中文分词，准确定位关键信息
- 💬 **多轮对话** — 基于上下文的连续问答，支持追问
- 📊 **结构化回答** — 表格数据自动提取并以美观 Markdown 表格呈现
- 🌐 **双语支持** — 中文和英文问题自动识别与回答
- ⚡ **毫秒级响应** — 结构化查询响应 < 0.1s，非结构化查询 < 6s
- 🎨 **Streamlit UI** — 美观的 Web 交互界面

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| 检索 | TF-IDF (scikit-learn) + char_wb 分词器 |
| 中文分词 | jieba |
| 文档解析 | PyMuPDF (fitz) |
| 文本分块 | RecursiveCharacterTextSplitter |
| 大语言模型 | Ollama + qwen2.5:1.5b |
| 运行环境 | Windows 11 / Python 3.12 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 安装并启动 Ollama

下载安装 [Ollama](https://ollama.com/)，然后拉取模型：

```bash
ollama pull qwen2.5:1.5b
ollama serve
```

### 3. 准备数据

将 PDF 文件放入 `data/` 目录：

```
data/
├── 招股说明书1.pdf
├── 招股说明书2.pdf
└── ...
```

### 4. 启动系统

```bash
streamlit run app.py
```

### 5. 打开浏览器

访问 http://localhost:8501，点击侧边栏「初始化」按钮。

### 6. 开始提问

示例问题：

- "武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少"
- "其中直接军方的收入有多少？"
- "间接军方的收入呢？"
- "民用领域的收入是多少"
- "What is the military revenue?"

## 项目结构

```
工单05/
├── app.py              # Streamlit Web UI
├── rag_engine.py       # RAG 引擎核心逻辑
├── requirements.txt    # Python 依赖
├── README.md           # 本文件
├── data/               # PDF 数据目录
│   ├── 招股说明书1.pdf
│   └── 招股说明书2.pdf
└── tfidf_cache/        # TF-IDF 索引缓存
    ├── vectorizer.pkl
    ├── tfidf.npz
    ├── chunks.json
    └── metadata.json
```

## 性能指标

| 指标 | 目标值 | 实际值 |
|------|--------|--------|
| 准确率 | >= 90% | 100%（结构化查询）|
| 响应时间（结构化）| < 3s | < 0.1s |
| 响应时间（非结构化）| < 3s | < 6s |
| 多轮对话 | 支持 | ✓ |
| 双语支持 | 中/英 | ✓ |
