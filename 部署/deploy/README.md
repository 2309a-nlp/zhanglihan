# Deploy - 部署脚本目录

本目录包含 Expert RAG System 的部署脚本和配置模板。

## 文件说明

| 文件 | 说明 | 适用平台 |
|------|------|----------|
| `deploy.sh` | 一键部署脚本（完整安装） | Linux / WSL2 |
| `deploy.ps1` | Windows PowerShell 部署脚本 | Windows (PowerShell) |
| `start.sh` | 启动服务（由 deploy.sh 自动生成） | Linux / WSL2 |
| `start.ps1` | 启动服务（由 deploy.ps1 自动生成） | Windows (PowerShell) |
| `start.bat` | 双击启动（由 deploy.ps1 自动生成） | Windows |
| `stop.sh` | 停止服务（由 deploy.sh 自动生成） | Linux / WSL2 |
| `.env.example` | 环境变量模板 | — |

## 快速开始

### Linux / WSL2

```bash
cd deploy/
chmod +x deploy.sh
./deploy.sh
```

部署完成后：

```bash
# 启动服务
bash start.sh

# 停止服务
bash stop.sh
```

### Windows (PowerShell)

```powershell
cd deploy
.\deploy.ps1
```

部署完成后双击 `start.bat` 或运行 `.\start.ps1`。

### Windows (CMD)

```cmd
cd deploy
powershell -ExecutionPolicy Bypass -File deploy.ps1
```

## 部署选项

### deploy.sh

| 选项 | 说明 |
|------|------|
| `--skip-milvus` | 跳过 Milvus Docker 启动 |
| `--skip-mysql` | 跳过 MySQL Docker 启动 |
| `--skip-frontend` | 跳过前端构建 |
| `--skip-models` | 跳过模型下载 |
| `--skip-index` | 跳过索引构建 |
| `--force / -f` | 覆盖已有配置 |

示例：

```bash
# 仅安装 Python 依赖 + 前端构建（快速部署）
./deploy.sh --skip-milvus --skip-mysql --skip-models --skip-index

# 完整重新部署
./deploy.sh -f
```

### deploy.ps1

| 参数 | 说明 |
|------|------|
| `-SkipFrontend` | 跳过前端构建 |
| `-SkipMilvus` | 跳过 Milvus |
| `-SkipMysql` | 跳过 MySQL |
| `-SkipModels` | 跳过模型下载 |
| `-SkipIndex` | 跳过索引构建 |

## 前置依赖

- **Python 3.10+**（必需）
- **Node.js 18+**（构建前端时需要）
- **Docker**（启动 Milvus / MySQL 时需要，可选）
- **WSL2 + Ubuntu**（Windows 部署时需要）

## 架构

```
用户浏览器 ──HTTP──> FastAPI (8001) ────> DeepSeek API (LLM)
                      │
                      ├──> bge-m3 (本地嵌入)
                      ├──> bge-reranker (本地重排序)
                      ├──> FAISS + BM25 (本地向量库)
                      ├──> Milvus (对话历史，可选)
                      └──> MySQL (长期记忆，可选)
```
