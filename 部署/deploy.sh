#!/usr/bin/env bash
# =====================================================
# Expert RAG System - 一键部署脚本（Linux / WSL2）
# 用法：
#   chmod +x deploy.sh
#   ./deploy.sh                    # 完整部署
#   ./deploy.sh --skip-milvus      # 跳过 Milvus
#   ./deploy.sh --skip-mysql       # 跳过 MySQL
#   ./deploy.sh --skip-frontend    # 跳过前端构建
#   ./deploy.sh --help             # 查看所有选项
# =====================================================
set -euo pipefail

# ---------- 颜色 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
log()  { echo -e "${CYAN}[部署]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }

# ---------- 参数解析 ----------
SKIP_MILVUS=false
SKIP_MYSQL=false
SKIP_FRONTEND=false
SKIP_MODELS=false
SKIP_INDEX=false
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-milvus)   SKIP_MILVUS=true    ; shift ;;
    --skip-mysql)    SKIP_MYSQL=true     ; shift ;;
    --skip-frontend) SKIP_FRONTEND=true  ; shift ;;
    --skip-models)   SKIP_MODELS=true    ; shift ;;
    --skip-index)    SKIP_INDEX=true     ; shift ;;
    --force|-f)      FORCE=true          ; shift ;;
    --help|-h)
      echo "用法: $0 [选项]"
      echo "  --skip-milvus      跳过 Milvus Docker 启动"
      echo "  --skip-mysql       跳过 MySQL Docker 启动"
      echo "  --skip-frontend    跳过前端构建（使用已有 dist）"
      echo "  --skip-models      跳过模型下载（bge-m3, bge-reranker）"
      echo "  --skip-index       跳过索引构建"
      echo "  --force / -f       覆盖已有配置"
      exit 0
      ;;
    *) err "未知参数: $1"; exit 1 ;;
  esac
shift
done

# ---------- 项目路径 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
MODEL_DIR="$BACKEND_DIR/bge-m3"
RERANKER_CACHE="$HOME/.cache/huggingface/hub"
VENV_DIR="$PROJECT_DIR/.venv"

log "项目目录: $PROJECT_DIR"

# =============================================
# 第 1 步：检查系统依赖
# =============================================
check_prerequisites() {
  log "检查系统依赖..."

  # Python 3
  if command -v python3 &>/dev/null; then
    local pyver
    pyver=$(python3 --version 2>&1)
    ok "Python: $pyver"
  else
    err "未找到 python3，请安装 Python 3.10+"
    exit 1
  fi

  # Node.js
  if command -v node &>/dev/null; then
    local nodever
    nodever=$(node --version 2>&1)
    ok "Node.js: $nodever"
  else
    warn "未找到 Node.js（如不构建前端可忽略）"
  fi

  # npm
  if command -v npm &>/dev/null; then
    ok "npm: $(npm --version)"
  fi

  # Docker
  if command -v docker &>/dev/null; then
    ok "Docker: $(docker --version 2>&1)"
  else
    warn "未找到 Docker（Milvus/MySQL 需要 Docker）"
  fi

  # unzip（解压模型用）
  if command -v unzip &>/dev/null; then
    ok "unzip 已安装"
  fi
}

# =============================================
# 第 2 步：创建 Python 虚拟环境 + 安装依赖
# =============================================
setup_python_env() {
  log "配置 Python 虚拟环境..."

  if [ ! -d "$VENV_DIR" ] || [ "$FORCE" = true ]; then
    python3 -m venv "$VENV_DIR"
    ok "虚拟环境已创建: $VENV_DIR"
  else
    ok "虚拟环境已存在，跳过"
  fi

  source "$VENV_DIR/bin/activate"

  # 升级 pip
  pip install --upgrade pip -q

  log "安装 Python 依赖..."
  pip install -r "$PROJECT_DIR/requirements.txt" -q

  ok "Python 依赖安装完成"
}

# =============================================
# 第 3 步：安装前端依赖 + 构建
# =============================================
build_frontend() {
  if [ "$SKIP_FRONTEND" = true ]; then
    warn "跳过前端构建"
    return
  fi

  local frontend_dir="$PROJECT_DIR/frontend"
  if [ ! -d "$frontend_dir/node_modules" ] || [ "$FORCE" = true ]; then
    log "安装前端依赖..."
    cd "$frontend_dir"
    npm install
    ok "前端依赖安装完成"
  else
    ok "前端 node_modules 已存在"
  fi

  log "构建前端（输出到 frontend/dist 和 backend/dist）..."
  cd "$frontend_dir"
  npm run build

  # 复制到 backend/dist（FastAPI 静态文件挂载点）
  if [ -d "$BACKEND_DIR/dist" ]; then
    rm -rf "$BACKEND_DIR/dist"
  fi
  cp -r "$frontend_dir/dist" "$BACKEND_DIR/dist"
  ok "前端构建完成，已复制到 backend/dist"
}

# =============================================
# 第 4 步：配置 .env
# =============================================
setup_env() {
  local env_file="$BACKEND_DIR/.env"

  if [ -f "$env_file" ] && [ "$FORCE" != true ]; then
    ok ".env 已存在，跳过（用 --force 覆盖）"
    return
  fi

  if [ -f "$SCRIPT_DIR/.env.example" ]; then
    cp "$SCRIPT_DIR/.env.example" "$env_file"
    ok "已从模板创建 .env，请编辑 $env_file 填入 DEEPSEEK_API_KEY"
    warn ">>> 注意：请编辑 $env_file 填入你的 DeepSeek API Key <<<"
  else
    warn "未找到 .env.example，跳过"
  fi
}

# =============================================
# 第 5 步：下载本地模型
# =============================================
download_models() {
  if [ "$SKIP_MODELS" = true ]; then
    warn "跳过模型下载"
    return
  fi

  source "$VENV_DIR/bin/activate"

  # bge-m3 嵌入模型
  if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    ok "bge-m3 模型已存在"
  else
    log "下载 BAAI/bge-m3 嵌入模型（约 2.2GB，首次需等待）..."
    mkdir -p "$MODEL_DIR"
    python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3', cache_folder='$MODEL_DIR')
model.save('$MODEL_DIR')
print('bge-m3 下载完成')
" 2>&1 | tail -5
    ok "bge-m3 模型下载完成"
  fi

  # bge-reranker-v2-m3 重排序模型
  log "缓存 BAAI/bge-reranker-v2-m3 重排序模型（首次运行时会自动下载）..."
  python3 -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3', cache_dir='$RERANKER_CACHE')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', cache_dir='$RERANKER_CACHE')
print('bge-reranker 下载完成')
" 2>&1 | tail -5
  ok "重排序模型已缓存"
}

# =============================================
# 第 6 步：构建知识库索引
# =============================================
build_index() {
  if [ "$SKIP_INDEX" = true ]; then
    warn "跳过索引构建"
    return
  fi

  source "$VENV_DIR/bin/activate"

  # 检查是否已有索引
  local roles_dir="$BACKEND_DIR/vector_db/roles"
  local has_index=false
  for role in Medical Finance Law Education Psychology; do
    if [ -f "$roles_dir/$role/faiss_index.pkl" ]; then
      has_index=true
      break
    fi
  done

  if [ "$has_index" = true ] && [ "$FORCE" != true ]; then
    ok "知识库索引已存在，跳过（用 --force 重建）"
    return
  fi

  log "构建 FAISS + BM25 混合索引（从 data/ 目录读取文档）..."
  cd "$BACKEND_DIR"

  if [ -f "vector_db/build_index_v2.py" ]; then
    python3 vector_db/build_index_v2.py
    ok "索引构建完成"
  else
    warn "未找到 build_index_v2.py，跳过索引构建"
  fi
}

# =============================================
# 第 7 步：启动 Milvus（Docker）
# =============================================
start_milvus() {
  if [ "$SKIP_MILVUS" = true ]; then
    warn "跳过 Milvus 启动"
    return
  fi

  if ! command -v docker &>/dev/null; then
    warn "Docker 未安装，跳过 Milvus"
    return
  fi

  # 检查 Milvus 是否已在运行
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "milvus"; then
    ok "Milvus 已在运行"
    return
  fi

  log "启动 Milvus（Docker 单机模式）..."
  docker run -d \
    --name milvus-standalone \
    -p 19530:19530 \
    -p 9091:9091 \
    -v milvus-data:/var/lib/milvus \
    milvusdb/milvus:latest

  # 等待启动
  log "等待 Milvus 就绪..."
  for i in $(seq 1 30); do
    if curl -s http://localhost:9091/health 2>/dev/null | grep -q "OK"; then
      ok "Milvus 就绪（端口 19530）"
      return
    fi
    sleep 2
  done
  warn "Milvus 启动超时，请手动检查 docker logs milvus-standalone"
}

# =============================================
# 第 8 步：启动 MySQL（Docker）
# =============================================
start_mysql() {
  if [ "$SKIP_MYSQL" = true ]; then
    warn "跳过 MySQL 启动"
    return
  fi

  if ! command -v docker &>/dev/null; then
    warn "Docker 未安装，跳过 MySQL"
    return
  fi

  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "mysql-rag"; then
    ok "MySQL 已在运行"
    return
  fi

  log "启动 MySQL（Docker）..."
  docker run -d \
    --name mysql-rag \
    -p 3306:3306 \
    -e MYSQL_ROOT_PASSWORD=root \
    -e MYSQL_DATABASE=rag \
    -v mysql-rag-data:/var/lib/mysql \
    mysql:8.0

  # 等待启动
  log "等待 MySQL 就绪..."
  for i in $(seq 1 20); do
    if docker exec mysql-rag mysqladmin ping -uroot -proot --silent 2>/dev/null; then
      ok "MySQL 就绪（端口 3306）"

      # 初始化表结构
      log "初始化长期记忆表..."
      docker exec -i mysql-rag mysql -uroot -proot rag <<'SQL'
CREATE TABLE IF NOT EXISTS user_memory (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  memory_key VARCHAR(255) NOT NULL,
  memory_value TEXT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_user_key (user_id, memory_key)
) DEFAULT CHARSET=utf8mb4;
SQL
      ok "MySQL 表结构初始化完成"
      return
    fi
    sleep 3
  done
  warn "MySQL 启动超时，请手动检查 docker logs mysql-rag"
}

# =============================================
# 第 9 步：创建启动/停止快捷脚本
# =============================================
create_scripts() {
  cat > "$SCRIPT_DIR/start.sh" <<'SHELL'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
BACKEND_DIR="$PROJECT_DIR/backend"

cd "$BACKEND_DIR"
source "$VENV_DIR/bin/activate"

echo "[启动] Expert RAG System..."
echo "[启动] 后端 => http://127.0.0.1:8001"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python3 main.py
SHELL
  chmod +x "$SCRIPT_DIR/start.sh"
  ok "启动脚本已创建: deploy/start.sh"

  cat > "$SCRIPT_DIR/stop.sh" <<'SHELL'
#!/usr/bin/env bash
echo "[停止] 关闭 Expert RAG System..."
# 停止后端进程
pkill -f "uvicorn.*main" 2>/dev/null && echo "后端已停止" || echo "后端未运行"
# 可选：停止 Docker 容器
# docker stop milvus-standalone mysql-rag 2>/dev/null
echo "完成"
SHELL
  chmod +x "$SCRIPT_DIR/stop.sh"
  ok "停止脚本已创建: deploy/stop.sh"
}

# =============================================
# 第 10 步：验证部署
# =============================================
verify() {
  log "验证部署..."

  local ok_count=0
  local total=6

  # 1. Python 依赖
  source "$VENV_DIR/bin/activate"
  if python3 -c "import fastapi; import uvicorn; import langchain_core" 2>/dev/null; then
    ok "[1/$total] Python 依赖正常"
    ((ok_count++))
  else
    err "[1/$total] Python 依赖异常"
  fi

  # 2. 前端构建
  if [ -f "$BACKEND_DIR/dist/index.html" ]; then
    ok "[2/$total] 前端构建存在"
    ((ok_count++))
  else
    warn "[2/$total] 前端构建不存在"
  fi

  # 3. 模型
  if [ -f "$MODEL_DIR/config.json" ]; then
    ok "[3/$total] bge-m3 模型存在"
    ((ok_count++))
  else
    warn "[3/$total] bge-m3 模型未下载"
  fi

  # 4. 索引
  if [ -d "$BACKEND_DIR/vector_db/roles/Medical" ] && ls "$BACKEND_DIR/vector_db/roles/Medical/"*.pkl &>/dev/null 2>&1; then
    ok "[4/$total] 知识库索引存在"
    ((ok_count++))
  else
    warn "[4/$total] 知识库索引未构建"
  fi

  # 5. .env
  if [ -f "$BACKEND_DIR/.env" ]; then
    ok "[5/$total] .env 配置存在"
    ((ok_count++))
  else
    warn "[5/$total] .env 未配置"
  fi

  # 6. Python 版本
  local pyver
  pyver=$(python3 --version 2>&1)
  ok "[6/$total] $pyver"
  ((ok_count++))

  echo ""
  if [ "$ok_count" -eq "$total" ]; then
    ok "全部 $total/$total 验证通过！"
  else
    warn "验证 $ok_count/$total 通过，请检查上面的警告"
  fi
}

# =============================================
# 打印部署总结
# =============================================
print_summary() {
  echo ""
  echo "=============================================="
  echo "  Expert RAG System - 部署完成"
  echo "=============================================="
  echo ""
  echo "  启动服务:"
  echo "    cd $(dirname "$PROJECT_DIR")/deploy && bash start.sh"
  echo ""
  echo "  访问地址:"
  echo "    http://127.0.0.1:8001"
  echo ""
  echo "  API 文档:"
  echo "    http://127.0.0.1:8001/docs"
  echo ""
  echo "  管理命令:"
  echo "    bash deploy/stop.sh    停止服务"
  echo "    bash deploy/start.sh   启动服务"
  echo ""
  echo "  可选服务（Docker）:"
  echo "    docker start milvus-standalone    启动 Milvus"
  echo "    docker start mysql-rag            启动 MySQL"
  echo ""
  echo "  配置文件:"
  echo "    $BACKEND_DIR/.env"
  echo "=============================================="
}

# =============================================
# 主流程
# =============================================
main() {
  echo ""
  echo "=============================================="
  echo "  Expert RAG System - 一键部署"
  echo "=============================================="
  echo ""

  check_prerequisites
  setup_python_env
  setup_env
  build_frontend
  download_models
  build_index
  start_milvus
  start_mysql
  create_scripts
  verify
  print_summary
}

main
