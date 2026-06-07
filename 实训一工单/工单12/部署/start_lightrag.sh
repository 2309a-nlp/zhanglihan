#!/bin/bash
# ============================================================
# LightRAG 一键部署启动脚本
# 环境: WSL2 (Ubuntu) + uv tool + DeepSeek API + Ollama (Win)
# 用途: 启动 LightRAG 服务器，自动检测网关 IP
# ============================================================
set -e

# ── 颜色输出 ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()   { echo -e "${RED}[ERR]${NC}  $1"; }

echo ""
echo "======================================"
echo "  LightRAG 部署启动脚本"
echo "======================================"
echo ""

# ── 1. 加载环境变量 ──
ENV_FILE="$HOME/.lightrag_env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    log_ok "加载环境变量: $ENV_FILE"

    # 检查必要变量
    if [ -z "$OPENAI_API_KEY" ]; then
        log_warn "OPENAI_API_KEY 未设置"
    else
        log_info "OPENAI_API_KEY = ${OPENAI_API_KEY:0:12}..."
    fi
    if [ -z "$OPENAI_API_BASE" ]; then
        log_warn "OPENAI_API_BASE 未设置，使用默认值"
        export OPENAI_API_BASE="https://api.deepseek.com/v1"
    else
        log_info "OPENAI_API_BASE = $OPENAI_API_BASE"
    fi
else
    log_warn "未找到 $ENV_FILE"
    log_info "请创建该文件并写入:"
    echo '  OPENAI_API_KEY="sk-your-key-here"'
    echo '  OPENAI_API_BASE="https://api.deepseek.com/v1"'
    exit 1
fi

# ── 2. 检查 lightrag-server 是否可用 ──
if ! command -v lightrag-server &> /dev/null; then
    log_err "未找到 lightrag-server 命令"
    log_info "请先安装: uv tool install lightrag-hku --python python3.13"
    exit 1
fi
log_ok "lightrag-server 可用"

# ── 3. 动态获取 WSL 网关 IP ──
GATEWAY_IP=$(ip route show default | awk '{print $3}')
if [ -z "$GATEWAY_IP" ]; then
    log_err "无法获取 WSL 网关 IP"
    log_info "请手动设置 OLLAMA_HOST"
    exit 1
fi
OLLAMA_HOST="http://${GATEWAY_IP}:11434"
log_info "WSL 网关 IP: $GATEWAY_IP"

# ── 4. 检查 Windows Ollama 是否可达 ──
log_info "检查 Ollama 连接..."
if curl -s --connect-timeout 3 "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    log_ok "Ollama 可达 ($OLLAMA_HOST)"

    # 检查嵌入端点
    EMBED_TEST=$(curl -s --connect-timeout 5 \
        "$OLLAMA_HOST/api/embed" \
        -d '{"model":"qwen2.5:1.5b","input":"hello"}' 2>&1)

    if echo "$EMBED_TEST" | grep -q "not support\|error\|refused"; then
        log_warn "Ollama 嵌入功能未启用！"
        echo ""
        echo "  请按以下步骤操作:"
        echo "  1. 关闭 Windows 上当前的 ollama serve 窗口"
        echo "  2. 打开新的 CMD 窗口"
        echo "  3. 执行: set OLLAMA_EMBEDDINGS=1"
        echo "  4. 执行: ollama serve"
        echo "  5. 重新运行本脚本"
        echo ""
        echo "  或者创建快捷方式:"
        echo "  桌面新建 ollama-with-embed.bat，内容:"
        echo "  ┌─────────────────────────────────────┐"
        echo "  │ @echo off                           │"
        echo "  │ set OLLAMA_EMBEDDINGS=1             │"
        echo "  │ ollama serve                        │"
        echo "  └─────────────────────────────────────┘"
        echo ""
    else
        log_ok "Ollama 嵌入功能正常"
    fi
else
    log_warn "Ollama 不可达 ($OLLAMA_HOST)"
    log_info "请确保 Windows 上 Ollama 已启动"
fi

# ── 5. 创建数据目录 ──
mkdir -p "$HOME/rag_storage"
mkdir -p "$HOME/inputs"
log_ok "数据目录就绪"

# ── 6. 启动 LightRAG 服务器 ──
echo ""
echo "======================================"
echo -e "  ${GREEN}启动 LightRAG Server...${NC}"
echo "======================================"
echo ""
log_info "LLM: DeepSeek (deepseek-chat)"
log_info "Embedding: Ollama (qwen2.5:1.5b)"
log_info "Ollama Host: $OLLAMA_HOST"
log_info "数据目录: $HOME/rag_storage"
log_info "端口: 9621"
echo ""

exec lightrag-server \
    --host 0.0.0.0 \
    --port 9621 \
    --llm-binding openai \
    --llm-model deepseek-chat \
    --embedding-binding ollama \
    --ollama-host "$OLLAMA_HOST" \
    --embedding-model qwen2.5:1.5b \
    --working-dir "$HOME/rag_storage"
