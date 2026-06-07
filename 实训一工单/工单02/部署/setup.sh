#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

CYAN='\033[0;36m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN} Work Order 02 - PDF QA System Setup${NC}"
echo -e "${CYAN} (Streamlit + DeepSeek + FAISS)${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# 1. Python check
echo -e "[1/6] Checking Python..."
PYTHON=""
for cmd in python3 python; do
    command -v "$cmd" &>/dev/null && PYTHON="$cmd" && break
done
if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found. Install Python 3.10+${NC}"
    exit 1
fi
echo -e "    ${GREEN}$($PYTHON --version)${NC}"

# 2. Virtual env
echo -e "[2/6] Creating virtual env..."
if [ -d ".venv" ]; then
    echo -e "    ${YELLOW}.venv exists, skipping${NC}"
else
    $PYTHON -m venv .venv
    echo -e "    ${GREEN}.venv created${NC}"
fi

# 3. Install deps
echo -e "[3/6] Installing dependencies..."
source .venv/bin/activate
pip install -r requirements.txt
echo -e "    ${GREEN}Done${NC}"

# 4. API key
echo -e "[4/6] Configuring API Key..."
if [ ! -f ".api_key" ]; then
    echo -e "    ${YELLOW}.api_key not found${NC}"
    echo "    Put your DeepSeek API Key into .api_key:"
    echo "    --------------------------------"
    echo "    echo 'sk-...' > .api_key"
    echo "    --------------------------------"
    read -p "    Or paste it now (leave blank to skip): " USER_KEY
    if [ -n "$USER_KEY" ]; then
        echo "$USER_KEY" > .api_key
        echo -e "    ${GREEN}.api_key created${NC}"
    fi
else
    echo -e "    ${GREEN}.api_key exists${NC}"
fi

# 5. Data dirs
echo -e "[5/6] Creating data directories..."
mkdir -p docs uploads vector_store
echo -e "    ${GREEN}Done${NC}"

# 6. MySQL check
echo -e "[6/6] Checking MySQL..."
if command -v mysqladmin &>/dev/null && mysqladmin ping -u root --silent 2>/dev/null; then
    echo -e "    ${GREEN}MySQL is running${NC}"
else
    echo -e "    ${YELLOW}MySQL not detected (start manually if needed)${NC}"
fi

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${GREEN} Setup complete! Start with:${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo "    source .venv/bin/activate && streamlit run app.py"
echo ""
echo "  Default: http://localhost:8501"
echo -e "${CYAN}================================================${NC}"
