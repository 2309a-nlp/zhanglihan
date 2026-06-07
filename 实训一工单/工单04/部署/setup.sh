#!/bin/bash
# ========================================
#  工单04 高稳定版PDF问答系统 - Linux部署
# ========================================

set -e

echo "=============================="
echo " 工单04 部署脚本 (Linux/WSL)"
echo "=============================="

# 1. 系统要求
echo ""
echo "[1/5] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "  [错误] 未找到 Python3"
    exit 1
fi
python3 --version

# 2. MySQL 检查
echo ""
echo "[2/5] 检查 MySQL..."
if command -v mysqladmin &> /dev/null; then
    mysqladmin ping -u root -p 2>/dev/null && echo "  MySQL: 运行中"
else
    echo "  [警告] 未检测到 mysqladmin"
fi

# 3. 创建 .api_key
echo ""
echo "[3/5] 配置 API Key..."
if [ ! -f ".api_key" ]; then
    echo "placeholder_api_key_here" > .api_key
    echo "  [信息] .api_key 模板已创建，请编辑填入实际密钥（Base64编码）"
fi

# 4. 安装依赖
echo ""
echo "[4/5] 安装 Python 依赖..."
pip3 install -r requirements.txt
echo "  [完成] 依赖安装完成"

# 5. 启动服务
echo ""
echo "[5/5] 启动 Streamlit 服务..."
echo ""
echo "  ========================================"
echo "  启动命令: streamlit run app.py --server.port 8503"
echo "  访问地址: http://localhost:8503"
echo "  ========================================"
echo ""

echo "正在启动..."
export="root"
streamlit run app.py --server.port 8503
