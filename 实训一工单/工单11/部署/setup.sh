#!/bin/bash
# 工单11 - BGE模型微调 环境部署脚本 (Linux/macOS)
# BGE-base + Matryoshka 医疗问答检索模型

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " 工单11 环境部署脚本 (Linux/macOS)"
echo " BGE-base + Matryoshka 医疗问答检索模型"
echo "============================================"
echo ""

# 1. 检查 Python
echo "[1/5] 检查 Python 环境..."
python3 --version || python --version
PYTHON=$(command -v python3 || command -v python)
echo "    OK - 使用 $PYTHON"

# 2. 创建虚拟环境
echo "[2/5] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "    虚拟环境已创建"
else
    echo "    虚拟环境已存在，跳过"
fi

# 3. 安装依赖
echo "[3/5] 安装 Python 依赖..."
source venv/bin/activate
pip install -r requirements.txt || {
    echo "    [警告] 部分依赖安装失败，尝试分步安装..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers datasets pandas
}
echo "    依赖安装完成"

# 4. 检查模型文件
echo "[4/5] 检查模型文件..."
MODEL_DIR="$PROJECT_DIR/bge-finetuned-final"
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo "    OK - 微调模型权重存在 ($MODEL_DIR)"
else
    echo "    [警告] model.safetensors 未找到!"
    echo ""
    echo "    模型文件较大 (~418MB)，未包含在 Git 仓库中。"
    echo "    请从以下方式获取:"
    echo "      a) 从原始项目目录复制: bge-finetuned-final/model.safetensors"
    echo "      b) 从 HuggingFace 下载替换: BAAI/bge-base-en-v1.5"
    echo ""
    echo "    下载命令:"
    echo "    python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5').save_pretrained('$MODEL_DIR')\""
    echo ""
fi

# 5. 运行测试
echo "[5/5] 运行模型测试..."
python test_model.py && {
    echo ""
    echo "============================================"
    echo " 部署成功! 模型已就绪"
    echo "============================================"
    echo ""
    echo " 使用方式:"
    echo "    source venv/bin/activate"
    echo '    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('\''./bge-finetuned-final'\''); emb = model.encode('\''测试文本'\''); print('\''向量维度:'\'', len(emb))"'
}
