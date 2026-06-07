@echo off
chcp 65001 >nul
title 工单11 - BGE模型微调 环境部署

echo ============================================
echo  工单11 环境部署脚本 (Windows)
echo  BGE-base + Matryoshka 医疗问答检索模型
echo ============================================
echo.

:: 设置项目路径
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: 1. 检查 Python
echo [1/5] 检查 Python 环境...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未安装 Python，请安装 Python 3.9+ https://www.python.org/downloads/
    pause
    exit /b 1
)
python -c "import sys; ver=sys.version_info; assert ver.major==3 and ver.minor>=9, 'Python版本需要3.9+'"
echo     OK - Python %ERRORLEVEL% 可用

:: 2. 创建虚拟环境
echo [2/5] 创建虚拟环境...
if not exist "venv\" (
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo     虚拟环境已创建
) else (
    echo     虚拟环境已存在，跳过
)

:: 3. 安装依赖
echo [3/5] 安装 Python 依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [警告] 部分依赖安装失败，尝试分步安装...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers datasets pandas
)
echo     依赖安装完成

:: 4. 检查模型文件
echo [4/5] 检查模型文件...
set "MODEL_DIR=%PROJECT_DIR%bge-finetuned-final"
if exist "%MODEL_DIR%\model.safetensors" (
    echo     OK - 微调模型权重存在 (%MODEL_DIR%)
) else (
    echo [警告] model.safetensors 未找到!
    echo.
    echo     模型文件较大 (~418MB)，未包含在 Git 仓库中。
    echo     请从以下方式获取:
    echo       a) 从原始项目目录复制: bge-finetuned-final\model.safetensors
    echo       b) 从 HuggingFace 下载替换: BAAI/bge-base-en-v1.5
    echo.
    echo     下载命令:
    echo     python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5').save_pretrained('%MODEL_DIR%')"
    echo.
)

:: 5. 运行测试
echo [5/5] 运行模型测试...
python test_model.py
if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================
    echo  部署成功! 模型已就绪
    echo ============================================
    echo.
    echo  使用方式:
    echo     call venv\Scripts\activate
    echo     python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('./bge-finetuned-final'); emb = model.encode('测试文本'); print('向量维度:', len(emb))"
) else (
    echo [错误] 模型测试失败，请检查模型文件
)

pause
