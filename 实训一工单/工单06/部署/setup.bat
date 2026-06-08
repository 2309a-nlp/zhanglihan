@echo off
chcp 65001 >nul
echo ========================================
echo RAG 多轮对话问答系统 v2 - 安装脚本
echo ========================================
echo.

echo [1/3] 安装 Python 依赖...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 安装失败，请检查 pip 配置
    pause
    exit /b 1
)

echo.
echo [2/3] 创建模型目录...
if not exist models mkdir models
if not exist data mkdir data
if not exist vector_store mkdir vector_store
if not exist fulltext_index mkdir fulltext_index
if not exist uploads mkdir uploads

echo.
echo [3/3] 请配置 API Key...
echo.
echo 请在 .api_key 文件中写入你的 API Key（DeepSeek 或 OpenAI）
echo 格式: sk-your-api-key-here
echo.
echo ========================================
echo 安装完成！
echo.
echo 启动方式:
echo   streamlit run app.py
echo ========================================
pause
