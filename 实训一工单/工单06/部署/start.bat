@echo off
chcp 65001 >nul
title RAG 多轮问答系统 v2 - 端口 8506

echo ========================================
echo  启动 RAG 多轮问答系统 v2
echo ========================================
echo.

REM 检查 .api_key
if not exist .api_key (
    echo [警告] .api_key 文件不存在，请先配置 API Key
    echo 格式: echo sk-your-key ^> .api_key
    pause
    exit /b 1
)

REM 检查依赖
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [提示] 首次使用请先运行 setup.bat 安装依赖
    pause
    exit /b 1
)

echo [OK] 正在启动 Streamlit 服务...
echo [OK] 访问地址: http://localhost:8506
echo [OK] 按 Ctrl+C 停止服务
echo.

python -m streamlit run app.py --server.port 8506

pause
