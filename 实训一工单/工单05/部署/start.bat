@echo off
chcp 65001 >nul
title 工单05 - RAG系统

set BASE_DIR=%~dp0
cd /d "%BASE_DIR%"

:: Check .api_key
if not exist .api_key (
    echo [ERROR] .api_key not found! Please create .api_key file.
    pause
    exit /b 1
)

echo ========================================
echo  工单05 - TF-IDF RAG 问答系统
echo  端口: 8505
echo  模型: DeepSeek Chat
echo  索引: TF-IDF (2176 chunks)
echo ========================================
echo.
echo 启动中...

D:n\envs\zg5\python.exe -m streamlit run app.py --server.port 8505
pause
