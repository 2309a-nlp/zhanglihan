@echo off
chcp 65001 >nul
title 📅 日程提醒智能体

echo ============================================================
echo   📅 日程提醒智能体 - 启动脚本
echo ============================================================
echo.

REM 检查 Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查虚拟环境
if not exist "venv\" (
    echo 📦 正在创建虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
)

REM 激活虚拟环境并安装依赖
echo 📦 正在安装依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ⚠️ 部分依赖安装失败，尝试继续...
)

REM 检查 .env
if not exist ".env" (
    echo.
    echo ⚠️ 未找到 .env 文件！
    echo 请复制 .env.example 为 .env 并填入你的 DeepSeek API Key
    echo.
    copy .env.example .env >nul
    notepad .env
    echo.
    pause
)

echo.
echo 🚀 启动应用...
echo.
echo   访问地址: http://127.0.0.1:5000
echo.
echo ============================================================

python app.py

pause
