@echo off
chcp 65001 >nul
title 📒 智能记账本 - 部署启动脚本

:: ==========================================================
:: 智能记账本 - 一键部署启动脚本
:: 使用方式：双击运行 或 在命令行执行 deploy.bat
:: ==========================================================

echo ╔═══════════════════════════════════════════════╗
echo ║       📒 智能记账本 - 部署启动脚本            ║
echo ╚═══════════════════════════════════════════════╝
echo.

:: ---------- 1. 检查 Python ----------
echo [1/5] 🔍 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误：未找到 Python，请先安装 Python 3.9+
    echo    下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo    ✅ Python %PY_VER%

:: ---------- 2. 检查 .env ----------
echo [2/5] 🔑 检查 API Key 配置...
if not exist ".env" (
    echo    ⚠️  未找到 .env 文件
    if exist ".env.example" (
        echo    📋 正在从 .env.example 创建 .env...
        copy ".env.example" ".env" >nul
        echo    ⚠️  请编辑 .env 文件，填入你的 DeepSeek API Key
        echo       文件路径：%cd%\.env
        echo       获取地址：https://platform.deepseek.com
        pause
        exit /b 1
    ) else (
        echo    ❌ 未找到 .env.example 模板文件
        pause
        exit /b 1
    )
) else (
    echo    ✅ .env 文件已存在
)

:: ---------- 3. 创建虚拟环境 ----------
echo [3/5] 📦 设置 Python 虚拟环境...
if not exist "venv" (
    echo    🔄 正在创建虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo    ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo    ✅ 虚拟环境已创建
) else (
    echo    ✅ 虚拟环境已存在
)

:: ---------- 4. 安装依赖 ----------
echo [4/5] 📥 安装项目依赖...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo    ❌ 虚拟环境激活失败
    pause
    exit /b 1
)

pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo    ⚠️  部分依赖安装失败，尝试详细模式...
    pip install -r requirements.txt
)

echo    ✅ 依赖安装完成

:: ---------- 5. 启动应用 ----------
echo [5/5] 🚀 启动智能记账本...
echo.
echo ╔═══════════════════════════════════════════════╗
echo ║        📒 智能记账本 启动中...                ║
echo ║                                               ║
echo ║    🌐 访问地址：http://127.0.0.1:5000         ║
echo ║                                               ║
echo ║    ❌ 关闭此窗口 = 停止服务                   ║
echo ╚═══════════════════════════════════════════════╝
echo.

python app.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 应用异常退出，错误码：%errorlevel%
    pause
)
