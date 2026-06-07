@echo off
chcp 65001 >nul
title 工单02 - 部署安装脚本
color 0B

echo ╔══════════════════════════════════════════════════════╗
echo ║  工单02 PDF智能问答系统 - 一键部署                   ║
echo ║  速度优化版  (Streamlit + DeepSeek + FAISS)          ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: ─── 1. 检测 Python ───
echo [1/6] 检测 Python 环境...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ 未检测到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo    Python %PYVER% ✓

:: ─── 2. 创建虚拟环境 ───
echo [2/6] 创建 Python 虚拟环境...
if exist ".venv\" (
    echo    虚拟环境已存在，跳过创建
) else (
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo    ⚠ venv 创建失败，将使用系统 Python
    ) else (
        echo    ✅ 虚拟环境已创建: .venv
    )
)

:: ─── 3. 安装依赖 ───
echo [3/6] 安装 Python 依赖包...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ❌ 依赖安装失败，请检查网络连接
    pause
    exit /b 1
)
echo    ✅ 依赖安装完成

:: ─── 4. 配置 API Key ───
echo [4/6] 配置 API Key...
if not exist ".api_key" (
    echo    ⚠ 未检测到 .api_key 文件
    echo    请将你的 DeepSeek API Key 写入 .api_key 文件:
    echo    ────────────────────────────────
    echo    echo sk-you...here ^> .api_key
    echo    ────────────────────────────────
    echo    或者直接输入（粘贴后回车）:
    set /p USER_KEY="  API Key: "
    if not "%USER_KEY%"=="" (
        echo %USER_KEY%> .api_key
        echo    ✅ .api_key 已创建
    )
) else (
    echo    ✅ .api_key 已存在
)

:: ─── 5. 创建必要目录 ───
echo [5/6] 创建数据目录...
if not exist "docs" mkdir docs
if not exist "uploads" mkdir uploads
if not exist "vector_store" mkdir vector_store
echo    ✅ 目录就绪

:: ─── 6. 检测 MySQL ───
echo [6/6] 检测 MySQL 服务...
sc query MySQL80 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo    ✅ MySQL80 服务运行中
) else (
    echo    ⚠ MySQL80 服务未运行
    echo    请确保 MySQL 已安装并运行在 127.0.0.1:3306
)

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║  🚀 部署完成！启动命令:                           ║
echo ╚══════════════════════════════════════════════════════╝
echo.
if exist ".venv\Scripts\activate.bat" (
    echo    .venv\Scripts\activate ^&^& streamlit run app.py
) else (
    echo    streamlit run app.py
)
echo.
echo  首次启动可能需要下载模型(bge-small-zh)，请耐心等待
echo  默认地址: http://localhost:8501
echo ═══════════════════════════════════════════════════════
echo.
pause
