@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================
echo   智能挂号助手 - 一键部署脚本
echo   工单11: 医疗挂号管理 Agent
echo ============================================
echo.

:: ==========================================
:: Step 1: 检查 Python 环境
:: ==========================================
echo [1/6] 检查 Python 环境...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo   [FAIL] 未检测到 Python，请先安装 Python 3.10+
    echo   下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo   [OK] Python !PY_VER!

:: ==========================================
:: Step 2: 创建虚拟环境
:: ==========================================
echo [2/6] 检查虚拟环境...
if not exist "venv\Scripts\python.exe" (
    echo   创建虚拟环境...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo   [FAIL] 虚拟环境创建失败，请检查 Python 安装
        pause
        exit /b 1
    )
    echo   [OK] venv 创建成功
) else (
    echo   [OK] 虚拟环境已存在
)

:: ==========================================
:: Step 3: 安装依赖
:: ==========================================
echo [3/6] 安装依赖...
call venv\Scripts\activate.bat
if not exist "requirements.txt" (
    echo   [FAIL] requirements.txt 不存在
    pause
    exit /b 1
)

:: 使用清华镜像源加速（国内网络）
venv\Scripts\python.exe -m pip install -r requirements.txt ^
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple ^
    --quiet
if !errorlevel! neq 0 (
    echo   [FAIL] 依赖安装失败，尝试官方源重试...
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo   [FAIL] 依赖安装仍然失败
        pause
        exit /b 1
    )
)
echo   [OK] 依赖安装完成

:: ==========================================
:: Step 4: 检查/创建 .env 配置文件
:: ==========================================
echo [4/6] 检查 API 配置...
if not exist ".env" (
    echo   创建 .env 模板...
    echo # DeepSeek API 配置 > .env
    echo DEEPSEEK_API_KEY=请在此处填入你的API_KEY >> .env
    echo DEEPSEEK_API_BASE=https://api.deepseek.com/v1 >> .env
    echo DEEPSEEK_MODEL=deepseek-chat >> .env
    echo.
    echo   [WARN] .env 已创建，但 API Key 未配置！
    echo   请编辑 .env 文件，将 DEEPSEEK_API_KEY 替换为真实密钥。
    echo   继续部署将跳过 API 验证。
    echo.
    choice /c YN /m "是否现在编辑 .env 文件？(Y=是, N=稍后编辑)"
    if !errorlevel! equ 1 (
        notepad .env
    )
) else (
    echo   [OK] .env 文件已存在
)

:: ==========================================
:: Step 5: 初始化数据库
:: ==========================================
echo [5/6] 初始化数据库...
venv\Scripts\python.exe database.py
if !errorlevel! neq 0 (
    echo   [FAIL] 数据库初始化失败
    pause
    exit /b 1
)
echo   [OK] 数据库初始化完成

:: ==========================================
:: Step 6: 运行部署验证
:: ==========================================
echo [6/6] 运行部署验证...
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); import database; import agent; print('模块导入测试通过')"
if !errorlevel! neq 0 (
    echo   [FAIL] 模块验证失败
    pause
    exit /b 1
)

echo.
echo ============================================
echo   部署完成！
echo.
echo   启动方式 1: python main.py
echo   启动方式 2: 双击 start.bat
echo   运行测试:   python test.py
echo   访问地址:   http://localhost:8511
echo ============================================
echo.
pause

