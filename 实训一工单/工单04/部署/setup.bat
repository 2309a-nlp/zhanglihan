@echo off
chcp 65001 >nul
echo ========================================
echo  工单04 高稳定版PDF问答系统 - Windows部署
echo ========================================
echo.

:: 1. 系统要求
echo [1/5] 检查 Python 环境...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
python --version

:: 2. MySQL 检查
echo.
echo [2/5] 检查 MySQL...
mysqladmin ping -u root -p 2>nul
if %ERRORLEVEL% neq 0 (
    echo   [警告] MySQL 未检测到，请确保 MySQL 80 服务已启动
)

:: 3. 创建 .api_key 文件
echo.
echo [3/5] 配置 API Key...
if not exist ".api_key" (
    echo   [信息] 创建 .api_key 模板文件
    echo   [提示] 请用实际 API Key（Base64编码）替换 .api_key 内容
    echo placeholder_api_key_here > .api_key
    echo   请编辑 .api_key 文件并填入您的 DeepSeek API Key
    echo   （格式: base64编码后的密钥）
)

:: 4. 安装依赖
echo.
echo [4/5] 安装 Python 依赖...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo   [错误] 依赖安装失败
    pause
    exit /b 1
) else (
    echo   [完成] 依赖安装完成
)

:: 5. 启动服务
echo.
echo [5/5] 启动 Streamlit 服务...
echo.
echo   ========================================
echo   启动命令: streamlit run app.py --server.port 8503
echo   访问地址: http://localhost:8503
echo   ========================================
echo.
echo 启动中...
streamlit run app.py --server.port 8503

pause
