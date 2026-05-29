# =============================================
# Expert RAG System - Windows / WSL 部署脚本
# 用法（在 PowerShell 中运行）:
#   .\deploy.ps1
#   .\deploy.ps1 -SkipFrontend
# =============================================

param(
    [switch]$SkipFrontend,
    [switch]$SkipMilvus,
    [switch]$SkipMysql,
    [switch]$SkipModels,
    [switch]$SkipIndex
)

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Expert RAG System - Windows 部署" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectDir "backend"
$VenvDir = Join-Path $ProjectDir ".venv"

Write-Host "`n[信息] 项目目录: $ProjectDir" -ForegroundColor Cyan

# ---------- 1. 检查 WSL ----------
Write-Host "`n[步骤 1/7] 检查 WSL..." -ForegroundColor Yellow
$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if ($wsl) {
    Write-Host "  [✓] WSL 可用" -ForegroundColor Green
} else {
    Write-Host "  [✗] 未找到 WSL，请安装 WSL2 + Ubuntu" -ForegroundColor Red
    exit 1
}

# ---------- 2. 通过 WSL 创建虚拟环境 + 安装 Python 依赖 ----------
Write-Host "`n[步骤 2/7] 通过 WSL 安装 Python 依赖..." -ForegroundColor Yellow
wsl bash -c "
cd '$ProjectDir'
python3 -m venv .venv 2>/dev/null || python3 -m venv $VenvDir
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo 'Python 依赖安装完成'
"

# ---------- 3. 配置 .env ----------
Write-Host "`n[步骤 3/7] 配置环境变量..." -ForegroundColor Yellow
$envFile = Join-Path $BackendDir ".env"
if (-not (Test-Path $envFile)) {
    $envExample = Join-Path $ScriptDir ".env.example"
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Host "  [✓] 已创建 .env 文件: $envFile" -ForegroundColor Green
        Write-Host "  [!] 请用记事本编辑该文件，填入你的 DEEPSEEK_API_KEY" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [✓] .env 文件已存在" -ForegroundColor Green
}

# ---------- 4. 构建前端 ----------
if (-not $SkipFrontend) {
    Write-Host "`n[步骤 4/7] 构建前端..." -ForegroundColor Yellow
    $frontendDir = Join-Path $ProjectDir "frontend"

    if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
        Write-Host "  安装 npm 依赖..." -ForegroundColor Gray
        Set-Location $frontendDir
        npm install
    }

    Write-Host "  构建中..." -ForegroundColor Gray
    Set-Location $frontendDir
    npm run build

    # 复制到 backend/dist
    $backendDist = Join-Path $BackendDir "dist"
    if (Test-Path $backendDist) { Remove-Item -Recurse -Force $backendDist }
    Copy-Item -Recurse (Join-Path $frontendDir "dist") $backendDist
    Write-Host "  [✓] 前端构建完成" -ForegroundColor Green
} else {
    Write-Host "`n[步骤 4/7] 跳过前端构建" -ForegroundColor Yellow
}

# ---------- 5. 下载模型 ----------
if (-not $SkipModels) {
    Write-Host "`n[步骤 5/7] 下载本地模型..." -ForegroundColor Yellow
    $modelDir = Join-Path $BackendDir "bge-m3"
    $modelConfig = Join-Path $modelDir "config.json"

    if (-not (Test-Path $modelConfig)) {
        Write-Host "  下载 bge-m3 嵌入模型（约 2.2GB，需等待）..." -ForegroundColor Gray
        wsl bash -c "
cd '$ProjectDir'
source .venv/bin/activate
python3 -c '' 2>/dev/null || true
python3 -c \"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3', cache_folder='$modelDir')
model.save('$modelDir')
print('bge-m3 下载完成')
\"
"
        Write-Host "  [✓] bge-m3 下载完成" -ForegroundColor Green
    } else {
        Write-Host "  [✓] bge-m3 模型已存在" -ForegroundColor Green
    }
} else {
    Write-Host "`n[步骤 5/7] 跳过模型下载" -ForegroundColor Yellow
}

# ---------- 6. 构建索引 ----------
if (-not $SkipIndex) {
    Write-Host "`n[步骤 6/7] 构建知识库索引..." -ForegroundColor Yellow
    $rolesDir = Join-Path $BackendDir "vector_db" "roles"
    $medicalPkl = Join-Path $rolesDir "Medical" "faiss_index.pkl"

    if (-not (Test-Path $medicalPkl)) {
        wsl bash -c "
cd '$BackendDir'
source '$VenvDir/bin/activate'
python3 vector_db/build_index_v2.py 2>&1 || echo '索引构建脚本未找到，跳过'
"
        Write-Host "  [✓] 索引构建完成" -ForegroundColor Green
    } else {
        Write-Host "  [✓] 索引已存在" -ForegroundColor Green
    }
} else {
    Write-Host "`n[步骤 6/7] 跳过索引构建" -ForegroundColor Yellow
}

# ---------- 7. 启动 ----------
Write-Host "`n[步骤 7/7] 启动服务..." -ForegroundColor Yellow

# 创建 Windows 启动脚本
$startPs1 = @"
# Expert RAG System - WSL 启动脚本 (PowerShell)
`$ScriptDir = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$ProjectDir = Split-Path -Parent `$ScriptDir
`$BackendDir = Join-Path `$ProjectDir "backend"

Write-Host "启动 Expert RAG System..." -ForegroundColor Cyan
Write-Host "访问地址: http://127.0.0.1:8001" -ForegroundColor Cyan

wsl bash -c "
cd '$BackendDir'
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
source '$VenvDir/bin/activate'
python3 main.py
"
"@
$startPs1 | Out-File -FilePath (Join-Path $ScriptDir "start.ps1") -Encoding utf8
Write-Host "  [✓] 启动脚本已创建: deploy\start.ps1" -ForegroundColor Green

# 创建 start.bat（双击运行）
$startBat = "@echo off
echo ==============================================
echo  Expert RAG System - 启动
echo ==============================================
echo.
echo 启动中，请稍候...
echo 访问地址: http://127.0.0.1:8001
echo.
wsl bash -c ""cd '%~dp0..\backend' && export TRANSFORMERS_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && source '%~dp0..\.venv/bin/activate' && python3 main.py""
pause"
$startBat | Out-File -FilePath (Join-Path $ScriptDir "start.bat") -Encoding ascii
Write-Host "  [✓] 双击启动脚本已创建: deploy\start.bat" -ForegroundColor Green

# ---------- 完成 ----------
Write-Host "`n==============================================" -ForegroundColor Cyan
Write-Host "  部署完成！" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "`n  ⚡ 启动方式："
Write-Host "     1. 双击 deploy\start.bat"
Write-Host "     2. 或在终端中运行: deploy\start.ps1"
Write-Host "`n  📍 访问地址："
Write-Host "     http://127.0.0.1:8001"
Write-Host "     http://127.0.0.1:8001/docs  (API 文档)"
Write-Host "`n  📁 配置文件："
Write-Host "     backend\.env"
Write-Host "`n==============================================" -ForegroundColor Cyan
