@echo off
chcp 65001 >nul 2>&1
title SD WebUI 部署脚本
echo ============================================
echo   Stable Diffusion WebUI 一键部署脚本
echo   版本: v1.10.1
echo   路径: E:\sd-webui
echo ============================================
echo.

set "SD_DIR=E:\sd-webui"
set "VENV_PYTHON=%SD_DIR%\venv\Scripts\python.exe"
set "PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "HF_MIRROR=https://hf-mirror.com"

echo [1/8] 检查 Python 环境...
if exist "%VENV_PYTHON%" (
    echo   [OK] venv Python 存在
    "%VENV_PYTHON%" --version
) else (
    echo   [ERROR] venv Python 不存在: %VENV_PYTHON%
    echo   请先运行 webui.bat 初始化虚拟环境
    pause
    exit /b 1
)

echo.
echo [2/8] 修复 Git 仓库 Commit Hash...
echo   修正 repositories/ 下各子模块的 commit 对齐问题

REM 检查 stable-diffusion-stability-ai 的 commit
cd /d "%SD_DIR%\repositories\stable-diffusion-stability-ai" 2>nul
if %errorlevel% equ 0 (
    for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "SD_COMMIT=%%i"
    if not "%SD_COMMIT%"=="21f890f" (
        echo   [WARN] stable-diffusion 当前: %SD_COMMIT%, 期望: 21f890f
        echo   [FIX] 切换到正确 commit...
        git checkout 21f890f 2>nul
    ) else (
        echo   [OK] stable-diffusion commit 正确: %SD_COMMIT%
    )
)

REM 检查 k-diffusion 的 commit
cd /d "%SD_DIR%\repositories\k-diffusion" 2>nul
if %errorlevel% equ 0 (
    for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "KD_COMMIT=%%i"
    if not "%KD_COMMIT%"=="4601bf0" (
        echo   [WARN] k-diffusion 当前: %KD_COMMIT%, 期望: 4601bf0
        echo   [FIX] 切换到正确 commit...
        git checkout 4601bf0 2>nul
    ) else (
        echo   [OK] k-diffusion commit 正确: %KD_COMMIT%
    )
)

REM 检查 BLIP 的 commit
cd /d "%SD_DIR%\repositories\BLIP" 2>nul
if %errorlevel% equ 0 (
    for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "BLIP_COMMIT=%%i"
    if not "%BLIP_COMMIT%"=="056a169" (
        echo   [WARN] BLIP 当前: %BLIP_COMMIT%, 期望: 056a169
        echo   [FIX] 切换到正确 commit...
        git checkout 056a169 2>nul
    ) else (
        echo   [OK] BLIP commit 正确: %BLIP_COMMIT%
    )
)

echo.
echo [3/8] 修复 CompVis 仓库兼容性问题...
REM 添加 use_linear 属性到 SpatialTransformer
set "ATTENTION_FILE=%SD_DIR%\repositories\stable-diffusion-stability-ai\ldm\modules\attention.py"
if exist "%ATTENTION_FILE%" (
    "%VENV_PYTHON%" -c "import sys; sys.exit(0 if 'use_linear' in open(r'%ATTENTION_FILE%', 'r', encoding='utf-8', errors='replace').read() else 1)" 2>nul
    if %errorlevel% neq 0 (
        echo   [FIX] 添加 use_linear 属性到 SpatialTransformer...
        "%VENV_PYTHON%" -c "
import re
path = r'%ATTENTION_FILE%'
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
# 在 SpatialTransformer.__init__ 中添加 use_linear
if 'self.use_linear' not in content:
    # 找到 __init__ 方法中的适当位置插入
    content = content.replace(
        'self.num_heads = num_heads',
        'self.num_heads = num_heads\n        self.use_linear = False'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('    [DONE] use_linear 已添加')
else:
    print('    [OK] use_linear 已存在')
"
    ) else (
        echo   [OK] use_linear 修复已应用
    )
) else (
    echo   [WARN] attention.py 不存在: %ATTENTION_FILE%
)

echo.
echo [4/8] 安装/修复关键依赖...
echo   [INFO] 使用清华 pip 镜像: %PIP_MIRROR%

REM 检查并修复 numpy 版本
"%VENV_PYTHON%" -c "import numpy; print(f'numpy {numpy.__version__}')" 2>nul
"%VENV_PYTHON%" -m pip install "numpy<2" --index-url %PIP_MIRROR% --quiet
echo   [OK] numpy 已锁定 < 2.0

REM 检查并修复 mediapipe 版本
"%VENV_PYTHON%" -c "import mediapipe; print(f'mediapipe {mediapipe.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo   [INFO] 安装 mediapipe 0.10.7...
    "%VENV_PYTHON%" -m pip install mediapipe==0.10.7 --index-url %PIP_MIRROR% --quiet --no-deps
) else (
    "%VENV_PYTHON%" -c "import mediapipe; exit(0 if mediapipe.__version__ == '0.10.7' else 1)" 2>nul
    if %errorlevel% neq 0 (
        echo   [FIX] 降级 mediapipe 到 0.10.7...
        "%VENV_PYTHON%" -m pip install mediapipe==0.10.7 --index-url %PIP_MIRROR% --quiet --no-deps
    ) else (
        echo   [OK] mediapipe 版本正确: 0.10.7
    )
)

REM 确保 numpy 没被重新升级
"%VENV_PYTHON%" -c "import numpy; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>nul
if %errorlevel% neq 0 (
    echo   [FIX] numpy 被重新升级，重新降级...
    "%VENV_PYTHON%" -m pip install numpy==1.26.4 --index-url %PIP_MIRROR% --quiet
)

REM 安装其他可能缺失的依赖
echo   [INFO] 检查 taming-transformers, dctorch, insightface...
"%VENV_PYTHON%" -m pip install taming-transformers dctorch==0.1.2 insightface --index-url %PIP_MIRROR% --quiet 2>nul
echo   [OK] 依赖检查完成

echo.
echo [5/8] 修复 ControlNet 扩展嵌套问题...
set "CN_DIR=%SD_DIR%\extensions\sd-webui-controlnet"
if exist "%CN_DIR%\sd-webui-controlnet-main" (
    echo   [FIX] 检测到 ZIP 嵌套，修复目录结构...
    "%VENV_PYTHON%" -c "
import shutil, os
nested = r'%CN_DIR%\sd-webui-controlnet-main'
cn_dir = os.path.dirname(nested)
for item in os.listdir(nested):
    src = os.path.join(nested, item)
    dst = os.path.join(cn_dir, item)
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    shutil.move(src, dst)
os.rmdir(nested)
print('    [DONE] 嵌套已修复')
"
) else (
    if exist "%CN_DIR%\scripts\controlnet.py" (
        echo   [OK] ControlNet 目录结构正确
    ) else (
        echo   [WARN] ControlNet scripts/ 目录不存在
    )
)

echo.
echo [6/8] 修复 midas 模块缺失问题...
set "MIDAS_DIR=%SD_DIR%\repositories\stable-diffusion-stability-ai\ldm\modules\midas"
if not exist "%MIDAS_DIR%" (
    echo   [FIX] 创建 midas 模块 stub...
    mkdir "%MIDAS_DIR%" 2>nul
    echo from . import api > "%MIDAS_DIR%\__init__.py"
    echo ISL_PATHS = [] > "%MIDAS_DIR%\api.py"
    echo def load_model(*args, **kwargs): pass >> "%MIDAS_DIR%\api.py"
    echo   [OK] midas stub 已创建
) else (
    echo   [OK] midas 模块已存在
)

REM 检查 ldm/data/util.py 的 AddMiDaS
set "UTIL_FILE=%SD_DIR%\repositories\stable-diffusion-stability-ai\ldm\data\util.py"
if exist "%UTIL_FILE%" (
    "%VENV_PYTHON%" -c "import sys; sys.exit(0 if 'AddMiDaS' in open(r'%UTIL_FILE%', 'r', encoding='utf-8', errors='replace').read() else 1)" 2>nul
    if %errorlevel% neq 0 (
        echo   [FIX] 添加 AddMiDaS stub...
        echo. >> "%UTIL_FILE%"
        echo class AddMiDaS: >> "%UTIL_FILE%"
        echo     def __init__(self, *args, **kwargs): pass >> "%UTIL_FILE%"
        echo     def __call__(self, data): return data >> "%UTIL_FILE%"
        echo   [OK] AddMiDaS stub 已添加
    )
)

echo.
echo [7/8] 检查 ControlNet 预处理器模型...
set "ANNOT_DIR=%SD_DIR%\extensions\sd-webui-controlnet\annotator\downloads"
mkdir "%ANNOT_DIR%" 2>nul

"%VENV_PYTHON%" -c "
import os
annot_dir = r'%ANNOT_DIR%'
needed = [
    'body_pose_model.pth',
    'hand_pose_model.pth',
    'facenet.pth',
    'yolox_l.onnx',
    'dw-ll_ucoco_384.onnx'
]
missing = [f for f in needed if not os.path.exists(os.path.join(annot_dir, f))]
if missing:
    print(f'    [WARN] 缺少预处理器模型:')
    for f in missing:
        print(f'      - {f}')
    print('    [INFO] 首次使用 OpenPose 时会自动下载，或手动运行 download_annotators.py')
else:
    print('    [OK] 所有预处理器模型已下载')
"

echo.
echo [8/8] 验证部署状态...
echo   检查关键文件...

set "MISSING=0"

if not exist "%SD_DIR%\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors" (
    echo   [MISSING] SD 基础模型
    set "MISSING=1"
) else (
    echo   [OK] SD 基础模型
)

if not exist "%SD_DIR%\models\ControlNet\control_v11p_sd15_openpose.pth" (
    echo   [MISSING] ControlNet OpenPose 模型
    set "MISSING=1"
) else (
    echo   [OK] ControlNet OpenPose 模型
)

if not exist "%SD_DIR%\config.json" (
    echo   [WARN] config.json 不存在（首次启动后生成）
) else (
    echo   [OK] config.json
)

if %MISSING% equ 1 (
    echo.
    echo [WARN] 部分模型文件缺失，请手动下载后重新运行
)

echo.
echo ============================================
echo   部署脚本执行完成
echo ============================================
echo.
echo 下一步:
echo   1. 运行 webui-user.bat 启动 WebUI
echo   2. 浏览器访问 http://localhost:7860
echo   3. 首次使用 ControlNet 时预处理器模型会自动下载
echo.
echo 如需下载 ControlNet 预处理器模型，运行:
echo   python download_annotators.py
echo.
pause
