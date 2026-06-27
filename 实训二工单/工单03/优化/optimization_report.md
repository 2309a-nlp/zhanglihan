---
title: "Stable Diffusion WebUI 优化报告"
date: "2026-06-27"
target: "RTX 4060 Laptop 8GB"
---

# Stable Diffusion WebUI 优化报告

## 0. 硬件概况

| 项目 | 值 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| 显存 | 8 GB |
| 驱动版本 | 596.49 |
| 架构 | Ada Lovelace (SM 8.9) |
| CPU | 笔记本平台 |
| 部署盘 | E:\ (已有 ~12 GB 占用) |

---

## 1. 优化总览

| 类别 | 当前状态 | 优化后 | 预期提升 |
|------|----------|--------|----------|
| 启动参数 | `--skip-version-check` | 增加 `--xformers --medvram --opt-sdp-attention` | 速度 30-50%↑，显存 -1.5GB |
| HyperTile | 未启用 | 启用 | 速度 20-40%↑，质量几乎无损 |
| VAE | 使用内置 VAE | 添加独立 VAE | 色彩准确度↑，消除灰色雾感 |
| ControlNet | 仅 OpenPose | 添加 Canny/Depth/ILimit | 控制维度从 1→4+ |
| LoRA | 无 | 添加常用 LoRA | 风格/角色一致性 |
| 面部修复 | 模型未下载 | 添加 GFPGAN/CodeFormer | 人脸质量显著↑ |
| 超分辨率 | 框架就绪无模型 | 添加 4x-UltraSharp | 输出分辨率 2x-4x |
| 磁盘占用 | ~12 GB | 清理后 | 节省 1-2 GB |

---

## 2. 高优先级优化（立即执行）

### 2.1 启动参数优化

**当前：**
```bat
set COMMANDLINE_ARGS=--skip-version-check
```

**建议改为：**
```bat
set COMMANDLINE_ARGS=--skip-version-check --xformers --opt-sdp-attention --medvram --no-half-vae
```

| 参数 | 作用 | 影响 |
|------|------|------|
| `--xformers` | 启用 xformers 注意力优化 | 速度↑ 30-50%，显存↓ 1-2GB |
| `--opt-sdp-attention` | PyTorch 2.0+ SDPA 注意力 | 速度↑，作为 xformers 的补充 |
| `--medvram` | 中等显存模式 | 允许更高分辨率生成，速度轻微↓ |
| `--no-half-vae` | VAE 不使用半精度 | 防止 VAE 输出 NaN（灰色图像），RTX 40 系必须 |
| `--no-half` | (备选) 全部不使用半精度 | 解决黑图问题，但速度↓，显存↑ |

**⚠️ RTX 4060 特别注意：**
- 必须加 `--no-half-vae`，否则 VAE 解码会产生全灰/全黑图像
- xformers 需要额外安装：`pip install xformers`

### 2.2 安装 xformers

```bat
cd /d E:\sd-webui
venv\Scripts\python.exe -m pip install xformers ^
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

安装后验证：启动 WebUI 时日志应显示 `Using xformers cross attention`

### 2.3 启用 HyperTile（内置优化）

在 WebUI 设置 → Settings 中启用：

| 设置项 | 当前值 | 建议值 |
|--------|--------|--------|
| hypertile_enable_unet | False | **True** |
| hypertile_enable_vae | False | **True** |
| hypertile_max_tile_unet | 256 | 256（保持不变） |
| hypertile_swap_size_unet | 3 | 3（保持不变） |
| hypertile_max_tile_vae | 128 | 128（保持不变） |

**效果：** 在不降低图像质量的前提下，生成速度提升 20-40%。对 1024x512 及以上分辨率效果更明显。

### 2.4 添加独立 VAE 模型

当前使用模型内置 VAE，色彩可能出现灰暗/雾感。

**推荐 VAE：**

| VAE 模型 | 大小 | 特点 | 下载链接 |
|----------|------|------|----------|
| vae-ft-mse-840000-ema-pruned | 822 MB | SD 1.5 最佳通用 VAE，色彩鲜艳 | hf-mirror.com/stabilityai/sd-vae-ft-mse-original |
| kl-f8-anime2 | 208 MB | 动漫风格推荐 | hf-mirror.com/hakurei/kl-f8-anime2 |

**安装：**
下载 `.safetensors` 文件放入 `E:\sd-webui\models\VAE\`

**使用：** 在 WebUI 设置 → VAE 中选择下载的模型，或将 `sd_vae` 添加到 config.json

---

## 3. 中优先级优化（按需扩展）

### 3.1 扩展 ControlNet 模型

当前仅有 OpenPose（人体姿态），建议添加：

| 模型 | 用途 | 大小 | 推荐场景 |
|------|------|------|----------|
| control_v11p_sd15_canny | 边缘线稿控制 | 1.4 GB | 线稿上色、建筑草图 |
| control_v11f1p_sd15_depth | 深度图控制 | 1.4 GB | 3D 场景构图、产品渲染 |
| control_v11p_sd15_scribble | 涂鸦控制 | 1.4 GB | 手绘草图细化 |
| control_v11p_sd15_inpaint | 修复引导 | 1.4 GB | 局部替换修复 |
| control_v11p_sd15_lineart | 线稿提取 | 1.4 GB | 动漫线稿上色 |

**批量下载命令：**
```bat
set HF_ENDPOINT=https://hf-mirror.com
curl -L -o "E:\sd-webui\models\ControlNet\control_v11p_sd15_canny.pth" ^
  "https://hf-mirror.com/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth"
```

### 3.2 面部修复模型

当前 GFPGAN/CodeFormer 模型目录为空，人脸生成质量不佳。

| 模型 | 大小 | 用途 |
|------|------|------|
| GFPGAN v1.4 | 341 MB | 人脸修复/增强 |
| CodeFormer | 376 MB | 人脸恢复（效果更好但稍慢） |
| realesrgan-x4plus | 66 MB | 通用 4x 超分辨率 |
| realesr-animevideov3 | 45 MB | 动漫视频超分 |

**安装：** 放入 `E:\sd-webui\models\GFPGAN\` 和 `E:\sd-webui\models\Codeformer\`

### 3.3 LoRA 模型

| LoRA 类型 | 推荐模型 | 大小 | 用途 |
|-----------|----------|------|------|
| 细节增强 | add_detail | ~200 MB | 增加图像细节丰富度 |
| 低权重通用 | low Detail Tweaker | ~100 MB | 微调细节 |
| 动漫风格 | Anime style LoRA | ~200 MB | 动漫画风 |
| 摄影风格 | FilmGrain / Photography | ~100 MB | 真实摄影风格 |

**安装：** 放入 `E:\sd-webui\models\Lora\`，在 WebUI 中通过 LoRA 标签 `<lora:name:weight>` 调用

### 3.4 超分辨率 Upscaler

| 模型 | 大小 | 用途 |
|------|------|------|
| 4x-UltraSharp | 66 MB | 通用 4x 锐化超分（推荐） |
| 4x_NMKD-Siax_200k | 78 MB | 写实照片超分 |
| R-ESRGAN 4x+ | 66 MB | 通用超分 |
| LDSR | 较大 | 高质量但慢 |

**安装：** 放入 `E:\sd-webui\models\ESRGAN\`

---

## 4. 低优先级优化（高级调优）

### 4.1 交叉注意力优化选择

| 方法 | 速度 | 显存 | 兼容性 | 推荐度 |
|------|------|------|--------|--------|
| xformers | 最快 | 最低 | 需安装 | ★★★★★ |
| SDPA (PyTorch 2.0+) | 快 | 低 | 内置 | ★★★★ |
| Sub-quadratic | 中 | 中 | 内置 | ★★ |
| Doggettx | 慢 | 高 | 默认 | ★ |

**当前 torch 2.12.0 已内置 SDPA**，如果 xformers 安装失败，至少添加 `--opt-sdp-attention`

### 4.2 显存管理

RTX 4060 Laptop 8GB 显存分配建议：

| 场景 | 推荐设置 | 最大分辨率 |
|------|----------|------------|
| 普通文生图 | --medvram | 1024x1024 |
| 高清修复 | --medvram --no-half-vae | 512→2048 |
| ControlNet 多单位 | --medvram --lowvram | 512x512 |
| 模型融合 | --lowvram | N/A |

**显存碎片化优化：**
```bat
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```
这可以减少显存碎片化，防止 "out of memory" 错误。

### 4.3 批量生成优化

```json
{
  "outdir_samples": "",
  "outdir_grids": "",
  "outdir_txt2img_samples": "E:\\sd-webui\\outputs\\txt2img-images",
  "outdir_img2img_samples": "E:\\sd-webui\\outputs\\img2img-images",
  "outdir_extras_samples": "E:\\sd-webui\\outputs\\extras-images",
  "outdir_grids": "",
  "grid_save": true,
  "grid_extended_filename": true,
  "grid_only_if_multiple": true,
  "n_rows": 0
}
```

### 4.4 采样器选择建议

| 采样器 | 速度 | 质量 | 推荐用途 |
|--------|------|------|----------|
| DPM++ 2M Karras | 快 | 高 | 日常使用首选 |
| DPM++ SDE Karras | 中 | 最高 | 追求质量 |
| Euler a | 快 | 中 | 风格化/创意 |
| DDIM | 快 | 中 | 兼容性测试 |
| UniPC | 最快 | 中高 | 快速预览 |

### 4.5 CFG Scale 建议

| 范围 | 效果 | 推荐场景 |
|------|------|----------|
| 3-5 | 宽松，创意多 | 艺术探索 |
| 6-8 | 平衡 | 日常使用 |
| 9-12 | 严格遵循提示词 | 精确控制 |
| >15 | 过度拟合，色彩过饱和 | 不推荐 |

---

## 5. 存储优化

### 5.1 磁盘清理

| 目录 | 当前 | 操作 | 可节省 |
|------|------|------|--------|
| `__pycache__/` | 4.8 KB | 清理 | 微量 |
| `tmp/` | 0.2 KB | 清理 | 微量 |
| `cache/` | 160 KB | 保留 | - |
| `outputs/` | 2.7 MB | 按需清理旧图 | 视情况 |
| `repositories/.git/` | ~100 MB | 浅克隆优化 | 50-80 MB |

### 5.2 Git 仓库浅克隆优化

当前 repositories/ 占 198.8 MB，可通过浅克隆减少：

```bat
cd /d E:\sd-webui\repositories\stable-diffusion-stability-ai
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### 5.3 输出管理建议

```
outputs/
├── txt2img-images/          # 按日期子目录管理
│   ├── 2026-06-27/
│   └── 2026-06-28/
├── img2img-images/
├── extras-images/
└── grids/                   # 网格对比图
```

在 config.json 中设置 `samples_filename_pattern: "[datetime]-[seed]-[prompt]"` 方便检索。

---

## 6. 一键优化脚本

建议创建的优化脚本 `optimize.bat`：

```bat
@echo off
echo ============================================
echo   SD WebUI 一键优化
echo ============================================

set "SD_DIR=E:\sd-webui"
set "VENV_PYTHON=%SD_DIR%\venv\Scripts\python.exe"
set "PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

echo.
echo [1/4] 安装 xformers...
%VENV_PYTHON% -m pip install xformers --index-url %PIP_MIRROR%

echo.
echo [2/4] 锁定 numpy（xformers 可能重新升级）...
%VENV_PYTHON% -m pip install numpy==1.26.4 --index-url %PIP_MIRROR%

echo.
echo [3/4] 更新 webui-user.bat 启动参数...
rem 自动添加优化参数到 COMMANDLINE_ARGS

echo.
echo [4/4] 下载 VAE 模型...
set HF_ENDPOINT=https://hf-mirror.com
curl -L -o "%SD_DIR%\models\VAE\vae-ft-mse-840000-ema-pruned.safetensors" ^
  "https://hf-mirror.com/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"

echo.
echo 优化完成！重启 WebUI 后生效。
pause
```

---

## 7. 优化路线图

```
Phase 1（立即执行，30 分钟）
├── 安装 xformers
├── 更新 webui-user.bat 启动参数
├── 启用 HyperTile
└── 下载 VAE 模型

Phase 2（按需扩展，2-3 小时）
├── 下载 2-3 个额外 ControlNet 模型
├── 下载 GFPGAN + CodeFormer
├── 下载 4x-UltraSharp 超分模型
└── 下载 1-2 个 LoRA 模型

Phase 3（长期维护）
├── 定期清理 outputs/ 目录
├── 关注 SD WebUI 版本更新
├── 按需添加新 ControlNet/LoRA
└── 监控显存使用情况，调整参数
```

---

## 8. 预期性能基准

以 RTX 4060 Laptop 8GB + SD 1.5 为基准：

| 场景 | 优化前 | 优化后（xformers + HyperTile） |
|------|--------|-------------------------------|
| 512x512, 20 步 | ~8-10 秒/张 | ~4-6 秒/张 |
| 768x768, 20 步 | ~15-20 秒/张 | ~8-12 秒/张 |
| 1024x1024, 20 步 | ~25-35 秒/张 | ~14-20 秒/张 |
| 512→2048 Hires fix | ~45-60 秒/张 | ~25-35 秒/张 |
| 显存占用 | ~5.5-6.5 GB | ~3.5-4.5 GB |

> 实际速度取决于具体采样器、CFG 步数、是否使用 ControlNet 等因素。

---

## 9. 风险提醒

| 优化 | 风险 | 缓解措施 |
|------|------|----------|
| xformers | 可能与 torch 版本不兼容 | 安装失败则回退到 `--opt-sdp-attention` |
| --medvram | 极端高分辨率可能变慢 | 保持 1024x1024 以下 |
| HyperTile | 极高分权重可能轻微影响细节 | 对 SD 1.5 影响极小 |
| numpy 降级 | 某些包可能要求 numpy>=2.0 | 保持 numpy 1.26.x，SD 生态兼容 |

