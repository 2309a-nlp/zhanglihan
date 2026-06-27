#!/usr/bin/env python3
"""
SD WebUI 部署验证脚本
检查所有关键组件是否正常安装和配置
"""

import os
import sys
import subprocess

SD_DIR = r"E:\sd-webui"
VENV_PYTHON = os.path.join(SD_DIR, "venv", "Scripts", "python.exe")

PASS = "\u2705"
FAIL = "\u274c"
WARN = "\u26a0\ufe0f"
SKIP = "\u23ed\ufe0f"

results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    if not condition and "optional" in detail.lower():
        status = WARN
    results.append((status, name, detail))


def main():
    print("=" * 60)
    print("  Stable Diffusion WebUI 部署验证")
    print(f"  路径: {SD_DIR}")
    print("=" * 60)
    print()

    # 1. Python 环境
    print("[1/6] Python 环境检查...")
    check("Python 3.10 venv", os.path.exists(VENV_PYTHON), f"{VENV_PYTHON}")

    if os.path.exists(VENV_PYTHON):
        try:
            result = subprocess.run(
                [VENV_PYTHON, "--version"],
                capture_output=True, text=True, timeout=10
            )
            check("Python 版本", "3.10" in result.stdout, result.stdout.strip())
        except Exception as e:
            check("Python 执行", False, str(e))

        # 2. 关键依赖
        print("[2/6] 关键依赖检查...")
        deps = {
            "torch": None,
            "numpy": lambda v: v.startswith("1."),
            "gradio": lambda v: v.startswith("3."),
            "safetensors": None,
            "mediapipe": lambda v: v == "0.10.7",
            "insightface": None,
            "onnxruntime": None,
            "opencv-python": None,
            "transformers": None,
            "taming-transformers": None,
        }

        for pkg, version_check in deps.items():
            try:
                mod_name = pkg.replace('-', '_')
                # Special handling for packages with non-standard __version__
                if pkg == "gradio":
                    check_cmd = f"import gradio; print(getattr(gradio, '__version__', 'unknown'))"
                elif pkg == "mediapipe":
                    check_cmd = f"import mediapipe; print(getattr(mediapipe, '__version__', 'unknown'))"
                else:
                    check_cmd = f"import {mod_name}; print({mod_name}.__version__)"

                result = subprocess.run(
                    [VENV_PYTHON, "-c", check_cmd],
                    capture_output=True, text=True, timeout=10
                )
                version = result.stdout.strip()
                if version_check:
                    passed = version_check(version)
                else:
                    passed = True
                check(f"{pkg} {version}", passed, f"版本: {version}")
            except Exception as e:
                check(f"{pkg}", False, str(e))

    # 3. 模型文件
    print("[3/6] 模型文件检查...")
    models = [
        ("基础 SD 模型", r"models\Stable-diffusion\v1-5-pruned-emaonly.safetensors", True),
        ("ControlNet OpenPose", r"models\ControlNet\control_v11p_sd15_openpose.pth", True),
        ("ControlNet YAML", r"models\ControlNet\control_v11p_sd15_openpose.yaml", True),
        ("VAE-approx", r"models\VAE-approx\model.pt", False),
    ]

    for name, rel_path, required in models:
        full_path = os.path.join(SD_DIR, rel_path)
        exists = os.path.exists(full_path)
        size = os.path.getsize(full_path) if exists else 0
        size_str = f"{size/1024/1024/1024:.2f} GB" if size > 1024**3 else f"{size/1024/1024:.1f} MB" if size > 1024**2 else f"{size:,} B"
        check(f"{name} ({size_str})", exists, rel_path)

    # 4. Git 仓库
    print("[4/6] Git 仓库检查...")
    repos = {
        "stable-diffusion": (r"repositories\stable-diffusion-stability-ai", "21f890f"),
        "k-diffusion": (r"repositories\k-diffusion", "4601bf0"),
        "BLIP": (r"repositories\BLIP", "056a169"),
    }

    for name, (rel_path, expected_hash) in repos.items():
        full_path = os.path.join(SD_DIR, rel_path)
        if os.path.exists(full_path):
            try:
                result = subprocess.run(
                    ["git", "-C", full_path, "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=10
                )
                current = result.stdout.strip()[:7]
                expected = expected_hash[:7]
                check(f"{name} [{current}]", current == expected, f"期望: {expected}")
            except Exception as e:
                check(f"{name}", False, str(e))
        else:
            check(f"{name}", False, f"目录不存在: {rel_path}")

    # 5. ControlNet 扩展
    print("[5/6] ControlNet 扩展检查...")
    cn_scripts = os.path.join(SD_DIR, r"extensions\sd-webui-controlnet\scripts")
    check("ControlNet scripts/ 目录", os.path.exists(cn_scripts))

    if os.path.exists(cn_scripts):
        for script in ["controlnet.py", "cldm.py", "hook.py"]:
            full_path = os.path.join(cn_scripts, script)
            check(f"scripts/{script}", os.path.exists(full_path))

    # 6. 配置文件
    print("[6/6] 配置文件检查...")
    check("config.json", os.path.exists(os.path.join(SD_DIR, "config.json")))
    check("ui-config.json", os.path.exists(os.path.join(SD_DIR, "ui-config.json")))
    check("webui-user.bat", os.path.exists(os.path.join(SD_DIR, "webui-user.bat")))

    # 输出结果
    print()
    print("=" * 60)
    print(f"  验证结果: {sum(1 for s,_,_ in results if s==PASS)}/{len(results)} 通过")
    print("=" * 60)
    print()

    for status, name, detail in results:
        detail_str = f" - {detail}" if detail else ""
        print(f"  {status} {name}{detail_str}")

    print()

    failed = [name for status, name, _ in results if status == FAIL]
    if failed:
        print(f"[WARN] {len(failed)} 项检查失败:")
        for name in failed:
            print(f"  - {name}")
        print()
        print("运行 deploy.bat 修复大多数问题")
        return 1
    else:
        print("[OK] 所有检查通过，部署状态良好")
        return 0


if __name__ == "__main__":
    sys.exit(main())
