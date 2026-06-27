#!/usr/bin/env python3
"""
ControlNet 预处理器模型下载脚本
从 hf-mirror.com 下载 OpenPose 等预处理器所需的模型文件
"""

import os
import sys
import urllib.request
import urllib.error

# 配置
HF_MIRROR = "https://hf-mirror.com"
ANNOT_DIR = r"E:\sd-webui\extensions\sd-webui-controlnet\annotator\downloads"

# 模型下载清单
MODELS = [
    {
        "name": "body_pose_model.pth",
        "url": f"{HF_MIRROR}/lllyasviel/Annotators/resolve/main/body_pose_model.pth",
        "size_hint": "~200 MB",
    },
    {
        "name": "hand_pose_model.pth",
        "url": f"{HF_MIRROR}/lllyasviel/Annotators/resolve/main/hand_pose_model.pth",
        "size_hint": "~45 MB",
    },
    {
        "name": "facenet.pth",
        "url": f"{HF_MIRROR}/lllyasviel/Annotators/resolve/main/facenet.pth",
        "size_hint": "~4 MB",
    },
    {
        "name": "yolox_l.onnx",
        "url": f"{HF_MIRROR}/yzd-v/DWPose/resolve/main/yolox_l.onnx",
        "size_hint": "~230 MB",
    },
    {
        "name": "dw-ll_ucoco_384.onnx",
        "url": f"{HF_MIRROR}/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
        "size_hint": "~100 MB",
    },
]


def download_file(url, dest, description=""):
    """带进度显示的下载函数"""
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size > 0:
            print(f"  [SKIP] {os.path.basename(dest)} 已存在 ({size:,} bytes)")
            return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    print(f"  [DOWN] {description or os.path.basename(dest)}")
    print(f"         URL: {url}")

    try:
        # 使用 urllib 避免 httpx SSL 问题
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')

        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = response.getheader('Content-Length')
            total_size = int(total_size) if total_size else None

            downloaded = 0
            chunk_size = 8192

            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        progress = downloaded / total_size * 100
                        print(f"\r         进度: {progress:.1f}% ({downloaded/1024/1024:.1f} MB)", end="")
                    else:
                        print(f"\r         已下载: {downloaded/1024/1024:.1f} MB", end="")

            print()  # 换行
            print(f"  [DONE] {os.path.basename(dest)} ({downloaded/1024/1024:.1f} MB)")
            return True

    except Exception as e:
        print(f"\n  [ERROR] 下载失败: {e}")
        # 清理未完成的部分
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    print("=" * 50)
    print("  ControlNet 预处理器模型下载")
    print(f"  目标目录: {ANNOT_DIR}")
    print("=" * 50)
    print()

    os.makedirs(ANNOT_DIR, exist_ok=True)

    success_count = 0
    total_count = len(MODELS)

    for model in MODELS:
        name = model["name"]
        url = model["url"]
        size_hint = model["size_hint"]
        dest = os.path.join(ANNOT_DIR, name)

        print(f"[{success_count + 1}/{total_count}] {name} ({size_hint})")

        if download_file(url, dest, name):
            success_count += 1
        print()

    # 验证
    print("=" * 50)
    print(f"  下载完成: {success_count}/{total_count} 成功")
    print("=" * 50)

    if success_count == total_count:
        print("\n  [OK] 所有预处理器模型已就绪")
        print("  重启 SD WebUI 后 ControlNet OpenPose 即可使用")
    else:
        print(f"\n  [WARN] {total_count - success_count} 个模型下载失败")
        print("  请检查网络连接后重新运行此脚本")
        sys.exit(1)


if __name__ == "__main__":
    main()
