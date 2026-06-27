#!/usr/bin/env python
"""
医疗挂号管理 Agent - 主入口
用法: python main.py
"""

import os
import sys
import subprocess
import time


def kill_old_streamlit():
    """Kill any orphaned Streamlit processes before starting a new one."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/IM", "streamlit*"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            ["pkill", "-f", "streamlit run"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(1)


def main():
    # 检查是否在 Streamlit 环境中
    if os.environ.get("STREAMLIT_SERVER_PORT"):
        print("[错误] 请使用 python main.py 启动，不要用 streamlit run main.py")
        sys.exit(1)
    
    # 清理旧进程
    kill_old_streamlit()
    
    # 启动 Streamlit
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print("启动智能挂号助手...")
    print(f"Streamlit 地址: http://localhost:8511")
    
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", "8511"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # 等待 Streamlit 启动
    time.sleep(3)
    print("浏览器已自动打开")


if __name__ == "__main__":
    main()

