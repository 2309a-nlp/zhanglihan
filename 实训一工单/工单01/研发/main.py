# -*- coding: utf-8 -*-
"""
PDF 智能问答系统 - 入口文件

用法:
    python main.py             启动网页界面（推荐）
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # 防递归：如果已经由 Streamlit 运行（或已派发过一次），不再派发
    if os.environ.get("_RAG_STREAMLIT_ACTIVE") == "1":
        print("[错误] main.py 不能直接由 streamlit run 启动，请用 python main.py")
        sys.exit(1)

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print(f"[启动] 正在打开网页界面...")
    os.environ["_RAG_STREAMLIT_ACTIVE"] = "1"
    subprocess.Popen(
        ["streamlit", "run", app_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
