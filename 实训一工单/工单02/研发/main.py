# -*- coding: utf-8 -*-
"""
工单02 - PDF 智能问答系统（入口文件）

用法:
    python main.py             启动网页界面（推荐）

优化特性:
  - 自动预热 LLM 连接池（省 0.5-1.5s TCP 握手）
  - 语义缓存（相似问题秒回）
  - 64 tokens 快速生成
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
    print(f"[优化] 连接池复用 + 语义缓存 + LLM_MAX_TOKENS=64")
    os.environ["_RAG_STREAMLIT_ACTIVE"] = "1"
    subprocess.Popen(
        ["streamlit", "run", app_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
