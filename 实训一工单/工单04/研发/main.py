# -*- coding: utf-8 -*-
"""
工单04 - PDF 智能问答系统（高稳定性 · 结构化输出版 · 入口文件）

用法:
    python main.py             启动网页界面（推荐）

新增特性:
  - 高稳定性：数据库自动重连 + 向量索引自动修复 + 健康检查线程
  - 异常容错：输入校验 + PDF多策略解析 + API重试+熔断 + 优雅降级
  - 结构化输出：列表格式（"· 条目"）替代段落式回答
"""
import os
import sys
import time
import signal
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def kill_old_streamlit():
    """杀掉残留的 streamlit 进程（防止旧进程占用端口）"""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq streamlit*"],
                capture_output=True, text=True, timeout=5,
            )
            if "streamlit" in result.stdout.lower():
                subprocess.run(
                    ["taskkill", "/F", "/IM", "streamlit*"],
                    capture_output=True, timeout=5,
                )
                print("[清理] 已终止残留的 Streamlit 进程")
                time.sleep(1)
        else:
            result = subprocess.run(
                ["pgrep", "-f", "streamlit run"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().splitlines()
                for pid in pids:
                    os.kill(int(pid), signal.SIGTERM)
                print(f"[清理] 已终止 {len(pids)} 个残留 Streamlit 进程")
                time.sleep(1)
    except Exception:
        pass  # 杀不掉也无所谓，反正新进程会覆盖


def main():
    # 防递归：如果已经由 Streamlit 运行（或已派发过一次），不再派发
    if os.environ.get("_RAG_STREAMLIT_ACTIVE") == "1":
        print("[错误] main.py 不能直接由 streamlit run 启动，请用 python main.py")
        sys.exit(1)

    # 清理旧进程
    kill_old_streamlit()

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print(f"[启动] 正在打开网页界面...")
    print(f"[特性] 高稳定性 · 自动恢复 · 熔断保护 · 结构化输出")
    print(f"[优化] 连接池复用 + 语义缓存 + 健康检查 + 异常容错")
    os.environ["_RAG_STREAMLIT_ACTIVE"] = "1"

    try:
        proc = subprocess.Popen(
            ["streamlit", "run", app_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[启动] Streamlit PID: {proc.pid}")
        print(f"[提示] 浏览器打开后，首次查询会预热连接池，稍慢属正常现象")
        print(f"[提示] 按 Ctrl+C 终止系统")
        proc.wait()
    except KeyboardInterrupt:
        print("\n[关闭] 正在终止系统...")
        proc.terminate()
        proc.wait(timeout=5)
        print("[关闭] 已退出")
    except Exception as e:
        print(f"[错误] 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
