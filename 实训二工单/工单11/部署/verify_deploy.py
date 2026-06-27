"""
部署验证脚本 - 检查项目完整性
用法: venv\Scripts\python.exe verify_deploy.py
"""

import os
import sys
import importlib
import sqlite3

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJ_DIR, "venv", "Scripts", "python.exe")

# 颜色输出
class Colors:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    RESET = "\033[0m"

passed = 0
failed = 0
warnings = 0

def check(name, condition, msg=""):
    global passed, failed, warnings
    if condition:
        passed += 1
        print(f"  {Colors.OK}[PASS]{Colors.RESET} {name}")
        return True
    elif condition is None:
        warnings += 1
        print(f"  {Colors.WARN}[WARN]{Colors.RESET} {name} - {msg}")
        return False
    else:
        failed += 1
        print(f"  {Colors.FAIL}[FAIL]{Colors.RESET} {name} - {msg}")
        return False

print("=" * 60)
print("智能挂号助手 - 部署验证")
print("=" * 60)

# 1. 文件完整性检查
print("\n[1/5] 文件完整性检查")
required_files = [
    "database.py", "agent.py", "app.py", "main.py", "test.py",
    "requirements.txt", "README.md", "start.bat"
]
for f in required_files:
    path = os.path.join(PROJ_DIR, f)
    check(f"存在 {f}", os.path.exists(path), "文件缺失")

# 2. Python 环境检查
print("\n[2/5] Python 环境检查")
check("虚拟环境", os.path.exists(VENV_PYTHON), "venv 不存在")

# 3. 依赖检查
print("\n[3/5] 依赖包检查")
deps = {
    "openai": ">=1.0.0",
    "streamlit": ">=1.30.0",
    "dotenv": ">=1.0.0",
    "sqlite3": "内置",
}
sys.path.insert(0, PROJ_DIR)

for mod_name, expected in deps.items():
    try:
        if mod_name == "dotenv":
            mod = importlib.import_module("dotenv")
        elif mod_name == "sqlite3":
            mod = importlib.import_module("sqlite3")
        else:
            mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "unknown")
        check(f"{mod_name} {expected}", True, f"已安装 (版本: {ver})")
    except ImportError as e:
        check(f"{mod_name} {expected}", False, f"导入失败: {e}")

# 4. 数据库检查
print("\n[4/5] 数据库检查")
db_path = os.path.join(PROJ_DIR, "hospital.db")
check("数据库文件", os.path.exists(db_path), "请先运行 database.py 初始化")

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = {"department", "doctor", "patientstatus", "patient", 
                          "diagnosis", "worker", "register", "schedule"}
        missing = expected_tables - set(tables)
        check(f"表完整性 ({len(tables)}/8)", len(missing) == 0, 
              f"缺失表: {', '.join(missing)}" if missing else "")
        
        # 检查种子数据
        dept_count = conn.execute("SELECT COUNT(*) FROM department").fetchone()[0]
        doc_count = conn.execute("SELECT COUNT(*) FROM doctor").fetchone()[0]
        sched_count = conn.execute("SELECT COUNT(*) FROM schedule").fetchone()[0]
        check(f"种子数据 (科室={dept_count}, 医生={doc_count}, 排班={sched_count})", 
              dept_count >= 6 and doc_count >= 6, "数据量不足")
        
        conn.close()
    except Exception as e:
        check("数据库连接", False, f"连接失败: {e}")

# 5. API 配置检查
print("\n[5/5] API 配置检查")
env_path = os.path.join(PROJ_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    has_key = "DEEPSEEK_API_KEY" in content
    has_base = "DEEPSEEK_API_BASE" in content
    
    if has_key:
        key_val = ""
        for line in content.splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key_val = line.split("=", 1)[1].strip()
                break
        is_placeholder = "请在此处填入" in key_val or "***" in key_val
        if is_placeholder:
            check("API Key 已配置", None, "当前为占位符，需替换为真实 Key")
        else:
            check("API Key 已配置", True)
    else:
        check("API Key 已配置", False, "未找到 DEEPSEEK_API_KEY")
    
    check("API Base URL", has_base, "未找到 DEEPSEEK_API_BASE")
else:
    check(".env 文件", False, "文件不存在")

# 总结
print("\n" + "=" * 60)
print(f"验证结果: {Colors.OK}{passed} 通过{Colors.RESET} | "
      f"{Colors.FAIL}{failed} 失败{Colors.RESET} | "
      f"{Colors.WARN}{warnings} 警告{Colors.RESET}")

if failed > 0:
    print(f"\n请修复 {failed} 个失败项后重新部署")
    sys.exit(1)
elif warnings > 0:
    print(f"\n警告: {warnings} 项需要手动配置，但部署可继续进行")
else:
    print("\n部署验证全部通过，系统就绪！")
print("=" * 60)
