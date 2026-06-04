import os
import logging
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- 1. 配置日志系统 ---
logger = logging.getLogger("ExpertRAG")  # 获取名为 ExpertRAG 的日志记录器
logger.setLevel(logging.INFO)  # 设置最低日志级别为 INFO

# 定义日志输出格式：包含时间、模块名、函数名、行号、级别和具体消息
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

# 创建控制台处理器（让日志显示在 PyCharm 的终端里）
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- 2. 强制指定 .env 路径并加载 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")

if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"环境变量文件加载成功: {env_path}")  # 替换原有的 print ✅
else:
    logger.warning(f"未找到 .env 文件，请检查路径: {env_path}")  # 替换原有的 print ❌

# --- 3. 导入业务模块 ---
# settings 会在内部读取刚才加载的环境变量
from config import settings
from api.routes import router as api_router

# --- 4. 创建应用实例 ---
app = FastAPI(title="Expert RAG System")

# --- 5. 中间件配置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. 注册路由 ---
app.include_router(api_router, prefix="/api")

# --- 7. 静态文件挂载 ---
app.mount("/", StaticFiles(directory=settings.STATIC_DIR, html=True), name="static")

# --- 8. 启动服务 ---
if __name__ == "__main__":
    # 替换原有的 print 🚀
    logger.info(f"后端启动中... 请访问 http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)