# -*- coding: utf-8 -*-
# backend/api/routes.py

import logging
import json
import time
import os
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils.auth import get_user, create_user
from services.agent import run_chat

logger = logging.getLogger(__name__)

_milvus_available = False
_save_to_milvus = None
try:
    from milvus_store import save_chat_to_milvus as _save_to_milvus
    _milvus_available = True
except Exception as e:
    logger.warning(f"Milvus 未就绪，对话存储到 Milvus 功能暂时禁用: {e}")

def save_chat_to_milvus(dialog_id, user_id, question, answer):
    if _milvus_available and _save_to_milvus:
        try:
            _save_to_milvus(dialog_id, user_id, question, answer)
        except Exception as e:
            logger.warning(f"Milvus 存储失败: {e}")

router = APIRouter(tags=["核心业务接口"])

@router.post("/register")
async def register(data: dict):
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return JSONResponse({"code": 400, "msg": "用户名或密码不能为空"})
    if create_user(username, password):
        return JSONResponse({"code": 200, "msg": "注册成功"})
    else:
        return JSONResponse({"code": 400, "msg": "用户已存在"})

@router.post("/login")
async def login(data: dict):
    username = data.get("username")
    password = data.get("password")
    if get_user(username) != password:
        return JSONResponse({"code": 400, "msg": "账号或密码错误"})
    return JSONResponse({"code": 200, "msg": "登录成功"})

@router.post("/chat")
async def chat(data: dict):
    try:
        question = data.get("question", "").strip()
        if not question:
            return JSONResponse({"code": 400, "msg": "问题不能为空！"})

        username = (data.get("username") or "guest").strip() or "guest"
        role = data.get("role", "Medical")
        if role not in ("Medical", "Finance", "Law", "Education", "Psychology"):
            role = "Medical"

        result = run_chat(question, session_id=username, role=role, user_id=username)

        answer = ""
        contexts = []
        meta = {}
        scores = {}

        if isinstance(result, tuple) and len(result) >= 3:
            answer, contexts, meta = result[0], result[1], result[2]
        elif isinstance(result, tuple) and len(result) >= 2:
            answer, contexts = result[0], result[1]
        elif isinstance(result, str):
            answer = result
        else:
            answer = str(result) if result else ""

        if not isinstance(answer, str):
            answer = str(answer) if answer else ""

        # 若 answer 为空但 meta 可能有错误信息
        if not answer and isinstance(meta, dict) and meta.get("error"):
            answer = f"回答生成失败: {meta['error']}"

        if not answer and isinstance(result, str) and "error" in result.lower():
            # 保持原样
            pass

        try:
            save_chat_to_milvus(
                dialog_id=username, user_id=username,
                question=question, answer=answer
            )
        except Exception as db_err:
            logger.warning(f"存入 Milvus 失败: {db_err}")

        # RAGAS 评分
        if contexts:
            try:
                from services.ragas_evaluator import evaluate_single_qa
                scores = evaluate_single_qa(question, answer, contexts, use_ragas=True)
            except Exception as e:
                logger.warning(f"RAGAS 评分失败（不影响回答）: {e}")

        # 可视化 RAGAS 评分（追加到 answer 尾部）
        if scores:
            ragas_lines = [
                "",
                "--- RAGAS 评分 ---",
                f"忠实性(上下文): {scores.get('faithfulness', 0):.2f}",
                f"相关性(回答切题): {scores.get('answer_relevancy', 0):.2f}",
                f"精确度(上下文精准): {scores.get('context_precision', 0):.2f}",
                f"召回率(上下文覆盖): {scores.get('context_recall', 0):.2f}",
                f"综合评分: {scores.get('overall', 0):.2f}",
            ]
            answer += "\n" + "\n".join(ragas_lines)

        return JSONResponse({
            "code": 200,
            "msg": "成功",
            "answer": answer,
            "contexts": contexts,
            "ragas_score": scores,
            "kb_hit": bool(meta.get("kb_hit")),
            "kb_available": bool(meta.get("kb_available")),
            "answer_mode": meta.get("answer_mode", ""),
            "source_label": meta.get("source_label", ""),
            "citations": meta.get("citations") or [],
            "best_similarity": meta.get("best_similarity"),
            "similarity_threshold": meta.get("similarity_threshold"),
        })
    except Exception as e:
        logger.error(f"服务器错误：{e}")
        return JSONResponse({"code": 500, "msg": f"服务器错误：{e}"})

# ================= RAGAS 评估模块 =================

@router.post("/evaluate")
async def evaluate_rag(data: dict):
    """触发 RAGAS 评估。请求参数: role(可选), questions(可选)"""
    try:
        role = data.get("role")
        questions = data.get("questions")

        from services.ragas_evaluator import evaluate_role, evaluate_all_roles

        if role:
            report = evaluate_role(role, questions=questions, verbose=True)
        else:
            results = evaluate_all_roles(verbose=True)
            report = {
                "all_roles": {r: res for r, res in results.items()},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        return JSONResponse({"code": 200, "msg": "评估完成", "data": report})
    except Exception as e:
        logger.error(f"RAGAS 评估失败: {e}")
        return JSONResponse({"code": 500, "msg": f"评估失败: {e}"})


@router.post("/score")
async def score_qa(data: dict):
    """对单个问答对进行 RAGAS 评分"""
    try:
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        contexts = data.get("contexts", [])
        use_ragas = data.get("use_ragas", True)

        if not question or not answer:
            return JSONResponse({"code": 400, "msg": "question 和 answer 不能为空"})
        if not contexts:
            return JSONResponse({"code": 400, "msg": "contexts 不能为空"})

        from services.ragas_evaluator import evaluate_single_qa
        scores = evaluate_single_qa(question, answer, contexts, use_ragas=use_ragas)

        return JSONResponse({"code": 200, "msg": "评分完成", "data": scores})
    except Exception as e:
        logger.error(f"RAGAS 评分失败: {e}")
        return JSONResponse({"code": 500, "msg": f"评分失败: {e}"})


@router.get("/eval_results")
async def get_eval_results():
    """获取所有历史评估结果文件列表"""
    try:
        from services.ragas_evaluator import EVAL_RESULTS_DIR

        if not os.path.isdir(EVAL_RESULTS_DIR):
            return JSONResponse({"code": 200, "msg": "暂无评估结果", "data": []})

        files = []
        for fn in sorted(os.listdir(EVAL_RESULTS_DIR), reverse=True):
            if fn.endswith(".json"):
                fp = os.path.join(EVAL_RESULTS_DIR, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    files.append({
                        "filename": fn,
                        "timestamp": report.get("timestamp", ""),
                        "role": report.get("role", "all"),
                        "overall_score": report.get("overall_score"),
                        "num_questions": report.get("num_questions", 0),
                    })
                except:
                    files.append({"filename": fn, "timestamp": "", "role": "unknown"})

        return JSONResponse({"code": 200, "msg": "成功", "data": files})
    except Exception as e:
        logger.error(f"获取评估结果失败: {e}")
        return JSONResponse({"code": 500, "msg": f"获取失败: {e}"})
