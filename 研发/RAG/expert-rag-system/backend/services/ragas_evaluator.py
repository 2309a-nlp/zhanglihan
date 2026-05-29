# -*- coding: utf-8 -*-
"""
RAGAS 评估系统 - 基于 ragas 官方库 + 简化版关键词评估作为备选
"""

import os
import sys
import json
import logging
import random
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

EVAL_RESULTS_DIR = os.path.join(current_dir, "eval_results")
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

QUESTIONS_PER_ROLE = 5
EVAL_MODE = os.getenv("RAGAS_EVAL_MODE", "ragas").strip().lower()

# ==================== 测试问题 ====================

DEFAULT_TEST_QUESTIONS = {
    "Medical": [
        "高血压患者每天盐摄入量应控制在多少克以下？",
        "高血压的营养指导原则有哪些？",
        "高血压患者应该如何控制体重？",
        "高血压患者运动时应注意什么？",
        "哪些食物有助于控制高血压？",
    ],
    "Finance": [
        "什么是基金定投？",
        "如何评估股票的投资价值？",
        "个人理财规划应该从哪些方面考虑？",
        "什么是资产配置？",
        "如何选择适合自己的保险产品？",
    ],
    "Law": [
        "合同纠纷的诉讼时效是多久？",
        "劳动法中关于加班工资如何规定？",
        "什么是知识产权？包括哪些类型？",
        "离婚时财产如何分割？",
        "什么是正当防卫？",
    ],
    "Education": [
        "什么是探究式学习？",
        "如何培养学生的批判性思维？",
        "现代教育技术在教学中的应用有哪些？",
        "如何有效进行课堂管理？",
        "什么是差异化教学？",
    ],
    "Psychology": [
        "什么是认知行为疗法？",
        "如何缓解焦虑情绪？",
        "抑郁症的早期症状有哪些？",
        "什么是情绪智力？如何提升？",
        "如何建立健康的亲密关系？",
    ],
}

def generate_test_questions(role: str) -> List[str]:
    if role in DEFAULT_TEST_QUESTIONS:
        questions = DEFAULT_TEST_QUESTIONS[role].copy()
        random.shuffle(questions)
        return questions[:QUESTIONS_PER_ROLE]
    return []

# ==================== ragas 官方评分 ====================

_RAGAS_AVAILABLE = False
try:
    from ragas.metrics import faithfulness as _ragas_f, answer_relevancy as _ragas_ar
    _RAGAS_AVAILABLE = True
    logger.info("ragas 官方评估器导入成功")
except ImportError:
    logger.warning("ragas 未安装，将使用简化版评估")

# ==================== 简化版关键词评估（备选） ====================

def _faith_simple(answer: str, contexts: List[str]) -> float:
    if not answer or not contexts:
        return 0.0
    import jieba, re
    ctx_text = " ".join(contexts)
    sents = [s.strip() for s in re.split(r'[。！？\n]', answer) if len(s.strip()) > 3]
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        kw = [w for w in jieba.cut(s) if len(w.strip()) > 1]
        if not kw:
            ok += 1
            continue
        if sum(1 for w in kw if w in ctx_text) / len(kw) >= 0.3:
            ok += 1
    return ok / len(sents)

def _relev_simple(question: str, answer: str) -> float:
    if not question or not answer:
        return 0.0
    import jieba
    qk = {w for w in jieba.cut(question) if len(w.strip()) > 1}
    if not qk:
        return 1.0
    ak = set(jieba.cut(answer))
    return sum(1 for w in qk if w in ak) / len(qk)

def _prec_simple(question: str, contexts: List[str]) -> float:
    if not question or not contexts:
        return 0.0
    import jieba
    qk = {w for w in jieba.cut(question) if len(w.strip()) > 1}
    if not qk:
        return 1.0
    return sum(1 for c in contexts if any(kw in c for kw in qk)) / len(contexts)

def _recall_simple(question: str, contexts: List[str]) -> float:
    return _prec_simple(question, contexts)

# ==================== 核心评分函数 ====================

_g_judge = None
_g_judge_inited = False

def _get_judge():
    """获取 ragas 评判 LLM 实例（单例）"""
    global _g_judge, _g_judge_inited
    if _g_judge_inited:
        return _g_judge
    _g_judge_inited = True
    try:
        from ragas.llms import llm_factory
        from openai import OpenAI
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        # 设 OPENAI_API_KEY 为 DeepSeek Key，让 ragas 内部所有 OpenAI 客户端都能用
        if deepseek_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = deepseek_key
        client = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com/v1",
        )
        _g_judge = llm_factory("deepseek-chat", client=client)
        logger.info("RAGAS 评判模型: DeepSeek Chat")
    except Exception as e:
        logger.warning(f"创建评判模型失败: {e}")
        _g_judge = None
    return _g_judge


def evaluate_single_qa(
    question: str,
    answer: str,
    contexts: List[str],
    use_ragas: bool = True,
) -> Dict[str, float]:
    """
    对单个问答对评分。

    ragas 官方（faithfulness、answer_relevancy）
    简化版补全（context_precision、context_recall）
    """
    scores = {}

    if use_ragas and _RAGAS_AVAILABLE:
        judge = _get_judge()
        if judge:
            try:
                from ragas import evaluate as _evaluate
                from datasets import Dataset as _Dataset

                _ragas_f.llm = judge
                _ragas_ar.llm = judge

                ds = _Dataset.from_dict({
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                })

                result = _evaluate(ds, metrics=[_ragas_f, _ragas_ar], llm=judge)

                for name in ["faithfulness", "answer_relevancy"]:
                    try:
                        val = result[name]
                        if isinstance(val, (list, tuple)):
                            val = val[0]
                        if val is not None and not (isinstance(val, float) and val != val):
                            scores[name] = round(float(val), 4)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"ragas 评分失败: {e}")

    # 简化版补全
    if "faithfulness" not in scores:
        scores["faithfulness"] = round(_faith_simple(answer, contexts), 4)
    if "answer_relevancy" not in scores:
        scores["answer_relevancy"] = round(_relev_simple(question, answer), 4)
    if "context_precision" not in scores:
        scores["context_precision"] = round(_prec_simple(question, contexts), 4)
    if "context_recall" not in scores:
        scores["context_recall"] = round(_recall_simple(question, contexts), 4)

    # 综合分（加权平均）
    weights = {"faithfulness": 0.3, "answer_relevancy": 0.3,
               "context_precision": 0.2, "context_recall": 0.2}
    w_sum = sum(weights.get(k, 0) for k in scores if k != "overall")
    if w_sum > 0:
        scores["overall"] = round(
            sum(scores.get(k, 0) * weights.get(k, 0) for k in scores if k != "overall") / w_sum, 4
        )
    else:
        scores["overall"] = 0.0

    return scores


# ==================== 评估入口 ====================

def evaluate_role(
    role: str,
    questions: List[str] = None,
    verbose: bool = True,
) -> Dict:
    from services.rag_service import call_rag_llm

    use_ragas = _RAGAS_AVAILABLE and EVAL_MODE == "ragas"
    eval_mode = "ragas_official" if use_ragas else "simple"

    if questions is None:
        questions = generate_test_questions(role)

    if not questions:
        logger.warning(f"[{role}] 没有测试问题")
        return {"role": role, "overall_score": 0.0, "metrics_summary": {}, "details": []}

    all_results = []
    metric_sums = {}
    metric_counts = {}
    total_start = time.time()

    for i, question in enumerate(questions):
        q_start = time.time()
        try:
            answer, contexts, meta = call_rag_llm(
                question=question, chat_history=[],
                role=role, user_id="eval_user",
            )
            if not answer or not contexts:
                logger.warning(f"[{role}] Q{i+1}: 无结果，跳过")
                continue

            scores = evaluate_single_qa(question, answer, contexts, use_ragas=use_ragas)
            q_time = time.time() - q_start

            all_results.append({
                "question": question,
                "answer": answer[:500] + ("..." if len(answer) > 500 else ""),
                "answer_full_length": len(answer),
                "contexts_count": len(contexts),
                "scores": scores,
                "time_seconds": round(q_time, 2),
            })

            for m, s in scores.items():
                if m != "overall":
                    metric_sums[m] = metric_sums.get(m, 0) + s
                    metric_counts[m] = metric_counts.get(m, 0) + 1

            if verbose:
                logger.info(
                    f"[{role}] Q{i+1}/{len(questions)}: {question[:40]}... "
                    f"faith={scores.get('faithfulness', 0):.2f} "
                    f"rel={scores.get('answer_relevancy', 0):.2f} "
                    f"prec={scores.get('context_precision', 0):.2f} "
                    f"rec={scores.get('context_recall', 0):.2f} "
                    f"({q_time:.1f}s)"
                )
        except Exception as e:
            logger.error(f"[{role}] Q{i+1} 失败: {e}")
            continue

    total_time = time.time() - total_start

    metrics_summary = {}
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric_counts.get(m, 0) > 0:
            metrics_summary[m] = round(metric_sums[m] / metric_counts[m], 4)

    overall = 0.0
    if metrics_summary:
        w = {"faithfulness": 0.3, "answer_relevancy": 0.3,
             "context_precision": 0.2, "context_recall": 0.2}
        tw = sum(w.get(k, 0) for k in metrics_summary)
        if tw > 0:
            overall = round(sum(metrics_summary[k] * w.get(k, 0) for k in metrics_summary) / tw, 4)

    report = {
        "role": role,
        "overall_score": overall,
        "metrics_summary": metrics_summary,
        "eval_mode": eval_mode,
        "details": all_results,
        "num_questions": len(all_results),
        "total_time_seconds": round(total_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = os.path.join(EVAL_RESULTS_DIR, f"{role}_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if verbose:
        logger.info(
            f"\n{'='*50}\n[{role}] 评估完成 ({eval_mode})\n"
            f"  综合: {overall:.4f} | 忠实: {metrics_summary.get('faithfulness', 0):.4f} "
            f"| 相关: {metrics_summary.get('answer_relevancy', 0):.4f} "
            f"| 精确: {metrics_summary.get('context_precision', 0):.4f} "
            f"| 召回: {metrics_summary.get('context_recall', 0):.4f} "
            f"| 耗时: {total_time:.1f}s\n报告: {report_path}\n{'='*50}"
        )

    return report


def evaluate_all_roles(roles: List[str] = None, verbose: bool = True) -> Dict[str, Dict]:
    ALL_ROLES = ["Medical", "Finance", "Law", "Education", "Psychology"]
    if roles is None:
        roles = ALL_ROLES

    results = {}
    for role in roles:
        if role not in ALL_ROLES:
            continue
        logger.info(f"\n{'#'*60}\n开始评估 [{role}]\n{'#'*60}")
        try:
            results[role] = evaluate_role(role, verbose=verbose)
        except Exception as e:
            logger.error(f"[{role}] 失败: {e}")
            results[role] = {"role": role, "overall_score": 0.0, "error": str(e)}

    summary = []
    for role, r in results.items():
        summary.append({
            "role": role,
            "overall_score": r.get("overall_score", 0),
            "faithfulness": r.get("metrics_summary", {}).get("faithfulness", 0),
            "answer_relevancy": r.get("metrics_summary", {}).get("answer_relevancy", 0),
            "context_precision": r.get("metrics_summary", {}).get("context_precision", 0),
            "context_recall": r.get("metrics_summary", {}).get("context_recall", 0),
            "eval_mode": r.get("eval_mode", "simple"),
            "num_questions": r.get("num_questions", 0),
        })

    summary_report = {
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = os.path.join(EVAL_RESULTS_DIR, f"all_roles_summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    if verbose:
        logger.info(f"\n{'='*70}")
        logger.info(f"{'角色':<15} {'综合':<8} {'忠实':<8} {'相关':<8} {'精确':<8} {'召回':<8} {'模式':<8} {'题数':<6}")
        logger.info(f"{'-'*70}")
        for s in summary:
            logger.info(
                f"{s['role']:<15} {s['overall_score']:<8.4f} {s['faithfulness']:<8.4f} "
                f"{s['answer_relevancy']:<8.4f} {s['context_precision']:<8.4f} "
                f"{s['context_recall']:<8.4f} {s['eval_mode']:<8} {s['num_questions']:<6}"
            )
        logger.info(f"{'='*70}")
        logger.info(f"汇总报告: {summary_path}")

    return results


# ==================== 自定义问答评分 ====================

def score_single_qa(question: str, role: str = "Medical"):
    """传一个问题，调用 RAG 后评分，打印结果后退出"""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # 静默日志

    from services.rag_service import call_rag_llm

    use_ragas = _RAGAS_AVAILABLE and EVAL_MODE == "ragas"

    print(f"\n[问题] {question}")
    print(f"[角色] {role}")

    answer, contexts, meta = call_rag_llm(
        question=question, chat_history=[],
        role=role, user_id="eval_user",
    )

    print(f"\n[回答] {answer[:600]}{'...' if len(answer) > 600 else ''}")
    print(f"[上下文] {len(contexts)} 条{'' if contexts else ' -- 跳过评分'}")

    if contexts:
        scores = evaluate_single_qa(question, answer, contexts, use_ragas=use_ragas)
        print(f"\n----- RAGAS 评分 -----")
        print(f"  faithfulness  (忠实性) : {scores.get('faithfulness', 0):.4f}")
        print(f"  answer_relevancy(相关性): {scores.get('answer_relevancy', 0):.4f}")
        print(f"  context_precision(精确度): {scores.get('context_precision', 0):.4f}")
        print(f"  context_recall (召回率) : {scores.get('context_recall', 0):.4f}")
        print(f"  >>> 综合: {scores.get('overall', 0):.4f}")
        print(f"-----------------------")
    else:
        print("  (无上下文，不评分)")

    kb_hit = meta.get("kb_hit", False)
    print(f"[知识库] {'命中' if kb_hit else '未命中'}")


# ==================== CLI ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAGAS 评估系统")
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--questions", type=int, default=QUESTIONS_PER_ROLE)
    parser.add_argument("--mode", type=str, default="ragas", choices=["ragas", "simple"])
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("-q", "--query", type=str, help="自定义问题，调用 RAG + 评分后退出")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    import services.ragas_evaluator as ev
    ev.EVAL_MODE = args.mode

    if args.query:
        score_single_qa(args.query, role=args.role or "Medical")
        return

    if args.questions:
        ev.QUESTIONS_PER_ROLE = args.questions

    if args.role:
        evaluate_role(args.role, verbose=args.verbose)
    else:
        evaluate_all_roles(verbose=args.verbose)


if __name__ == "__main__":
    main()
