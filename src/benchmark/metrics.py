"""
Scoring for the mandatory three-way comparison.
Combines:
  1. LLM-as-a-Judge (Accuracy and Completeness against Gold Reference)
  2. BERTScore (Precision, Recall, F1 for token-level semantic match)
  3. Token Efficiency (Baseline / Candidate token ratio)
  4. Latency and Step metrics
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..shared.llm import complete_json
from .bert_scorer import compute_bert_score

JUDGE_SYSTEM = (
    "You are grading a candidate answer against a gold reference answer for "
    "a graph-investigation QA benchmark. Score strictly and consistently "
    "across candidates so scores are comparable."
)


def judge_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    compute_bert: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates candidate answer against gold answer using both LLM-as-a-judge
    and BERTScore.
    """
    cand = str(candidate_answer).strip()
    gold = str(gold_answer).strip()

    prompt = (
        f"Question: {question}\n"
        f"Gold reference answer: {gold}\n"
        f"Candidate answer: {cand}\n\n"
        "Score the candidate on two axes, each 0.0-1.0:\n"
        "- accuracy: does it state the correct facts, with no contradictions of the gold answer?\n"
        "- completeness: does it cover every part of what the gold answer covers, "
        "not just a subset?\n\n"
        'Return JSON: {"accuracy": float, "completeness": float, "justification": str}'
    )

    parsed, _ = complete_json(JUDGE_SYSTEM, prompt)
    accuracy = float(parsed.get("accuracy", 0.0))
    completeness = float(parsed.get("completeness", 0.0))
    justification = str(parsed.get("justification", ""))

    bert_metrics = {
        "bert_precision": 0.0,
        "bert_recall": 0.0,
        "bert_f1": 0.0,
    }
    if compute_bert and gold:
        bert_metrics = compute_bert_score(cand, gold)

    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "justification": justification,
        **bert_metrics,
    }


def token_efficiency(tokens_used: int, baseline_tokens: int) -> float:
    """
    1.0 = as efficient as the cheapest baseline; >1 means more efficient,
    <1 means it cost more tokens than the baseline naive_rag run for the
    same question. Reported as a ratio (baseline / candidate).
    """
    if tokens_used == 0:
        return float("inf")
    return round(baseline_tokens / tokens_used, 4)


def aggregate_scores(per_question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes per-pipeline summary statistics across all evaluated questions.
    """
    by_pipeline: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_question_results:
        by_pipeline.setdefault(r["pipeline"], []).append(r)

    summary = {}
    for pipeline, rows in by_pipeline.items():
        n = len(rows)
        if n == 0:
            continue

        acc_rows = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
        comp_rows = [r["completeness"] for r in rows if r.get("completeness") is not None]
        bert_f1_rows = [r["bert_f1"] for r in rows if r.get("bert_f1") is not None]
        bert_p_rows = [r["bert_precision"] for r in rows if r.get("bert_precision") is not None]
        bert_r_rows = [r["bert_recall"] for r in rows if r.get("bert_recall") is not None]

        summary[pipeline] = {
            "n_questions": n,
            "mean_accuracy": round(sum(acc_rows) / len(acc_rows), 4) if acc_rows else 0.0,
            "mean_completeness": round(sum(comp_rows) / len(comp_rows), 4) if comp_rows else 0.0,
            "mean_bert_f1": round(sum(bert_f1_rows) / len(bert_f1_rows), 4) if bert_f1_rows else 0.0,
            "mean_bert_precision": round(sum(bert_p_rows) / len(bert_p_rows), 4) if bert_p_rows else 0.0,
            "mean_bert_recall": round(sum(bert_r_rows) / len(bert_r_rows), 4) if bert_r_rows else 0.0,
            "mean_tokens": round(sum(r.get("tokens_used", 0) for r in rows) / n, 1),
            "total_tokens": sum(r.get("tokens_used", 0) for r in rows),
            "mean_latency_seconds": round(sum(r.get("latency_seconds", 0) for r in rows) / n, 2),
        }

    return summary
