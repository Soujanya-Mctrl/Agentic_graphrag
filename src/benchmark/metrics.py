"""
Scoring for the mandatory three-way comparison. Accuracy and completeness
use an LLM-as-judge (documented in the guidebook's grading methodology as
one of the methods used on you, so it's reasonable to self-apply the same
approach). Token efficiency is just arithmetic — no judge needed.
"""
from __future__ import annotations

from ..shared.llm import complete_json

JUDGE_SYSTEM = (
    "You are grading a candidate answer against a gold reference answer for "
    "a graph-investigation QA benchmark. Score strictly and consistently "
    "across candidates so scores are comparable."
)


def judge_answer(question: str, gold_answer: str, candidate_answer: str) -> dict:
    prompt = (
        f"Question: {question}\n"
        f"Gold reference answer: {gold_answer}\n"
        f"Candidate answer: {candidate_answer}\n\n"
        "Score the candidate on two axes, each 0.0-1.0:\n"
        "- accuracy: does it state the correct facts, with no contradictions of the gold answer?\n"
        "- completeness: does it cover every part of what the gold answer covers, "
        "not just a subset?\n\n"
        'Return JSON: {"accuracy": float, "completeness": float, "justification": str}'
    )
    parsed, _ = complete_json(JUDGE_SYSTEM, prompt)
    return {
        "accuracy": float(parsed.get("accuracy", 0.0)),
        "completeness": float(parsed.get("completeness", 0.0)),
        "justification": parsed.get("justification", ""),
    }


def token_efficiency(tokens_used: int, baseline_tokens: int) -> float:
    """1.0 = as efficient as the cheapest baseline; >1 means more efficient,
    <1 means it cost more tokens than the baseline naive_rag run for the
    same question. Reported as a ratio (baseline / candidate), not a 0-1 score,
    since 'agentic costs more but is far more accurate' is a legitimate
    trade-off the dashboard should show plainly rather than penalize blindly."""
    if tokens_used == 0:
        return float("inf")
    return baseline_tokens / tokens_used


def aggregate_scores(per_question_results: list[dict]) -> dict:
    """per_question_results: list of {"pipeline": str, "accuracy": float,
    "completeness": float, "tokens_used": int, ...}. Returns per-pipeline
    means for the dashboard."""
    by_pipeline: dict[str, list[dict]] = {}
    for r in per_question_results:
        by_pipeline.setdefault(r["pipeline"], []).append(r)

    summary = {}
    for pipeline, rows in by_pipeline.items():
        n = len(rows)
        summary[pipeline] = {
            "n_questions": n,
            "mean_accuracy": sum(r["accuracy"] for r in rows) / n,
            "mean_completeness": sum(r["completeness"] for r in rows) / n,
            "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
            "mean_latency_seconds": sum(r.get("latency_seconds", 0) for r in rows) / n,
        }
    return summary
