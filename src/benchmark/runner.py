"""
Entry point for producing the metrics dashboard artifact the submission
requires. Run with: python -m src.benchmark.runner
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..shared.config import CONFIG
from ..shared.tigergraph_client import get_client
from ..rag import pipeline as naive_rag
from ..graphrag import pipeline as graph_rag
from ..agentic_graphrag import pipeline as agentic_graphrag
from . import metrics


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_all(dataset_path: str | None = None, results_dir: str | None = None) -> dict:
    dataset_path = dataset_path or CONFIG.dataset_path
    results_dir = results_dir or CONFIG.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    questions = load_dataset(dataset_path)
    client = get_client()

    per_question_results = []
    raw_runs = []

    for q in questions:
        question_text = q["question"]
        gold = q.get("gold_answer", "")

        naive_out = naive_rag.run(question_text, client)
        graph_out = graph_rag.run(question_text, client)
        agentic_out = agentic_graphrag.run(question_text, client)

        for out in (naive_out, graph_out, agentic_out):
            judged = metrics.judge_answer(question_text, gold, out["answer"]) if gold else {
                "accuracy": None, "completeness": None, "justification": "no gold answer provided"
            }
            row = {
                "question_id": q.get("id", question_text[:20]),
                "question": question_text,
                "pipeline": out["pipeline"],
                "answer": out["answer"],
                "tokens_used": out["tokens_used"],
                "latency_seconds": out["latency_seconds"],
                "accuracy": judged["accuracy"],
                "completeness": judged["completeness"],
                "judge_justification": judged["justification"],
            }
            per_question_results.append(row)
            raw_runs.append(out)

    # only aggregate rows that got judged (gold answer available)
    judged_rows = [r for r in per_question_results if r["accuracy"] is not None]
    summary = metrics.aggregate_scores(judged_rows) if judged_rows else {}

    output = {
        "summary": summary,
        "per_question_results": per_question_results,
    }

    with open(os.path.join(results_dir, "benchmark_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    with open(os.path.join(results_dir, "raw_runs.json"), "w") as f:
        json.dump(raw_runs, f, indent=2)

    return output


if __name__ == "__main__":
    result = run_all()
    print(json.dumps(result["summary"], indent=2))
