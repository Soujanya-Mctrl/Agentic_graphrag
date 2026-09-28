"""
Benchmark Runner
================
Runs Naive RAG, Fixed GraphRAG, and Agentic GraphRAG across datasets
(supporting both .jsonl and .json formats).
Scores answers with LLM-as-a-judge and BERTScore.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.shared.config import CONFIG
from src.shared.tigergraph_client import get_client
from src.rag import pipeline as naive_rag
from src.graphrag import pipeline as graph_rag
from src.agentic_graphrag import pipeline as agentic_graphrag
from src.benchmark import metrics


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Loads benchmark questions from .json or .jsonl.
    Normalizes keys:
      - 'id' or 'qid' -> 'id'
      - 'question' -> 'question'
      - 'gold_answer' or 'answer' -> 'gold_answer'
      - 'qtype' -> 'qtype' (e.g. aggregation, temporal, multi_hop)
    """
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "questions", "eval_public.jsonl")
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at: {path}")

    records = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]

    normalized = []
    for idx, item in enumerate(records):
        q_id = str(item.get("id") or item.get("qid") or f"q-{idx+1:03d}")
        q_text = str(item.get("question", "")).strip()

        raw_gold = item.get("gold_answer") or item.get("answer", "")
        if isinstance(raw_gold, list):
            gold_str = ", ".join(str(x) for x in raw_gold)
        else:
            gold_str = str(raw_gold).strip()

        qtype = str(item.get("qtype") or item.get("complexity") or "general")

        normalized.append({
            "id": q_id,
            "question": q_text,
            "gold_answer": gold_str,
            "qtype": qtype,
            "gold_doc_ids": item.get("gold_doc_ids", []),
        })

    return normalized


def run_single_question(
    q: Dict[str, Any],
    client: Any,
    pipelines: Optional[List[str]] = None,
    compute_judge: bool = True,
    compute_bert: bool = True,
) -> Dict[str, Any]:
    pipelines = pipelines or ["RAG", "GraphRAG", "Agentic GraphRAG"]
    q_id = q.get("id", "q")
    question_text = q["question"]
    gold = q.get("gold_answer", "")
    qtype = q.get("qtype", "general")

    results = {}
    raw_outputs = {}

    pipeline_runners = {
        "RAG": lambda: naive_rag.run(question_text, client),
        "GraphRAG": lambda: graph_rag.run(question_text, client),
        "Agentic GraphRAG": lambda: agentic_graphrag.run(question_text, client),
    }

    for name in pipelines:
        if name not in pipeline_runners:
            continue
        out = pipeline_runners[name]()
        raw_outputs[name] = out

        if compute_judge and gold:
            judged = metrics.judge_answer(question_text, gold, out["answer"], compute_bert=compute_bert)
        else:
            judged = {
                "accuracy": None,
                "completeness": None,
                "justification": "No gold answer or judge skipped",
                "bert_precision": None,
                "bert_recall": None,
                "bert_f1": None,
            }

        results[name] = {
            "question_id": q_id,
            "question": question_text,
            "qtype": qtype,
            "pipeline": out["pipeline"],
            "pipeline_name": name,
            "answer": out["answer"],
            "tokens_used": out.get("tokens_used", 0),
            "latency_seconds": round(out.get("latency_seconds", 0.0), 3),
            "evidence_count": out.get("evidence_count", 0),
            "num_steps": out.get("num_steps", 1),
            "accuracy": judged["accuracy"],
            "completeness": judged["completeness"],
            "justification": judged["justification"],
            "bert_precision": judged.get("bert_precision"),
            "bert_recall": judged.get("bert_recall"),
            "bert_f1": judged.get("bert_f1"),
        }

    return {
        "question_id": q_id,
        "question": question_text,
        "qtype": qtype,
        "gold_answer": gold,
        "pipeline_results": results,
        "raw_outputs": raw_outputs,
    }


def run_all(
    dataset_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    limit: Optional[int] = None,
    pipelines: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    compute_judge: bool = True,
    compute_bert: bool = True,
) -> Dict[str, Any]:
    dataset_path = dataset_path or CONFIG.dataset_path
    results_dir = results_dir or CONFIG.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    questions = load_dataset(dataset_path)
    if limit and limit > 0:
        questions = questions[:limit]

    client = get_client()
    per_question_results: List[Dict[str, Any]] = []
    raw_runs: List[Dict[str, Any]] = []

    total_q = len(questions)
    for idx, q in enumerate(questions):
        try:
            q_eval = run_single_question(
                q,
                client,
                pipelines=pipelines,
                compute_judge=compute_judge,
                compute_bert=compute_bert,
            )

            for _p_name, row in q_eval["pipeline_results"].items():
                per_question_results.append(row)

            for _p_name, raw in q_eval["raw_outputs"].items():
                raw_runs.append(raw)

            if progress_callback:
                progress_callback(idx + 1, total_q, q_eval)
        except Exception as q_err:
            import logging
            logging.getLogger("benchmark").error("Error on question %s: %s", q.get("id"), q_err)
            if progress_callback:
                progress_callback(idx + 1, total_q, {"question": f"Error: {q_err}"})

    judged_rows = [r for r in per_question_results if r["accuracy"] is not None]
    summary = metrics.aggregate_scores(judged_rows) if judged_rows else {}

    output = {
        "timestamp": time.time(),
        "total_questions": len(questions),
        "summary": summary,
        "per_question_results": per_question_results,
    }

    with open(os.path.join(results_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    with open(os.path.join(results_dir, "raw_runs.json"), "w", encoding="utf-8") as f:
        json.dump(raw_runs, f, indent=2, default=str)

    return output


if __name__ == "__main__":
    result = run_all(limit=5)
    print(json.dumps(result["summary"], indent=2))
