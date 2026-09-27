"""
Unit tests for the BERTScore evaluation module and metrics engine.
"""
from src.benchmark.bert_scorer import compute_bert_score, compute_bert_score_batch
from src.benchmark import metrics


def test_bert_score_identical_strings():
    res = compute_bert_score("Chen Ding", "Chen Ding")
    assert res["bert_f1"] == 1.0
    assert res["bert_precision"] == 1.0
    assert res["bert_recall"] == 1.0


def test_bert_score_semantic_overlap():
    res = compute_bert_score("Chen Ding won gold in 20km walk", "Chen Ding")
    assert res["bert_recall"] > 0.6
    assert res["bert_f1"] > 0.0


def test_bert_score_empty():
    res = compute_bert_score("", "")
    assert res["bert_f1"] == 0.0


def test_metrics_judge_answer_includes_bert():
    out = metrics.judge_answer(
        question="Who won gold?",
        gold_answer="Naim Süleymanoğlu",
        candidate_answer="Naim Süleymanoğlu",
        compute_bert=True,
    )
    assert "bert_f1" in out
    assert "accuracy" in out
    assert "completeness" in out
    assert out["bert_f1"] == 1.0


def test_aggregate_scores_with_bert():
    sample_rows = [
        {
            "pipeline": "agentic_graphrag",
            "accuracy": 1.0,
            "completeness": 1.0,
            "bert_f1": 0.95,
            "bert_precision": 0.96,
            "bert_recall": 0.94,
            "tokens_used": 120,
            "latency_seconds": 1.5,
        }
    ]
    summary = metrics.aggregate_scores(sample_rows)
    assert "agentic_graphrag" in summary
    assert summary["agentic_graphrag"]["mean_bert_f1"] == 0.95
