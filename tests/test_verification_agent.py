"""
Unit tests for the VerificationAgent (LLM & HF Token BERTScore Validator).
"""
import os
import pytest

from src.agentic_graphrag.agents.verification_agent import (
    CheckResult,
    VerificationAgent,
    VerificationReport,
)


def test_agent_environment_check():
    agent = VerificationAgent(verbose=False)
    res = agent.verify_environment()
    assert isinstance(res, CheckResult)
    assert res.status in ("PASSED", "WARNING")
    assert "llm_provider" in res.details
    assert "has_hf_token" in res.details


def test_agent_huggingface_check():
    agent = VerificationAgent(verbose=False)
    res = agent.verify_huggingface_token()
    assert isinstance(res, CheckResult)
    if os.getenv("HF_TOKEN"):
        assert res.status == "PASSED"
        assert res.details.get("authenticated") is True
        assert "hf_user" in res.details
    else:
        assert res.status == "WARNING"


def test_agent_bert_scoring_check():
    agent = VerificationAgent(verbose=False)
    res = agent.verify_bert_scoring()
    assert isinstance(res, CheckResult)
    assert res.status == "PASSED"
    assert res.details["identical_match"]["bert_f1"] == 1.0
    assert res.details["paraphrase_match"]["bert_f1"] > 0.65
    assert res.details["dissimilar_match"]["bert_f1"] < 0.50


def test_agent_run_all():
    agent = VerificationAgent(verbose=False)
    report = agent.run_all()
    assert isinstance(report, VerificationReport)
    assert report.overall_status in ("HEALTHY", "DEGRADED")
    assert len(report.checks) >= 4
    # Ensure trial result was computed
    if report.overall_status == "HEALTHY":
        assert report.trial_result is not None
        assert "bert_f1" in report.trial_result
        assert report.trial_result["bert_f1"] > 0.0
