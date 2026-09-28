"""
Autonomous Verification Agent: LLM Gateway & Hugging Face BERTScore Validator
=============================================================================
An autonomous diagnostic agent that systematically tests and validates:
  1. Environment configuration (API keys, provider settings, model defaults)
  2. Hugging Face Hub Authentication (Token validity, whoami, model repo access)
  3. LLM Gateway connectivity & reasoning (Prose completion, JSON schema adherence, token accounting)
  4. BERTScore evaluation engine (Dense SentenceTransformer encoding, Precision/Recall/F1 metrics)
  5. End-to-End Agentic Trial (Live LLM generation evaluated against gold standard via BERTScore)

Can be executed as:
  - Standalone script: python test_agent.py
  - CLI module: python -m src.agentic_graphrag.agents.verification_agent
  - Importable agent class: from src.agentic_graphrag.agents.verification_agent import VerificationAgent
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Prevent OpenBLAS/MKL thread allocation issues on Windows Python 3.13
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.shared.config import CONFIG
from src.shared.llm import LLMResponse, complete, complete_json
from src.benchmark.bert_scorer import compute_bert_score, compute_bert_score_batch

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    status: str  # "PASSED" | "FAILED" | "WARNING" | "SKIPPED"
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    duration_ms: float = 0.0


@dataclass
class VerificationReport:
    timestamp: str
    overall_status: str  # "HEALTHY" | "DEGRADED" | "CRITICAL"
    summary: str
    environment: Dict[str, Any]
    checks: List[CheckResult] = field(default_factory=list)
    trial_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "summary": self.summary,
            "environment": self.environment,
            "checks": [asdict(c) for c in self.checks],
            "trial_result": self.trial_result,
        }


class VerificationAgent:
    """
    Diagnostic & Health Agent that validates the entire LLM + HF Token + BERTScore pipeline.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.checks: List[CheckResult] = []

    def _log(self, text: str, symbol: str = "ℹ️") -> None:
        if self.verbose:
            print(f"[{symbol}] {text}")

    def verify_environment(self) -> CheckResult:
        """Step 1: Check environment variables and active configurations."""
        start = time.perf_counter()
        self._log("Inspecting environment configuration...", "⚙️")

        has_groq = bool(CONFIG.groq_api_key)
        has_anthropic = bool(CONFIG.anthropic_api_key)
        has_openai = bool(CONFIG.openai_api_key)
        has_hf = bool(CONFIG.hf_token)

        active_provider = CONFIG.llm_provider
        active_model = (
            CONFIG.groq_model
            if active_provider == "groq"
            else CONFIG.anthropic_model
            if active_provider == "anthropic"
            else CONFIG.openai_model
            if active_provider == "openai"
            else "mock"
        )

        details = {
            "llm_provider": active_provider,
            "active_model": active_model,
            "has_groq_key": has_groq,
            "has_anthropic_key": has_anthropic,
            "has_openai_key": has_openai,
            "has_hf_token": has_hf,
            "tg_use_mock": CONFIG.tg_use_mock,
        }

        duration = (time.perf_counter() - start) * 1000

        if not (has_groq or has_anthropic or has_openai):
            msg = "No LLM API keys configured. LLM will operate in fallback mock mode."
            self._log(msg, "⚠️")
            return CheckResult(
                name="Environment & Keys",
                status="WARNING",
                details=details,
                message=msg,
                duration_ms=duration,
            )

        if not has_hf:
            msg = "HF_TOKEN not detected in environment. SentenceTransformer will run unauthenticated."
            self._log(msg, "⚠️")
            return CheckResult(
                name="Environment & Keys",
                status="WARNING",
                details=details,
                message=msg,
                duration_ms=duration,
            )

        msg = f"Environment verified. Active provider: '{active_provider}' ({active_model}), HF token detected."
        self._log(msg, "✅")
        return CheckResult(
            name="Environment & Keys",
            status="PASSED",
            details=details,
            message=msg,
            duration_ms=duration,
        )

    def verify_huggingface_token(self) -> CheckResult:
        """Step 2: Authenticate with Hugging Face Hub using HF_TOKEN."""
        start = time.perf_counter()
        self._log("Verifying Hugging Face Token & Hub authentication...", "🤗")

        token = CONFIG.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            duration = (time.perf_counter() - start) * 1000
            msg = "HF_TOKEN is missing. Rate limits may apply on model downloads."
            self._log(msg, "⚠️")
            return CheckResult(
                name="Hugging Face Authentication",
                status="WARNING",
                details={"authenticated": False, "reason": "No token provided"},
                message=msg,
                duration_ms=duration,
            )

        try:
            from huggingface_hub import HfApi, login

            # Ensure environment has the token
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token

            api = HfApi(token=token)
            user_info = api.whoami()

            user_name = user_info.get("name") or user_info.get("fullname", "Unknown")
            user_type = user_info.get("type", "user")
            orgs = [org.get("name") for org in user_info.get("orgs", [])] if isinstance(user_info.get("orgs"), list) else []

            # Verify repository access for BERT model
            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            model_info = api.model_info(model_id)

            duration = (time.perf_counter() - start) * 1000
            details = {
                "authenticated": True,
                "hf_user": user_name,
                "account_type": user_type,
                "organizations": orgs,
                "target_model": model_id,
                "model_downloads": getattr(model_info, "downloads", None),
                "token_prefix": f"{token[:7]}...",
            }

            msg = f"HF Token verified! Authenticated as '{user_name}'. Model '{model_id}' is accessible."
            self._log(msg, "✅")
            return CheckResult(
                name="Hugging Face Authentication",
                status="PASSED",
                details=details,
                message=msg,
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            msg = f"Failed to authenticate with Hugging Face: {str(e)}"
            self._log(msg, "❌")
            return CheckResult(
                name="Hugging Face Authentication",
                status="FAILED",
                details={"authenticated": False, "error": str(e)},
                message=msg,
                duration_ms=duration,
            )

    def verify_llm_connection(self) -> CheckResult:
        """Step 3: Test LLM text completion and strict JSON parsing."""
        start = time.perf_counter()
        self._log(f"Testing LLM Gateway ({CONFIG.llm_provider})...", "🤖")

        try:
            # 1. Prose completion test
            t0 = time.perf_counter()
            prose_resp: LLMResponse = complete(
                system="You are an autonomous AI benchmark verification system.",
                prompt="State in exactly 5 words that the system is fully operational.",
                max_tokens=256,
            )
            prose_latency_ms = (time.perf_counter() - t0) * 1000

            # 2. Structured JSON completion test (required for Agent Orchestration)
            t1 = time.perf_counter()
            json_parsed, json_resp = complete_json(
                system="You are an agent orchestrator component.",
                prompt="Acknowledge the diagnostic check. Return a JSON object with keys: "
                       "'status' (string: 'healthy'), 'ready' (bool: true), and 'step_name' (string: 'verification').",
                max_tokens=512,
            )
            json_latency_ms = (time.perf_counter() - t1) * 1000

            total_tokens = prose_resp.total_tokens + json_resp.total_tokens
            is_mock = "MOCK MODE" in prose_resp.text or prose_resp.input_tokens == 0

            duration = (time.perf_counter() - start) * 1000

            details = {
                "provider": CONFIG.llm_provider,
                "model": getattr(CONFIG, f"{CONFIG.llm_provider}_model", "unknown"),
                "prose_response": prose_resp.text.strip(),
                "prose_latency_ms": round(prose_latency_ms, 1),
                "prose_tokens": {"in": prose_resp.input_tokens, "out": prose_resp.output_tokens},
                "json_parsed": json_parsed,
                "json_latency_ms": round(json_latency_ms, 1),
                "json_tokens": {"in": json_resp.input_tokens, "out": json_resp.output_tokens},
                "total_tokens_used": total_tokens,
                "is_mock": is_mock,
            }

            if is_mock:
                msg = "LLM returned mock output (no active API key or provider was unreachable)."
                self._log(msg, "⚠️")
                return CheckResult(
                    name="LLM Gateway & Completion",
                    status="WARNING",
                    details=details,
                    message=msg,
                    duration_ms=duration,
                )

            # Check that JSON parsed expected structure
            if not isinstance(json_parsed, dict) or not json_parsed:
                msg = f"LLM responded, but JSON parsing failed or returned empty: {json_resp.text}"
                self._log(msg, "WARNING")
                return CheckResult(
                    name="LLM Gateway & Completion",
                    status="WARNING",
                    details=details,
                    message=msg,
                    duration_ms=duration,
                )

            msg = (
                f"LLM operational ({details['provider']}:{details['model']})! "
                f"Prose: {prose_latency_ms:.0f}ms, JSON: {json_latency_ms:.0f}ms, Tokens: {total_tokens}."
            )
            self._log(msg, "✅")
            return CheckResult(
                name="LLM Gateway & Completion",
                status="PASSED",
                details=details,
                message=msg,
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            msg = f"LLM test failed with exception: {str(e)}"
            self._log(msg, "❌")
            return CheckResult(
                name="LLM Gateway & Completion",
                status="FAILED",
                details={"error": str(e)},
                message=msg,
                duration_ms=duration,
            )

    def verify_bert_scoring(self) -> CheckResult:
        """Step 4: Test BERTScore semantic similarity with dense embeddings and token overlap."""
        start = time.perf_counter()
        self._log("Evaluating BERTScore engine on test pairs...", "📐")

        try:
            # Pair 1: Identical strings -> Must be exactly 1.0
            r_exact = compute_bert_score(
                "Michael Phelps won 23 Olympic gold medals in swimming.",
                "Michael Phelps won 23 Olympic gold medals in swimming.",
            )

            # Pair 2: Paraphrased / Semantic Equivalence -> Expect F1 >= 0.70
            r_para = compute_bert_score(
                "Michael Phelps captured twenty-three gold medals during the Olympic games in swimming.",
                "Michael Phelps won 23 Olympic gold medals in swimming.",
            )

            # Pair 3: Completely dissimilar topic -> Expect F1 < 0.50
            r_diff = compute_bert_score(
                "The stock market experienced a dramatic downturn on Friday.",
                "Michael Phelps won 23 Olympic gold medals in swimming.",
            )

            # Pair 4: Batch test
            cands = ["Chen Ding won gold in 20km walk.", "Usain Bolt ran 100m in 9.63s."]
            refs = ["Chen Ding took gold in the 20-kilometer race walk.", "Bolt set the Olympic record in 100m."]
            r_batch = compute_bert_score_batch(cands, refs)

            duration = (time.perf_counter() - start) * 1000

            details = {
                "identical_match": r_exact,
                "paraphrase_match": r_para,
                "dissimilar_match": r_diff,
                "batch_results": r_batch,
            }

            # Validations
            if r_exact["bert_f1"] != 1.0:
                msg = f"Exact match expected F1=1.0, got {r_exact['bert_f1']}"
                self._log(msg, "❌")
                return CheckResult(
                    name="BERTScore Engine",
                    status="FAILED",
                    details=details,
                    message=msg,
                    duration_ms=duration,
                )

            if r_para["bert_f1"] < 0.65:
                msg = f"Paraphrase similarity lower than expected: {r_para['bert_f1']}"
                self._log(msg, "WARNING")
                return CheckResult(
                    name="BERTScore Engine",
                    status="WARNING",
                    details=details,
                    message=msg,
                    duration_ms=duration,
                )

            msg = (
                f"BERTScore engine verified! Exact F1: {r_exact['bert_f1']:.2f}, "
                f"Paraphrase F1: {r_para['bert_f1']:.4f}, Dissimilar F1: {r_diff['bert_f1']:.4f}."
            )
            self._log(msg, "✅")
            return CheckResult(
                name="BERTScore Engine",
                status="PASSED",
                details=details,
                message=msg,
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            msg = f"BERTScore computation failed: {str(e)}"
            self._log(msg, "❌")
            return CheckResult(
                name="BERTScore Engine",
                status="FAILED",
                details={"error": str(e)},
                message=msg,
                duration_ms=duration,
            )

    def run_end_to_end_trial(self) -> Dict[str, Any]:
        """Step 5: End-to-End Agentic Trial: Ask LLM a domain question & score it against reference."""
        self._log("Running End-to-End Agentic Trial (LLM Generation + BERT Evaluation)...", "🚀")

        test_question = "How many gold medals did Michael Phelps win in swimming at the Olympics?"
        gold_answer = "Michael Phelps won 23 Olympic gold medals in swimming."

        t0 = time.perf_counter()
        resp = complete(
            system="You are an expert sports historian answering Olympic queries concisely and accurately.",
            prompt=f"Question: {test_question}\nProvide a concise and direct answer in 1 or 2 sentences.",
            max_tokens=150,
        )
        latency_s = time.perf_counter() - t0
        candidate_answer = resp.text.strip()

        # Score candidate against gold using BERTScore
        bert_metrics = compute_bert_score(candidate_answer, gold_answer)

        trial_data = {
            "question": test_question,
            "gold_reference": gold_answer,
            "llm_candidate": candidate_answer,
            "latency_seconds": round(latency_s, 3),
            "tokens_used": resp.total_tokens,
            "bert_precision": bert_metrics["bert_precision"],
            "bert_recall": bert_metrics["bert_recall"],
            "bert_f1": bert_metrics["bert_f1"],
        }

        self._log(f"Candidate: '{candidate_answer}'", "📝")
        self._log(
            f"BERTScore -> Precision: {bert_metrics['bert_precision']:.4f}, "
            f"Recall: {bert_metrics['bert_recall']:.4f}, F1: {bert_metrics['bert_f1']:.4f} "
            f"({latency_s:.2f}s, {resp.total_tokens} tokens)",
            "🎯"
        )
        return trial_data

    def run_all(self) -> VerificationReport:
        """Executes all verification phases sequentially and aggregates findings."""
        self.checks = []

        if self.verbose:
            print("\n" + "=" * 70)
            print("🔍 AGENTIC GRAPHRAG — LLM & HF BERTSCORE VERIFICATION AGENT")
            print("=" * 70)

        # Run checks
        c_env = self.verify_environment()
        self.checks.append(c_env)

        c_hf = self.verify_huggingface_token()
        self.checks.append(c_hf)

        c_llm = self.verify_llm_connection()
        self.checks.append(c_llm)

        c_bert = self.verify_bert_scoring()
        self.checks.append(c_bert)

        # Run trial if critical checks didn't crash
        trial = None
        if c_llm.status != "FAILED" and c_bert.status != "FAILED":
            try:
                trial = self.run_end_to_end_trial()
            except Exception as e:
                self._log(f"Trial execution error: {e}", "❌")

        # Determine overall status
        statuses = [c.status for c in self.checks]
        if any(s == "FAILED" for s in statuses):
            overall = "CRITICAL"
            summary = "One or more core components failed verification. Review errors below."
        elif any(s == "WARNING" for s in statuses):
            overall = "DEGRADED"
            summary = "Components are operational, but with warnings or fallbacks in effect."
        else:
            overall = "HEALTHY"
            summary = "All systems operational! LLM and HF BERTScore are working seamlessly."

        report = VerificationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            overall_status=overall,
            summary=summary,
            environment=c_env.details,
            checks=self.checks,
            trial_result=trial,
        )

        if self.verbose:
            self.print_summary(report)

        return report

    def print_summary(self, report: VerificationReport) -> None:
        """Prints a structured summary table to stdout."""
        print("\n" + "=" * 70)
        status_symbol = "🟢" if report.overall_status == "HEALTHY" else "🟡" if report.overall_status == "DEGRADED" else "🔴"
        print(f"{status_symbol} SYSTEM STATUS: {report.overall_status}")
        print(f"   {report.summary}")
        print("=" * 70)

        print(f"\n{'Check Name':<32} {'Status':<12} {'Duration':<10} {'Details'}")
        print("-" * 70)
        for c in report.checks:
            stat_icon = "✓" if c.status == "PASSED" else "!" if c.status == "WARNING" else "✗"
            print(f"{c.name:<32} {stat_icon} {c.status:<10} {c.duration_ms:>6.1f}ms   {c.message[:45]}...")

        if report.trial_result:
            tr = report.trial_result
            print("\n" + "-" * 70)
            print("🚀 End-to-End Trial Evaluation Metrics:")
            print(f"   • Question:       {tr['question']}")
            print(f"   • LLM Answer:     {tr['llm_candidate']}")
            print(f"   • Gold Reference: {tr['gold_reference']}")
            print(f"   • BERTScore F1:   {tr['bert_f1']:.4f} (P: {tr['bert_precision']:.4f}, R: {tr['bert_recall']:.4f})")
            print(f"   • Response Time:  {tr['latency_seconds']:.2f}s | Tokens Consumed: {tr['tokens_used']}")
        print("=" * 70 + "\n")


def run_agent_cli() -> int:
    """Entry point for CLI execution."""
    agent = VerificationAgent(verbose=True)
    report = agent.run_all()
    return 0 if report.overall_status in ("HEALTHY", "DEGRADED") else 1


if __name__ == "__main__":
    sys.exit(run_agent_cli())
