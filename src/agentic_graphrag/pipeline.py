"""
The system this hackathon is actually about. Everything else in
pipelines/ exists to give this one something to be measured against.
"""
from __future__ import annotations

import time

from .agents.orchestrator import run_investigation
from ..shared.state import InvestigationState


def run(question: str, client) -> dict:
    start = time.time()
    state = InvestigationState(question=question)
    state = run_investigation(state, client)

    return {
        "pipeline": "agentic_graphrag",
        "question": question,
        "answer": state.final_answer,
        "tokens_used": state.total_tokens,
        "evidence_count": len(state.evidence),
        "latency_seconds": time.time() - start,
        "num_steps": len(state.steps),
        "stop_reason": state.stop_reason,
        "confidence": state.final_confidence,
        "full_trace": state.to_dict(),  # for the explainability/audit-trail score
    }
