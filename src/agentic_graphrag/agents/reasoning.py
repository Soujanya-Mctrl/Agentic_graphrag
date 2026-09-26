"""
Reasoning-side specialized agents. These call the LLM to synthesize,
connect, or judge evidence already sitting in InvestigationState —
they never fetch new evidence themselves (that's the retrieval agents' job).
"""
from __future__ import annotations

from ...shared.llm import complete_json
from ...shared.state import InvestigationState


AGGREGATE_SYSTEM = (
    "You aggregate evidence fragments gathered during a graph investigation. "
    "Merge overlapping facts, note contradictions explicitly, and produce a "
    "single concise synthesis. Never invent facts not present in the evidence."
)

MULTIHOP_SYSTEM = (
    "You perform multi-hop reasoning over investigation evidence to answer "
    "part of a larger question. Chain facts together explicitly and say "
    "which evidence ids you used for each hop. If the chain is incomplete, "
    "say exactly what link is missing."
)

EVALUATE_SYSTEM = (
    "You are a strict evidence auditor for an investigation agent. Given the "
    "original question and the evidence gathered so far, decide: (1) is this "
    "evidence sufficient to answer confidently, (2) what specific gap remains "
    "if not, (3) are there any contradictions between evidence items. Be "
    "conservative — prefer flagging a gap over guessing."
)


def aggregation_agent(state: InvestigationState) -> dict:
    prompt = (
        f"Question: {state.question}\n\n"
        f"Evidence gathered so far:\n{state.evidence_summary()}\n\n"
        "Return JSON: {\"synthesis\": str, \"contradictions\": [str]}"
    )
    parsed, resp = complete_json(AGGREGATE_SYSTEM, prompt)
    state.total_tokens += resp.total_tokens
    return parsed


def multi_hop_reasoning_agent(state: InvestigationState, sub_question: str) -> dict:
    prompt = (
        f"Original question: {state.question}\n"
        f"Sub-question to chain through evidence: {sub_question}\n\n"
        f"Evidence gathered so far:\n{state.evidence_summary()}\n\n"
        "Return JSON: {\"chain\": [str], \"evidence_ids_used\": [str], "
        "\"conclusion\": str, \"complete\": bool, \"missing_link\": str}"
    )
    parsed, resp = complete_json(MULTIHOP_SYSTEM, prompt)
    state.total_tokens += resp.total_tokens
    return parsed


def evidence_evaluation_agent(state: InvestigationState) -> dict:
    prompt = (
        f"Question: {state.question}\n\n"
        f"Evidence gathered so far ({len(state.evidence)} items):\n{state.evidence_summary()}\n\n"
        "Return JSON: {\"sufficient\": bool, \"confidence\": float (0-1), "
        "\"gap\": str, \"contradictions\": [[str, str]]}"
    )
    parsed, resp = complete_json(EVALUATE_SYSTEM, prompt)
    state.total_tokens += resp.total_tokens
    return parsed
