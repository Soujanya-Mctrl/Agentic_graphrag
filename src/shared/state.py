"""
The agent harness: the one piece of shared, mutable state every agent
reads from and writes to. This is what makes the system "agentic" rather
than a fixed pipeline — the orchestrator decides its next move by
inspecting this object, not by following a hardcoded sequence.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ActionType(str, Enum):
    ENTITY_LINK = "entity_link"
    GRAPH_TRAVERSE = "graph_traverse"
    VECTOR_SEARCH = "vector_search"
    DOCUMENT_RETRIEVE = "document_retrieve"
    AGGREGATE = "aggregate"
    MULTI_HOP_REASON = "multi_hop_reason"
    EVALUATE_EVIDENCE = "evaluate_evidence"
    ANSWER = "answer"  # terminal action — orchestrator is done


@dataclass
class EvidenceItem:
    """One atomic piece of retrieved evidence, tagged with its provenance."""
    id: str
    source_action: ActionType
    content: str
    source_ref: str          # node id / doc id / query string that produced it
    confidence: float = 1.0  # source-reported confidence, not overall answer confidence
    contradicts: list[str] = field(default_factory=list)  # ids of conflicting evidence
    timestamp: float = field(default_factory=time.time)


@dataclass
class InvestigationStep:
    """One orchestrator decision + its result. Kept for the audit trail
    the benchmark's 'evidence quality and explainability' score needs."""
    step_index: int
    action: ActionType
    action_input: dict[str, Any]
    rationale: str                     # why the orchestrator picked this action
    evidence_ids: list[str] = field(default_factory=list)
    tokens_used: int = 0


@dataclass
class InvestigationState:
    """The full mutable state for answering ONE question. Create a fresh
    instance per question; never share across questions."""
    question: str
    question_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    evidence: dict[str, EvidenceItem] = field(default_factory=dict)
    steps: list[InvestigationStep] = field(default_factory=list)
    linked_entities: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)  # orchestrator's running "what's missing" notes
    stopped: bool = False
    stop_reason: str = ""
    final_answer: str = ""
    final_confidence: float = 0.0
    total_tokens: int = 0

    # -- convenience API used by agents/orchestrator --

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence[item.id] = item

    def evidence_list(self) -> list[EvidenceItem]:
        return list(self.evidence.values())

    def record_step(self, step: InvestigationStep) -> None:
        self.steps.append(step)
        self.total_tokens += step.tokens_used

    def evidence_summary(self, max_chars: int = 12000) -> str:
        """Condensed view of everything gathered so far, for feeding back
        into the orchestrator's next-action prompt. Truncated to keep
        token usage bounded as investigations get longer."""
        lines = []
        for ev in self.evidence_list():
            lines.append(f"[{ev.id}] ({ev.source_action.value}) {ev.content}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text or "(no evidence gathered yet)"

    def should_force_stop(self, max_steps: int) -> bool:
        return len(self.steps) >= max_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "final_answer": self.final_answer,
            "final_confidence": self.final_confidence,
            "stop_reason": self.stop_reason,
            "total_tokens": self.total_tokens,
            "num_steps": len(self.steps),
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": s.action.value,
                    "action_input": s.action_input,
                    "rationale": s.rationale,
                    "evidence_ids": s.evidence_ids,
                    "tokens_used": s.tokens_used,
                }
                for s in self.steps
            ],
            "evidence": [
                {
                    "id": e.id,
                    "source_action": e.source_action.value,
                    "content": e.content,
                    "source_ref": e.source_ref,
                    "confidence": e.confidence,
                    "contradicts": e.contradicts,
                }
                for e in self.evidence_list()
            ],
        }
