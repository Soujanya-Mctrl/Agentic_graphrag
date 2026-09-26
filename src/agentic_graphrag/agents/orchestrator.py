"""
Agentic GraphRAG — LangGraph Orchestrator
==========================================
Replaces the hand-rolled while-loop with a proper LangGraph StateGraph.

Graph topology:
  orchestrate  -->  [entity_link | graph_traverse | vector_search |
                     document_retrieve | aggregate | multi_hop_reason |
                     evaluate_evidence]  --> orchestrate
  orchestrate  -->  finalize  -->  END

Each node reads from and writes to `GraphState` (a TypedDict that wraps
InvestigationState). The conditional edge `route_action` decides which
specialist node to call next, or sends control to `finalize` when the
orchestrator picks `answer` or the step limit is hit.

Why LangGraph?
- Explicit, inspectable graph structure (judges can see the topology)
- Built-in support for streaming, checkpointing, visualization
- Natural fit for the orchestrator-specialist pattern required by Round 2
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ...shared.config import CONFIG
from ...shared.llm import complete, complete_json
from ...shared.state import (
    ActionType,
    EvidenceItem,
    InvestigationState,
    InvestigationStep,
)
from . import reasoning, retrieval

# ── LangGraph State ────────────────────────────────────────────────────────────
# LangGraph requires a TypedDict as the graph state type.
# We carry our rich InvestigationState inside it so all existing agents keep
# working unchanged.

class GraphState(TypedDict):
    inv: InvestigationState          # the full investigation state
    client: Any                      # TigerGraph client (injected at runtime)
    next_action: str                 # set by orchestrate, read by router
    next_input: dict                 # parameters for the chosen specialist
    rationale: str                   # why the orchestrator chose this action


# ── Orchestrator system prompt ─────────────────────────────────────────────────
ORCHESTRATOR_SYSTEM = """\
You are the orchestrator of an agentic graph-investigation system.
You do not answer the question yourself. Each turn, choose exactly ONE next
action from this fixed vocabulary based on the question, the evidence
gathered so far, and the most recently identified gap:

  entity_link       – resolve a name/mention to a graph entity
                      input: {"mention": str}
  graph_traverse    – multi-hop walk from a linked entity
                      input: {"start_node_id": str, "hops": int, "edge_types": [str]}
  vector_search     – semantic search over documents
                      input: {"query_text": str}
  document_retrieve – fetch one document by id (only if a doc_id appeared in prior evidence)
                      input: {"doc_id": str}
  aggregate         – merge/synthesise evidence gathered so far
                      input: {}
  multi_hop_reason  – chain evidence to answer a sub-question
                      input: {"sub_question": str}
  evaluate_evidence – check sufficiency and identify the remaining gap
                      input: {}
  answer            – you are confident. stop investigating.
                      input: {}

Rules:
  - Never repeat an identical action+input pair.
  - Prefer entity_link before graph_traverse (you need a start_node_id first).
  - Call evaluate_evidence before answer unless you just aggregated and are done.
  - Pick answer only when you can cite specific evidence for every claim.

Return ONLY valid JSON (no prose, no code fences):
{"action": str, "action_input": dict, "rationale": str}
"""


def _orchestrate_prompt(inv: InvestigationState) -> str:
    gap = inv.gaps[-1] if inv.gaps else "(none yet)"
    taken = [f"{s.action.value}({s.action_input})" for s in inv.steps]
    return (
        f"Question: {inv.question}\n\n"
        f"Steps taken so far: {taken or '(none)'}\n\n"
        f"Most recent gap: {gap}\n\n"
        f"Evidence so far:\n{inv.evidence_summary()}\n\n"
        "Choose the next action."
    )


# ── Node: orchestrate ──────────────────────────────────────────────────────────
def orchestrate(state: GraphState) -> GraphState:
    """LLM decides the next action. Writes next_action / next_input / rationale."""
    inv = state["inv"]

    # Force stop if step limit hit
    if inv.should_force_stop(CONFIG.max_investigation_steps):
        return {**state, "next_action": "answer", "next_input": {}, "rationale": "max_steps_reached"}

    parsed, resp = complete_json(ORCHESTRATOR_SYSTEM, _orchestrate_prompt(inv))
    action   = parsed.get("action", "evaluate_evidence")
    inp      = parsed.get("action_input", {}) or {}
    rationale = parsed.get("rationale", "")

    # Track orchestrator's own token spend as a step
    step = InvestigationStep(
        step_index=len(inv.steps),
        action=ActionType(action),
        action_input=inp,
        rationale=rationale,
        tokens_used=resp.total_tokens,
    )
    inv.record_step(step)

    return {**state, "next_action": action, "next_input": inp, "rationale": rationale}


# ── Routing edge ───────────────────────────────────────────────────────────────
def route_action(state: GraphState) -> str:
    """Map the orchestrator's choice to a node name (or END via finalize)."""
    action = state["next_action"]
    routing = {
        "entity_link":      "entity_link",
        "graph_traverse":   "graph_traverse",
        "vector_search":    "vector_search",
        "document_retrieve":"document_retrieve",
        "aggregate":        "aggregate",
        "multi_hop_reason": "multi_hop_reason",
        "evaluate_evidence":"evaluate_evidence",
        "answer":           "finalize",
    }
    return routing.get(action, "evaluate_evidence")


# ── Specialist nodes ───────────────────────────────────────────────────────────
def _last_step(inv: InvestigationState) -> InvestigationStep:
    return inv.steps[-1]


def node_entity_link(state: GraphState) -> GraphState:
    inv, client = state["inv"], state["client"]
    items = retrieval.entity_linking_agent(inv, client, **state["next_input"])
    _last_step(inv).evidence_ids = [i.id for i in items]
    return {**state, "inv": inv}


def node_graph_traverse(state: GraphState) -> GraphState:
    inv, client = state["inv"], state["client"]
    items = retrieval.graph_traversal_agent(inv, client, **state["next_input"])
    _last_step(inv).evidence_ids = [i.id for i in items]
    return {**state, "inv": inv}


def node_vector_search(state: GraphState) -> GraphState:
    inv, client = state["inv"], state["client"]
    items = retrieval.vector_search_agent(inv, client, **state["next_input"])
    _last_step(inv).evidence_ids = [i.id for i in items]
    return {**state, "inv": inv}


def node_document_retrieve(state: GraphState) -> GraphState:
    inv, client = state["inv"], state["client"]
    items = retrieval.document_retrieval_agent(inv, client, **state["next_input"])
    _last_step(inv).evidence_ids = [i.id for i in items]
    return {**state, "inv": inv}


def node_aggregate(state: GraphState) -> GraphState:
    inv = state["inv"]
    result = reasoning.aggregation_agent(inv)
    inv.gaps.append(result.get("synthesis", ""))
    return {**state, "inv": inv}


def node_multi_hop_reason(state: GraphState) -> GraphState:
    inv = state["inv"]
    sub_q = state["next_input"].get("sub_question", inv.question)
    result = reasoning.multi_hop_reasoning_agent(inv, sub_q)
    if not result.get("complete", True):
        inv.gaps.append(result.get("missing_link", "incomplete reasoning chain"))
    return {**state, "inv": inv}


def node_evaluate_evidence(state: GraphState) -> GraphState:
    inv = state["inv"]
    result = reasoning.evidence_evaluation_agent(inv)
    inv.final_confidence = float(result.get("confidence", 0.0))
    if result.get("sufficient") and inv.final_confidence >= CONFIG.min_confidence_to_stop:
        inv.gaps.append("(evidence sufficient)")
    else:
        inv.gaps.append(result.get("gap", "insufficient evidence"))
    return {**state, "inv": inv}


# ── Terminal node: finalize ────────────────────────────────────────────────────
def node_finalize(state: GraphState) -> GraphState:
    """Generate the final answer from accumulated evidence, then stop."""
    inv = state["inv"]
    system = (
        "Answer the question using ONLY the evidence provided. "
        "Cite evidence ids inline like [id]. "
        "If evidence is insufficient, say so explicitly rather than guessing."
    )
    prompt = f"Question: {inv.question}\n\nEvidence:\n{inv.evidence_summary()}\n\nAnswer:"
    resp = complete(system, prompt)
    inv.total_tokens += resp.total_tokens
    inv.final_answer = resp.text.strip()
    inv.stopped = True
    inv.stop_reason = inv.stop_reason or "answered"
    return {**state, "inv": inv}


# ── Build the graph ────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("orchestrate",       orchestrate)
    g.add_node("entity_link",       node_entity_link)
    g.add_node("graph_traverse",    node_graph_traverse)
    g.add_node("vector_search",     node_vector_search)
    g.add_node("document_retrieve", node_document_retrieve)
    g.add_node("aggregate",         node_aggregate)
    g.add_node("multi_hop_reason",  node_multi_hop_reason)
    g.add_node("evaluate_evidence", node_evaluate_evidence)
    g.add_node("finalize",          node_finalize)

    # Entry point
    g.set_entry_point("orchestrate")

    # Conditional routing: orchestrate -> specialist or finalize
    specialist_nodes = [
        "entity_link", "graph_traverse", "vector_search",
        "document_retrieve", "aggregate", "multi_hop_reason",
        "evaluate_evidence", "finalize",
    ]
    g.add_conditional_edges("orchestrate", route_action, {n: n for n in specialist_nodes})

    # All specialists loop back to orchestrate
    for node in specialist_nodes[:-1]:      # all except finalize
        g.add_edge(node, "orchestrate")

    # finalize -> END
    g.add_edge("finalize", END)

    return g.compile()


# Singleton compiled graph — built once, reused across calls
_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ── Public API (matches original orchestrator interface) ──────────────────────
def run_investigation(state: InvestigationState, client) -> InvestigationState:
    """
    Entry point called by agentic_graphrag/pipeline.py.
    Runs the LangGraph StateGraph to completion and returns the mutated state.
    """
    graph = _get_graph()
    initial: GraphState = {
        "inv": state,
        "client": client,
        "next_action": "",
        "next_input": {},
        "rationale": "",
    }
    final = graph.invoke(initial)
    return final["inv"]
