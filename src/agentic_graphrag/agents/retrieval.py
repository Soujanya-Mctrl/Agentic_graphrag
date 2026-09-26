"""
Retrieval-side specialized agents. Each is a plain function:
(state, tigergraph_client, **kwargs) -> list[EvidenceItem]
Kept as functions, not classes, because they're stateless — all state
lives in InvestigationState, which the orchestrator owns.
"""
from __future__ import annotations

import uuid

from ...shared.embeddings import embed
from ...shared.state import ActionType, EvidenceItem, InvestigationState


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def entity_linking_agent(state: InvestigationState, client, mention: str) -> list[EvidenceItem]:
    """Resolve a natural-language mention to graph node(s)."""
    matches = client.entity_search(mention, top_k=5)
    items = []
    for m in matches:
        state.linked_entities.append(m)
        item = EvidenceItem(
            id=_new_id(),
            source_action=ActionType.ENTITY_LINK,
            content=f"Linked '{mention}' -> {m.get('name', m.get('id'))} ({m.get('type', 'unknown')})",
            source_ref=str(m.get("id", mention)),
            confidence=0.9,
        )
        state.add_evidence(item)
        items.append(item)
    return items


def graph_traversal_agent(state: InvestigationState, client, start_node_id: str, hops: int = 2,
                            edge_types: list[str] | None = None) -> list[EvidenceItem]:
    """Multi-hop walk from a linked entity, following optional edge-type filters."""
    edges = client.traverse(start_node_id, hops=hops, edge_types=edge_types)
    items = []
    for e in edges:
        item = EvidenceItem(
            id=_new_id(),
            source_action=ActionType.GRAPH_TRAVERSE,
            content=f"{e['from']} --{e['relation']}--> {e['to']}",
            source_ref=f"{e['from']}->{e['to']}",
            confidence=0.85,
        )
        state.add_evidence(item)
        items.append(item)
    return items


def vector_search_agent(state: InvestigationState, client, query_text: str, top_k: int = 8) -> list[EvidenceItem]:
    """Semantic search over unstructured documents for evidence the graph
    doesn't capture structurally."""
    query_vec = embed(query_text)
    hits = client.vector_search(query_vec, top_k=top_k)
    items = []
    for h in hits:
        item = EvidenceItem(
            id=_new_id(),
            source_action=ActionType.VECTOR_SEARCH,
            content=h.get("text", ""),
            source_ref=h.get("doc_id", ""),
            confidence=float(h.get("score", 0.5)),
        )
        state.add_evidence(item)
        items.append(item)
    return items


def document_retrieval_agent(state: InvestigationState, client, doc_id: str) -> list[EvidenceItem]:
    """Pull a specific document by id — used when another agent's result
    references a doc_id that needs full-text follow-up."""
    doc = client.get_document(doc_id)
    if not doc:
        return []
    item = EvidenceItem(
        id=_new_id(),
        source_action=ActionType.DOCUMENT_RETRIEVE,
        content=doc.get("text", ""),
        source_ref=doc_id,
        confidence=1.0,
    )
    state.add_evidence(item)
    return [item]
