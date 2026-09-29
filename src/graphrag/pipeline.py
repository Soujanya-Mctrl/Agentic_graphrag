"""
Baseline #2: GraphRAG. A FIXED sequence — entity link -> 1-hop traverse ->
vector search -> generate. Uses graph structure (the thing naive RAG lacks)
but, unlike the agentic pipeline, always runs the same steps regardless of
whether they're needed or sufficient. This is the sequence-vs-orchestrator
line the benchmark is designed to isolate.
"""
from __future__ import annotations

import re
import time

from ..shared.embeddings import embed
from ..shared.llm import complete
from ..agentic_graphrag.agents.deterministic import deterministic_solve


SYSTEM = (
    "Answer the question using the provided graph relationships and document "
    "snippets. If they don't contain the answer, say so."
)


def _extract_candidate_mention(question: str) -> str:
    """Crude heuristic: take the longest capitalized run of words as the
    likely entity mention. Fine for a baseline — the agentic pipeline's
    entity_linking_agent is called deliberately by the orchestrator instead
    of guessed once like this."""
    matches = re.findall(r"(?:[A-Z][a-zA-Z]*\s*)+", question)
    return max(matches, key=len).strip() if matches else question.split()[0]


def run(question: str, client, hops: int = 1, top_k: int = 5) -> dict:
    start = time.time()
    tokens = 0

    mention = _extract_candidate_mention(question)
    entities = client.entity_search(mention, top_k=3)

    graph_facts = []
    for ent in entities:
        edges = client.traverse(ent["id"], hops=hops)
        graph_facts.extend(f"{e['from']} --{e['relation']}--> {e['to']}" for e in edges)

    query_vec = embed(question)
    doc_hits = client.vector_search(query_vec, top_k=top_k, query_text=question)

    # Also traverse graph from retrieved document nodes to link athletes, venues, games (top 3)
    for d in doc_hits[:3]:
        d_id = d.get("doc_id")
        if d_id:
            d_edges = client.traverse(d_id, hops=1)
            graph_facts.extend(f"{e['from']} --{e['relation']}--> {e['to']}" for e in d_edges)
            # Check Event:d_id as well
            ev_edges = client.traverse(f"Event:{d_id}", hops=1)
            graph_facts.extend(f"{e['from']} --{e['relation']}--> {e['to']}" for e in ev_edges)

    context = "Graph relationships:\n" + "\n".join(graph_facts[:15])
    context += "\n\nDocument snippets:\n" + "\n".join(
        f"[{h.get('doc_id')}] {h.get('text', '')}" for h in doc_hits[:12]
    )

    det = deterministic_solve(question, doc_hits)
    extra_context = ""
    if det["solved"] and det["confidence"] >= 0.9:
        extra_context = f"\n\nVerified Fact:\n{det['synthesis']}"

    prompt = f"Question: {question}\n\n{context}{extra_context}\n\nAnswer:"
    try:
        resp = complete(SYSTEM, prompt, max_tokens=250)
        answer = resp.text.strip()
        tokens += resp.total_tokens
    except Exception:
        answer = det["synthesis"] if det["solved"] else "Unable to answer due to API error."

    return {
        "pipeline": "graph_rag",
        "question": question,
        "answer": answer,
        "tokens_used": tokens,
        "evidence_count": len(graph_facts) + len(doc_hits),
        "latency_seconds": time.time() - start,
    }
