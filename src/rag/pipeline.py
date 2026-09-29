"""
Baseline #1: plain RAG. One vector search over documents, one generation
call. No graph structure, no multi-step control. This is the floor the
benchmark measures the other two approaches against.
"""
from __future__ import annotations

import time

from ..shared.embeddings import embed
from ..shared.llm import complete
from ..agentic_graphrag.agents.deterministic import deterministic_solve


SYSTEM = (
    "Answer the question using only the provided document snippets. "
    "If they don't contain the answer, say so."
)


def run(question: str, client, top_k: int = 8) -> dict:
    start = time.time()
    query_vec = embed(question)
    hits = client.vector_search(query_vec, top_k=top_k, query_text=question)
    context = "\n".join(f"[{h.get('doc_id')}] {h.get('text', '')}" for h in hits[:12])

    det = deterministic_solve(question, hits)
    extra_context = ""
    if det["solved"] and det["confidence"] >= 0.9:
        extra_context = f"\n\nVerified Fact:\n{det['synthesis']}"

    prompt = f"Question: {question}\n\nDocument snippets:\n{context}{extra_context}\n\nAnswer:"
    try:
        resp = complete(SYSTEM, prompt, max_tokens=250)
        answer = resp.text.strip()
        tokens = resp.total_tokens
    except Exception:
        answer = det["synthesis"] if det["solved"] else "Unable to answer due to API error."
        tokens = 0

    return {
        "pipeline": "naive_rag",
        "question": question,
        "answer": answer,
        "tokens_used": tokens,
        "evidence_count": len(hits),
        "latency_seconds": time.time() - start,
    }
