"""
Baseline #1: plain RAG. One vector search over documents, one generation
call. No graph structure, no multi-step control. This is the floor the
benchmark measures the other two approaches against.
"""
from __future__ import annotations

import time

from ..shared.embeddings import embed
from ..shared.llm import complete


SYSTEM = (
    "Answer the question using only the provided document snippets. "
    "If they don't contain the answer, say so."
)


def run(question: str, client, top_k: int = 8) -> dict:
    start = time.time()
    query_vec = embed(question)
    hits = client.vector_search(query_vec, top_k=top_k, query_text=question)
    context = "\n".join(f"[{h.get('doc_id')}] {h.get('text', '')}" for h in hits[:12])

    prompt = f"Question: {question}\n\nDocument snippets:\n{context}\n\nAnswer:"
    resp = complete(SYSTEM, prompt)

    return {
        "pipeline": "naive_rag",
        "question": question,
        "answer": resp.text.strip(),
        "tokens_used": resp.total_tokens,
        "evidence_count": len(hits),
        "latency_seconds": time.time() - start,
    }
