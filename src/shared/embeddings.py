"""Text -> vector. Real embedder if sentence-transformers is installed,
otherwise a deterministic hash-based fallback so vector_search still runs
(with junk relevance) in a bare environment."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from .config import CONFIG

_DIM = 384


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CONFIG.embedding_model)


def embed(text: str) -> list[float]:
    try:
        return _model().encode(text).tolist()
    except Exception:
        # Deterministic fallback: hash chunks of the string into a fixed-size
        # vector. Not semantically meaningful — replace by installing
        # sentence-transformers before you trust vector_search results.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b - 127.5) / 127.5 for b in h]
        return (vec * (_DIM // len(vec) + 1))[:_DIM]
