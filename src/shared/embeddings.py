"""Text -> vector. Real embedder if sentence-transformers is installed,
otherwise a deterministic hash-based fallback so vector_search still runs
(with junk relevance) in a bare environment."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

from .config import CONFIG

_DIM = 384


@lru_cache(maxsize=1)
def _model():
    try:
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise

    if CONFIG.hf_token:
        os.environ.setdefault("HF_TOKEN", CONFIG.hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", CONFIG.hf_token)

    try:
        return SentenceTransformer(CONFIG.embedding_model, token=CONFIG.hf_token or None, device="cpu")
    except TypeError:
        # Older sentence-transformers versions use a different auth keyword or no auth kwarg at all.
        try:
            return SentenceTransformer(CONFIG.embedding_model, use_auth_token=CONFIG.hf_token or None, device="cpu")
        except TypeError:
            return SentenceTransformer(CONFIG.embedding_model, device="cpu")


def embed(text: str) -> list[float]:
    if os.environ.get("USE_MOCK_EMBEDDINGS") == "1" or os.environ.get("RENDER") == "true":
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b - 127.5) / 127.5 for b in h]
        return (vec * (_DIM // len(vec) + 1))[:_DIM]
    try:
        return _model().encode(text).tolist()
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError):
        # Deterministic fallback: hash chunks of the string into a fixed-size
        # vector. Not semantically meaningful — replace by installing
        # sentence-transformers before you trust vector_search results.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b - 127.5) / 127.5 for b in h]
        return (vec * (_DIM // len(vec) + 1))[:_DIM]

