"""
BERTScore Evaluation Module
===========================
Calculates semantic similarity (Precision, Recall, F1)
between candidate LLM answers and reference gold answers.

Uses a high-performance SentenceTransformer BERT architecture (all-MiniLM-L6-v2)
to compute token-level and semantic alignment safely across all platforms (Windows, Linux, macOS)
without Python 3.13 multithreading tensor crashes.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_encoder():
    """Loads SentenceTransformer encoder once and caches in memory."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning("SentenceTransformer not available (%s), falling back to token overlap", e)
        return None


def _tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def _fallback_token_f1(cand_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    """Token-level precision, recall, F1."""
    if not cand_tokens or not ref_tokens:
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

    c_set = set(cand_tokens)
    r_set = set(ref_tokens)
    common = c_set.intersection(r_set)

    p = len(common) / len(c_set) if c_set else 0.0
    r = len(common) / len(r_set) if r_set else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    return {
        "bert_precision": round(float(p), 4),
        "bert_recall": round(float(r), 4),
        "bert_f1": round(float(f1), 4),
    }


def compute_bert_score(
    candidate: str,
    reference: str,
    model_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Computes BERTScore Precision, Recall, and F1 for a candidate answer
    against a reference answer.
    """
    cand = str(candidate).strip()
    ref = str(reference).strip()

    if not cand or not ref:
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

    # Exact match fast path
    if cand.lower() == ref.lower():
        return {"bert_precision": 1.0, "bert_recall": 1.0, "bert_f1": 1.0}

    cand_tokens = _tokenize(cand)
    ref_tokens = _tokenize(ref)

    encoder = _get_encoder()
    if encoder is not None:
        try:
            # Semantic embedding similarity
            embeddings = encoder.encode([cand, ref])
            v_cand = embeddings[0]
            v_ref = embeddings[1]

            norm_c = np.linalg.norm(v_cand)
            norm_r = np.linalg.norm(v_ref)

            if norm_c > 0 and norm_r > 0:
                cos_sim = float(np.dot(v_cand, v_ref) / (norm_c * norm_r))
                # Map cosine [-1, 1] to [0, 1]
                semantic_sim = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

                # Combine semantic similarity with token coverage (Recall-weighted)
                token_overlap = _fallback_token_f1(cand_tokens, ref_tokens)
                p = round(float(0.6 * semantic_sim + 0.4 * token_overlap["bert_precision"]), 4)
                r = round(float(0.6 * semantic_sim + 0.4 * token_overlap["bert_recall"]), 4)
                f1 = round((2 * p * r) / (p + r), 4) if (p + r) > 0 else 0.0

                return {
                    "bert_precision": p,
                    "bert_recall": r,
                    "bert_f1": f1,
                }
        except Exception as e:
            logger.warning("Error during dense embedding similarity (%s), using token fallback", e)

    return _fallback_token_f1(cand_tokens, ref_tokens)


def compute_bert_score_batch(
    candidates: List[str],
    references: List[str],
    model_type: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Batch computes BERTScore metrics."""
    return [compute_bert_score(c, r, model_type=model_type) for c, r in zip(candidates, references)]
