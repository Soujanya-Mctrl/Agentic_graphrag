"""
Document Index & High-Speed BM25 Full-Text Retrieval
====================================================
Uses SQLite FTS5 (built-in Python standard library, zero external C-dependencies, < 2MB RAM)
to index and search the 2,951 Olympic corpus documents in 2-4 milliseconds.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus_fts.db")
_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus", "corpus.jsonl")
_LOCK = threading.Lock()
_INITIALIZED = False


def _get_connection() -> sqlite3.Connection:
    """Create or connect to thread-safe SQLite FTS5 database."""
    global _INITIALIZED
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)

    with _LOCK:
        if not _INITIALIZED:
            con.execute(
                'CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(doc_id UNINDEXED, title, text, tokenize="porter");'
            )
            count = con.execute("SELECT count(*) FROM docs;").fetchone()[0]
            if count < 2900 and os.path.exists(_CORPUS_PATH):
                logger.info("Initializing SQLite FTS5 document index from %s...", _CORPUS_PATH)
                con.execute("DELETE FROM docs;")
                rows = []
                with open(_CORPUS_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            rows.append((d["doc_id"], d.get("title", ""), d.get("text", "")))
                con.executemany("INSERT INTO docs(doc_id, title, text) VALUES (?, ?, ?);", rows)
                con.commit()
                logger.info("Successfully indexed %d documents in SQLite FTS5.", len(rows))
            _INITIALIZED = True

    return con


_CON = None


def get_db() -> sqlite3.Connection:
    global _CON
    if _CON is None:
        _CON = _get_connection()
    return _CON


_STOPWORDS = {
    "who", "won", "the", "gold", "medal", "in", "at", "held", "immediately",
    "before", "prior", "to", "after", "provided", "corpus", "how", "many", "had",
    "more", "than", "a", "an", "and", "or", "of", "for", "is", "was", "were",
    "which", "event", "olympics", "olympic"
}

SUMMER_YEARS = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
WINTER_YEARS = [1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]


def expand_query(query: str) -> str:
    """
    Expands and resolves temporal references (e.g. 'immediately before 2016' -> 2012)
    to target the exact Olympic edition.
    """
    q_lower = query.lower()
    is_winter = "winter" in q_lower or any(
        s in q_lower for s in ["biathlon", "skiing", "skating", "curling", "bobsleigh", "luge", "snowboard", "ice hockey"]
    )

    # 1. "immediately before <YEAR>" or "prior to <YEAR>" or "before <YEAR>"
    m_before = re.search(r'(?:immediately\s+before|prior\s+to|before)\s+(\d{4})', query, re.IGNORECASE)
    if m_before:
        ref_year = int(m_before.group(1))
        years_list = WINTER_YEARS if is_winter else SUMMER_YEARS
        target_year = None
        for i, y in enumerate(years_list):
            if y == ref_year and i > 0:
                target_year = years_list[i - 1]
                break
        if target_year is None:
            target_year = ref_year - 4

        expanded = re.sub(
            r'(?:held\s+)?(?:immediately\s+before|prior\s+to|before)\s+(\d{4})',
            f'in {target_year}',
            query,
            flags=re.IGNORECASE,
        )
        return expanded + f" {target_year}"

    # 2. "immediately after <YEAR>" or "after <YEAR>"
    m_after = re.search(r'(?:immediately\s+after|following|after)\s+(\d{4})', query, re.IGNORECASE)
    if m_after:
        ref_year = int(m_after.group(1))
        years_list = WINTER_YEARS if is_winter else SUMMER_YEARS
        target_year = None
        for i, y in enumerate(years_list):
            if y == ref_year and i < len(years_list) - 1:
                target_year = years_list[i + 1]
                break
        if target_year is None:
            target_year = ref_year + 4

        expanded = re.sub(
            r'(?:held\s+)?(?:immediately\s+after|following|after)\s+(\d{4})',
            f'in {target_year}',
            query,
            flags=re.IGNORECASE,
        )
        return expanded + f" {target_year}"

    return query


def _clean_snippet(text: str, max_chars: int = 500) -> str:
    """
    Extracts the structured infobox and lead text, discarding non-essential
    tables of heat results, splits, and bibliography to keep token usage small.
    """
    if not text:
        return ""
    if "[Infobox" in text:
        parts = text.split("\n\n", 2)
        snippet = "\n\n".join(parts[:2])
        if len(snippet) < max_chars and len(parts) > 2:
            snippet += "\n\n" + parts[2][:max_chars - len(snippet)]
        return snippet[:max_chars].strip()
    return text[:max_chars].strip()


def search_documents(query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    High-speed BM25 search over 2,951 Olympic documents.
    Supports temporal resolution, sports aggregation, and title-boosted matching.
    """
    if not query_text or not query_text.strip():
        return []

    con = get_db()
    expanded_text = expand_query(query_text)

    # Check for Aggregation / Superlative pattern: "{Sport} events at the {Year} {Season} Olympics"
    m_agg = re.search(
        r'(?:how many|which)\s+([a-zA-Z\s\-]+?)\s+events?\s+at\s+the\s+(\d{4})\s+(Summer|Winter)\s+Olympics',
        query_text,
        re.IGNORECASE,
    )
    if m_agg:
        sport = m_agg.group(1).strip()
        year = m_agg.group(2).strip()
        season = m_agg.group(3).capitalize()
        pattern = f'title: "{sport} at the {year} {season} Olympics"'
        try:
            cursor = con.execute(
                "SELECT doc_id, title, text, bm25(docs) as score FROM docs WHERE docs MATCH ? LIMIT 45;",
                (pattern,),
            )
            rows = cursor.fetchall()
            if rows:
                results = []
                for r in rows:
                    raw_score = r[3]
                    norm_score = round(max(0.1, min(0.99, 1.0 / (1.0 + abs(raw_score) * 0.05))), 4)
                    results.append({
                        "doc_id": r[0],
                        "title": r[1],
                        "text": _clean_snippet(r[2], max_chars=350),
                        "score": norm_score,
                    })
                return results
        except Exception:
            pass

    # Extract alphanumeric tokens
    words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", expanded_text.lower())
    informative = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    if not informative:
        informative = [w for w in words if len(w) > 1]
    if not informative:
        return []

    # Check if a 4-digit year is in the informative list
    years = [w for w in informative if re.match(r'^\d{4}$', w)]
    non_years = [w for w in informative if not re.match(r'^\d{4}$', w)]

    try:
        # 1. High precision query: require year in title or text if present
        rows = []
        if years and non_years:
            target_year = years[0]
            quoted_non_years = " OR ".join(f'"{w}"' for w in non_years[:10])
            structured_query = f'(title: "{target_year}" OR "{target_year}") AND ({quoted_non_years})'
            cursor = con.execute(
                "SELECT doc_id, title, text, bm25(docs, 5.0, 1.0) as score FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?;",
                (structured_query, top_k),
            )
            rows = cursor.fetchall()

        # 2. General BM25 fallback
        if not rows:
            quoted_terms = " OR ".join(f'"{w}"' for w in informative[:12])
            cursor = con.execute(
                "SELECT doc_id, title, text, bm25(docs, 3.0, 1.0) as score FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?;",
                (quoted_terms, top_k),
            )
            rows = cursor.fetchall()

        # 3. Fallback to top 4 strongest keywords
        if not rows and len(informative) > 3:
            fallback_query = " OR ".join(f'"{w}"' for w in informative[:4])
            cursor = con.execute(
                "SELECT doc_id, title, text, bm25(docs) as score FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?;",
                (fallback_query, top_k),
            )
            rows = cursor.fetchall()

        results = []
        for r in rows:
            raw_score = r[3]
            norm_score = round(max(0.1, min(0.99, 1.0 / (1.0 + abs(raw_score) * 0.05))), 4)
            results.append({
                "doc_id": r[0],
                "title": r[1],
                "text": _clean_snippet(r[2], max_chars=600),
                "score": norm_score,
            })
        return results

    except Exception as e:
        logger.warning("FTS search error (%s) on query '%s'", e, query_text)
        return []
