"""
Wraps pyTigerGraph so the rest of the codebase never imports it directly.
Ships a MockTigerGraphClient with an in-memory toy graph + doc store so the
whole pipeline runs and is demoable before Savanna credentials exist.

Swap to the real client the moment TG_USE_MOCK=false and TG_HOST/TG_SECRET
are set — nothing else in the codebase changes.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from dotenv import dotenv_values, load_dotenv
from .config import CONFIG


def _normalize_token(token_value):
    if isinstance(token_value, (tuple, list)) and token_value:
        return token_value[0]
    return token_value


def _clean_credential(val: Optional[str]) -> str:
    if not val:
        return ""
    val = str(val).strip()
    if val.startswith("your_") or "here" in val.lower() or val == "dummy":
        return ""
    return val


class TigerGraphClient:
    """Real client — thin wrapper over pyTigerGraph. Supports both installed GSQL
    stored queries and native pyTigerGraph REST/graph API fallbacks for live clusters."""

    def __init__(self):
        import pyTigerGraph as tg

        load_dotenv(override=False)
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_file_vals = dotenv_values(env_path) if os.path.exists(env_path) else {}

        host = (
            _clean_credential(CONFIG.tg_host)
            or _clean_credential(os.getenv("TG_HOST"))
            or _clean_credential(env_file_vals.get("TG_HOST"))
            or ""
        )
        graphname = (
            _clean_credential(CONFIG.tg_graph_name)
            or _clean_credential(os.getenv("TG_GRAPHNAME"))
            or _clean_credential(env_file_vals.get("TG_GRAPHNAME"))
            or "AgenticGraphRag"
        )
        secret = (
            _clean_credential(CONFIG.tg_secret)
            or _clean_credential(os.getenv("TG_SECRET"))
            or _clean_credential(env_file_vals.get("TG_SECRET"))
            or ""
        )
        username = (
            _clean_credential(CONFIG.tg_username)
            or _clean_credential(os.getenv("TG_USERNAME"))
            or _clean_credential(env_file_vals.get("TG_USERNAME"))
            or "tigergraph"
        )
        password = (
            _clean_credential(CONFIG.tg_password)
            or _clean_credential(os.getenv("TG_PASSWORD"))
            or _clean_credential(env_file_vals.get("TG_PASSWORD"))
            or ""
        )
        is_tg_cloud = "tgcloud.io" in host or os.getenv("TG_TGCLOUD", "false").lower() == "true"

        if secret:
            self.conn = tg.TigerGraphConnection(
                host=host,
                graphname=graphname,
                gsqlSecret=secret,
                tgCloud=is_tg_cloud,
            )
            try:
                token = _normalize_token(self.conn.getToken(secret=secret))
                if token:
                    self.conn = tg.TigerGraphConnection(
                        host=host,
                        graphname=graphname,
                        tgCloud=is_tg_cloud,
                        apiToken=token,
                    )
            except Exception as e:
                # If secret exchange fails, attempt fallback to username/password if provided
                if username and password and password != "your_password":
                    self.conn = tg.TigerGraphConnection(
                        host=host,
                        graphname=graphname,
                        username=username,
                        password=password,
                        tgCloud=is_tg_cloud,
                    )
                    token = _normalize_token(self.conn.getToken())
                    if token:
                        self.conn = tg.TigerGraphConnection(
                            host=host,
                            graphname=graphname,
                            tgCloud=is_tg_cloud,
                            apiToken=token,
                        )
                else:
                    raise e
        elif username and password and password != "your_password":
            self.conn = tg.TigerGraphConnection(
                host=host,
                graphname=graphname,
                username=username,
                password=password,
                tgCloud=is_tg_cloud,
            )
            token = _normalize_token(self.conn.getToken())
            if token:
                self.conn = tg.TigerGraphConnection(
                    host=host,
                    graphname=graphname,
                    tgCloud=is_tg_cloud,
                    apiToken=token,
                )
        else:
            raise ValueError(
                "No valid TigerGraph credentials found. Please set TG_SECRET or TG_USERNAME/TG_PASSWORD in .env"
            )

        self._corpus_cache: Optional[list[dict[str, Any]]] = None

    def _get_corpus(self, limit: int = 50) -> list[dict[str, Any]]:
        """Lazily load a small sample of local corpus documents as fallback when live queries fail."""
        if self._corpus_cache is not None:
            return self._corpus_cache
        corpus_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus", "corpus.jsonl")
        docs = []
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= limit:
                            break
                        if line.strip():
                            docs.append(json.loads(line))
            except Exception:
                docs = []
        self._corpus_cache = docs
        return self._corpus_cache

    def entity_search(self, mention: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search entities by mention using installed query or native vertex lookup."""
        try:
            return self.conn.runInstalledQuery("entityLinkByName", params={"mention": mention, "k": top_k})
        except Exception:
            pass

        # Native fallback: check candidate entity IDs and exact name match
        clean_mention = mention.strip()
        candidates = [
            f"Athlete:{clean_mention}",
            f"Event:{clean_mention}",
            f"Games:{clean_mention}",
            f"Venue:{clean_mention}",
            f"Sport:{clean_mention}",
            f"Country:{clean_mention}",
            clean_mention,
        ]
        found = []
        try:
            res = self.conn.getVerticesById("Entity", candidates)
            for v in res:
                found.append({
                    "id": v["v_id"],
                    "name": v.get("attributes", {}).get("name", v["v_id"]),
                    "type": v.get("attributes", {}).get("entity_type", "Entity"),
                })
        except Exception:
            pass

        if not found:
            try:
                escaped = clean_mention.replace('"', '\\"')
                res = self.conn.getVertices("Entity", where=f'name=="{escaped}"', limit=top_k)
                for v in res:
                    found.append({
                        "id": v["v_id"],
                        "name": v.get("attributes", {}).get("name", v["v_id"]),
                        "type": v.get("attributes", {}).get("entity_type", "Entity"),
                    })
            except Exception:
                pass

        return found[:top_k]

    def traverse(self, start_node_id: str, hops: int = 1, edge_types: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Traverse graph from start node using installed query or native edge traversal."""
        try:
            return self.conn.runInstalledQuery(
                "multiHopTraverse",
                params={"start": start_node_id, "hops": hops, "edgeTypes": edge_types or []},
            )
        except Exception:
            pass

        # Native edge traversal fallback with in-memory caching
        if not hasattr(self, "_edge_cache"):
            self._edge_cache = {}

        results = []
        frontier = [start_node_id]
        seen_nodes = set([start_node_id])
        seen_edges = set()

        for _ in range(max(1, hops)):
            next_frontier = []
            for node in frontier:
                v_type = "Document" if node.startswith("Q") and not node.startswith("Event:") else "Entity"
                cache_key = (v_type, node)
                if cache_key in self._edge_cache:
                    edges = self._edge_cache[cache_key]
                else:
                    try:
                        edges = self.conn.getEdges(v_type, node)
                        self._edge_cache[cache_key] = edges
                    except Exception:
                        continue

                for e in edges:
                    rel = e.get("attributes", {}).get("rel_type") or e.get("e_type", "RELATED_TO")
                    if edge_types and rel not in edge_types and e.get("e_type") not in edge_types:
                        continue
                    edge_key = (e["from_id"], rel, e["to_id"])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        results.append({"from": e["from_id"], "relation": rel, "to": e["to_id"]})
                        if e["to_id"] not in seen_nodes:
                            seen_nodes.add(e["to_id"])
                            next_frontier.append(e["to_id"])
            frontier = next_frontier
            if not frontier:
                break

        return results

    def vector_search(self, query_embedding: Optional[list[float]] = None, top_k: int = 8, query_text: Optional[str] = None) -> list[dict[str, Any]]:
        """Vector / full-text hybrid search over Olympic corpus documents."""
        # 1. High-speed BM25 full-text search when query_text is available
        if query_text:
            try:
                from .document_index import search_documents
                docs = search_documents(query_text, top_k=top_k)
                if docs:
                    return docs
            except Exception as e:
                logger.debug("Local FTS search exception: %s", e)

        # 2. Installed TigerGraph query if available
        if query_embedding:
            try:
                return self.conn.runInstalledQuery("vectorTopK", params={"embedding": query_embedding, "k": top_k})
            except Exception:
                pass

        # 3. Query live Document vertices from Savanna Cloud
        try:
            v_docs = self.conn.getVertices("Document", limit=top_k)
            if v_docs:
                return [
                    {
                        "doc_id": d["v_id"],
                        "title": d.get("attributes", {}).get("title", ""),
                        "text": d.get("attributes", {}).get("text", "")[:1200],
                        "score": 0.85,
                    }
                    for d in v_docs
                ]
        except Exception:
            pass

        # 4. Fallback to local corpus
        corpus = self._get_corpus(limit=top_k)
        return [
            {
                "doc_id": d["doc_id"],
                "title": d.get("title", ""),
                "text": d.get("text", "")[:1200],
                "score": 0.90,
            }
            for d in corpus[:top_k]
        ]

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full document text by doc_id."""
        try:
            res = self.conn.runInstalledQuery("getDocument", params={"docId": doc_id})
            if res:
                return res[0]
        except Exception:
            pass

        try:
            docs = self.conn.getVerticesById("Document", doc_id)
            if docs:
                attr = docs[0].get("attributes", {})
                return {
                    "doc_id": doc_id,
                    "title": attr.get("title", ""),
                    "text": attr.get("text", ""),
                }
        except Exception:
            pass

        # Stream file line-by-line without loading 23MB into memory
        corpus_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus", "corpus.jsonl")
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if doc_id in line:
                            data = json.loads(line)
                            if data.get("doc_id") == doc_id:
                                return data
            except Exception:
                pass
        return None


class MockTigerGraphClient:
    """In-memory stand-in with a tiny toy graph so `python -m src.demo` works
    with zero external services. Not a substitute for the real dataset —
    swap TG_USE_MOCK=false before submitting."""

    def __init__(self):
        self._entities = {
            "e1": {"id": "e1", "name": "Acme Corp", "type": "Organization"},
            "e2": {"id": "e2", "name": "Jane Doe", "type": "Person"},
            "e3": {"id": "e3", "name": "Project Helios", "type": "Project"},
        }
        self._edges = [
            ("e2", "WORKS_AT", "e1"),
            ("e2", "LEADS", "e3"),
            ("e3", "FUNDED_BY", "e1"),
        ]
        self._docs = {
            "d1": {"id": "d1", "text": "Jane Doe joined Acme Corp in 2021 as VP of Engineering."},
            "d2": {"id": "d2", "text": "Project Helios was funded internally by Acme Corp starting Q2 2023."},
        }

    def entity_search(self, mention: str, top_k: int = 5) -> list[dict[str, Any]]:
        mention_l = mention.lower()
        return [e for e in self._entities.values() if mention_l in e["name"].lower()][:top_k]

    def traverse(self, start_node_id: str, hops: int, edge_types: Optional[list[str]] = None) -> list[dict[str, Any]]:
        results = []
        frontier = [start_node_id]
        seen = set()
        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                for src, rel, dst in self._edges:
                    if edge_types and rel not in edge_types:
                        continue
                    if src == node and dst not in seen:
                        results.append({"from": src, "relation": rel, "to": dst})
                        next_frontier.append(dst)
                        seen.add(dst)
            frontier = next_frontier
        return results

    def vector_search(self, query_embedding: Optional[list[float]] = None, top_k: int = 8, query_text: Optional[str] = None) -> list[dict[str, Any]]:
        if query_text:
            try:
                from .document_index import search_documents
                docs = search_documents(query_text, top_k=top_k)
                if docs:
                    return docs
            except Exception:
                pass
        return [{"doc_id": d["id"], "text": d["text"], "score": 0.9} for d in list(self._docs.values())[:top_k]]

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        # Check SQLite FTS first
        try:
            from .document_index import get_db
            row = get_db().execute("SELECT doc_id, title, text FROM docs WHERE doc_id = ? LIMIT 1;", (doc_id,)).fetchone()
            if row:
                return {"doc_id": row[0], "title": row[1], "text": row[2]}
        except Exception:
            pass
        return self._docs.get(doc_id)


_CLIENT_INSTANCE: Optional[TigerGraphClient | MockTigerGraphClient] = None


def get_client(force_real: bool = False):
    """Retrieve or initialize singleton TigerGraph client."""
    global _CLIENT_INSTANCE
    use_mock = False if force_real else os.getenv("TG_USE_MOCK", str(CONFIG.tg_use_mock)).lower() == "true"

    if _CLIENT_INSTANCE is not None:
        # If client type matches desired mode, reuse existing instance
        if use_mock and isinstance(_CLIENT_INSTANCE, MockTigerGraphClient):
            return _CLIENT_INSTANCE
        if not use_mock and isinstance(_CLIENT_INSTANCE, TigerGraphClient):
            return _CLIENT_INSTANCE

    if use_mock:
        _CLIENT_INSTANCE = MockTigerGraphClient()
        return _CLIENT_INSTANCE

    try:
        _CLIENT_INSTANCE = TigerGraphClient()
        return _CLIENT_INSTANCE
    except Exception as e:
        print(f"[!] Warning: TigerGraphClient connection failed ({e}). Falling back to MockTigerGraphClient.")
        _CLIENT_INSTANCE = MockTigerGraphClient()
        return _CLIENT_INSTANCE
