"""
Wraps pyTigerGraph so the rest of the codebase never imports it directly.
Ships a MockTigerGraphClient with an in-memory toy graph + doc store so the
whole pipeline runs and is demoable before Savanna credentials exist.

Swap to the real client the moment TG_USE_MOCK=false and TG_HOST/TG_SECRET
are set — nothing else in the codebase changes.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .config import CONFIG


class TigerGraphClient:
    """Real client — thin wrapper over pyTigerGraph. Fill in GSQL query
    names to match what you actually install on your Savanna instance."""

    def __init__(self):
        import pyTigerGraph as tg

        is_tg_cloud = "tgcloud.io" in CONFIG.tg_host or os.getenv("TG_TGCLOUD", "false").lower() == "true"

        if CONFIG.tg_secret:
            self.conn = tg.TigerGraphConnection(
                host=CONFIG.tg_host,
                graphname=CONFIG.tg_graph_name,
                gsqlSecret=CONFIG.tg_secret,
                tgCloud=is_tg_cloud,
            )
            # Exchange secret for authorization token
            self.conn.getToken(secret=CONFIG.tg_secret)
        else:
            self.conn = tg.TigerGraphConnection(
                host=CONFIG.tg_host,
                graphname=CONFIG.tg_graph_name,
                username=CONFIG.tg_username,
                password=CONFIG.tg_password,
                tgCloud=is_tg_cloud,
            )
            if CONFIG.tg_username and CONFIG.tg_password:
                self.conn.getToken()

    def entity_search(self, mention: str, top_k: int = 5) -> list[dict[str, Any]]:
        # Expect a GSQL query installed as e.g. `entityLinkByName`
        return self.conn.runInstalledQuery(
            "entityLinkByName", params={"mention": mention, "k": top_k}
        )

    def traverse(self, start_node_id: str, hops: int, edge_types: Optional[list[str]] = None) -> list[dict[str, Any]]:
        return self.conn.runInstalledQuery(
            "multiHopTraverse",
            params={"start": start_node_id, "hops": hops, "edgeTypes": edge_types or []},
        )

    def vector_search(self, query_embedding: list[float], top_k: int = 8) -> list[dict[str, Any]]:
        return self.conn.runInstalledQuery(
            "vectorTopK", params={"embedding": query_embedding, "k": top_k}
        )

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        result = self.conn.runInstalledQuery("getDocument", params={"docId": doc_id})
        return result[0] if result else None


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

    def vector_search(self, query_embedding: list[float], top_k: int = 8) -> list[dict[str, Any]]:
        # Mock: ignore the embedding, just return all docs ranked by id.
        return [{"doc_id": d["id"], "text": d["text"], "score": 0.9} for d in list(self._docs.values())[:top_k]]

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        return self._docs.get(doc_id)


def get_client():
    if CONFIG.tg_use_mock:
        return MockTigerGraphClient()
    return TigerGraphClient()
