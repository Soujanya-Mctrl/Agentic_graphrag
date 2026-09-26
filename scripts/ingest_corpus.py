#!/usr/bin/env python3
"""
Ingest the Agentic GraphRAG corpus (corpus.jsonl) into TigerGraph.
Populates:
  - Document vertices: id, title, url, text, approx_tokens
  - Entity vertices: Athletes, Venues, Games, Sports, Countries, Events
  - MENTIONS edges: Document -> Entity
  - RELATION edges: Entity -> Entity (HELD_AT, PART_OF_GAMES, WON_GOLD, etc.)
"""

import os
import sys
import json
import re
import time
import argparse
from typing import Dict, List, Tuple, Set
from dotenv import load_dotenv
import pyTigerGraph as tg

load_dotenv()

def extract_infobox(text: str) -> Dict[str, str]:
    """Extract key-value pairs from Wikipedia [Infobox Olympic event] sections."""
    info = {}
    lines = text.split("\n")
    in_box = False
    for line in lines:
        if line.startswith("[Infobox"):
            in_box = True
            continue
        if in_box:
            if line.strip() == "" or (line.startswith("[") and not line.startswith("[Infobox")):
                break
            m = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*(.+)$", line)
            if m:
                k = m.group(1).strip()
                v = m.group(2).strip()
                info[k] = v
    return info

def extract_sport(title: str) -> str:
    """Infer sport discipline from article title (e.g., 'Canoeing at the ...' -> 'Canoeing')."""
    m = re.match(r"^([a-zA-Z\s\-]+)\s+at the", title)
    if m:
        return m.group(1).strip()
    return ""

def clean_str(s: str) -> str:
    """Sanitize strings for TigerGraph JSON payload."""
    if not s:
        return ""
    # Remove null bytes or control characters that break JSON
    return s.replace("\x00", "").strip()

def run_ingest(limit: int = None, batch_size: int = 150):
    host = os.getenv("TG_HOST")
    graphname = os.getenv("TG_GRAPHNAME", "AgenticGraphRag")
    secret = os.getenv("TG_SECRET", "")
    username = os.getenv("TG_USERNAME", "tigergraph")
    password = os.getenv("TG_PASSWORD", "tigergraph")
    tg_cloud = os.getenv("TG_TGCLOUD", "false").lower() == "true"

    corpus_file = os.path.join(os.path.dirname(__file__), "..", "data", "corpus", "corpus.jsonl")
    if not os.path.exists(corpus_file):
        print(f"[!] Corpus file not found at: {corpus_file}")
        sys.exit(1)

    print("=" * 65)
    print("Agentic GraphRAG - TigerGraph Data Ingestion")
    print("=" * 65)
    print(f"Connecting to: {host} (Graph: {graphname})...")

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        username=username,
        password=password,
        gsqlSecret=secret,
        tgCloud=tg_cloud
    )
    if secret:
        conn.getToken(secret=secret)
    else:
        conn.getToken()

    print(f"[✓] Connected to TigerGraph {conn.getVer()}!")

    # Buffers for batch upserts
    doc_vertices = []
    entity_vertices: Dict[str, dict] = {}
    mentions_edges: Set[Tuple[str, str]] = set()
    relation_edges: Set[Tuple[str, str, str]] = set() # (src, dst, rel_type)

    total_docs = 0
    start_time = time.time()

    def flush_batches():
        nonlocal doc_vertices, entity_vertices, mentions_edges, relation_edges
        if doc_vertices:
            conn.upsertVertices("Document", doc_vertices)
            doc_vertices = []
        if entity_vertices:
            entities_list = [(k, v) for k, v in entity_vertices.items()]
            conn.upsertVertices("Entity", entities_list)
            entity_vertices = {}
        if mentions_edges:
            edges_list = [(d_id, e_id, {}) for d_id, e_id in mentions_edges]
            conn.upsertEdges("Document", "MENTIONS", "Entity", edges_list)
            mentions_edges = set()
        if relation_edges:
            edges_list = [(src, dst, {"rel_type": r}) for src, dst, r in relation_edges]
            conn.upsertEdges("Entity", "RELATION", "Entity", edges_list)
            relation_edges = set()

    print(f"\nProcessing corpus from: {corpus_file}")
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = item["doc_id"]
            title = clean_str(item.get("title", ""))
            url = clean_str(item.get("url", ""))
            text = clean_str(item.get("text", ""))
            tokens = int(item.get("approx_tokens", 0))

            # Add document vertex
            doc_vertices.append((
                doc_id,
                {
                    "title": title,
                    "url": url,
                    "text": text[:25000], # store substantial full text
                    "approx_tokens": tokens
                }
            ))

            # 1. Event Entity (the document itself is typically an event)
            event_ent_id = f"Event:{doc_id}"
            entity_vertices[event_ent_id] = {"name": title, "entity_type": "Event"}
            mentions_edges.add((doc_id, event_ent_id))

            # 2. Extract structured entities from Infobox
            infobox = extract_infobox(text)

            # Sport
            sport = extract_sport(title)
            if sport:
                sport_id = f"Sport:{sport}"
                entity_vertices[sport_id] = {"name": sport, "entity_type": "Sport"}
                mentions_edges.add((doc_id, sport_id))
                relation_edges.add((event_ent_id, sport_id, "IN_SPORT"))

            # Games (e.g. '2012 Summer', '2018 Winter Olympics')
            games = clean_str(infobox.get("games", ""))
            if games:
                games_id = f"Games:{games}"
                entity_vertices[games_id] = {"name": games, "entity_type": "Games"}
                mentions_edges.add((doc_id, games_id))
                relation_edges.add((event_ent_id, games_id, "PART_OF_GAMES"))

            # Venue (e.g. 'Eton Dorney')
            venue = clean_str(infobox.get("venue", ""))
            if venue:
                venue_id = f"Venue:{venue}"
                entity_vertices[venue_id] = {"name": venue, "entity_type": "Venue"}
                mentions_edges.add((doc_id, venue_id))
                relation_edges.add((event_ent_id, venue_id, "HELD_AT"))

            # Medalists
            for medal, rel in [("gold", "WON_GOLD"), ("silver", "WON_SILVER"), ("bronze", "WON_BRONZE")]:
                athletes = clean_str(infobox.get(medal, ""))
                if athletes and athletes.lower() != "none":
                    # Simple split if multiple athletes listed
                    ath_id = f"Athlete:{athletes[:64]}"
                    entity_vertices[ath_id] = {"name": athletes, "entity_type": "Athlete"}
                    mentions_edges.add((doc_id, ath_id))
                    relation_edges.add((ath_id, event_ent_id, rel))

            # Countries (goldNOC, silverNOC, bronzeNOC)
            for noc_key in ["goldNOC", "silverNOC", "bronzeNOC"]:
                noc = clean_str(infobox.get(noc_key, ""))
                if noc:
                    country_id = f"Country:{noc}"
                    entity_vertices[country_id] = {"name": noc, "entity_type": "Country"}
                    mentions_edges.add((doc_id, country_id))

            total_docs += 1

            if len(doc_vertices) >= batch_size:
                flush_batches()
                elapsed = time.time() - start_time
                docs_per_sec = total_docs / elapsed if elapsed > 0 else 0
                print(f"  -> Ingested {total_docs} documents ({docs_per_sec:.1f} docs/sec)...")

            if limit and total_docs >= limit:
                print(f"Reached limit of {limit} documents.")
                break

    # Flush remaining items
    flush_batches()
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"Ingestion completed in {elapsed:.1f} seconds! Total docs: {total_docs}")
    print("Checking database counts...")
    try:
        doc_count = conn.getVertexCount("Document")
        ent_count = conn.getVertexCount("Entity")
        print(f"  Document vertices : {doc_count}")
        print(f"  Entity vertices   : {ent_count}")
    except Exception as e:
        print("  (Vertex counts updating in background)")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into TigerGraph")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to ingest")
    parser.add_argument("--batch-size", type=int, default=150, help="Batch size for upserts")
    args = parser.parse_args()

    run_ingest(limit=args.limit, batch_size=args.batch_size)
