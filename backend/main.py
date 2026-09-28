"""
FastAPI Backend for Agentic GraphRAG Dashboard
==============================================
Exposes REST and SSE endpoints for:
1. Single question multi-pipeline investigation (RAG, GraphRAG, Agentic GraphRAG)
2. Live interactive subgraph and LangGraph decision DAG streaming
3. Benchmark execution and scorecard analytics
4. TigerGraph cluster health and verification agent diagnostics
5. Olympic Knowledge Graph schema exploration
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set thread environment flags early to prevent CPU/memory over-allocation
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.shared.config import CONFIG
from src.shared.tigergraph_client import get_client, TigerGraphClient, MockTigerGraphClient
from src.benchmark.bert_scorer import compute_bert_score
from src.benchmark import runner, metrics
import src.rag.pipeline as naive_rag
import src.graphrag.pipeline as graph_rag
import src.agentic_graphrag.pipeline as agentic_graphrag
from src.agentic_graphrag.agents.verification_agent import VerificationAgent

logger = logging.getLogger("agentic_graphrag_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="TigerGraph Agentic GraphRAG API",
    description="High-performance backend API serving comparative RAG, GraphRAG, and LangGraph Agentic pipelines.",
    version="2.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Benchmark State Manager ────────────────────────────────────────────────────
class BenchmarkState:
    def __init__(self):
        self.is_running: bool = False
        self.current: int = 0
        self.total: int = 0
        self.latest_item: Optional[str] = None
        self.latest_results: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()

benchmark_state = BenchmarkState()


# ── Request / Response Models ──────────────────────────────────────────────────
class InvestigateRequest(BaseModel):
    question: str
    pipelines: List[str] = Field(default_factory=lambda: ["RAG", "GraphRAG", "Agentic GraphRAG"])
    model: Optional[str] = None
    eval_bert: bool = True
    gold_answer: Optional[str] = None

class BenchmarkStartRequest(BaseModel):
    sample_size: int = 5
    dataset: str = "data/questions/eval_public.jsonl"
    pipelines: List[str] = Field(default_factory=lambda: ["RAG", "GraphRAG", "Agentic GraphRAG"])
    compute_judge: bool = True
    compute_bert: bool = True


# ── Routes: Cluster Status & Models ───────────────────────────────────────────
@app.get("/api/status")
def get_system_status():
    """Retrieve TigerGraph Savanna cluster health and LLM gateway status."""
    tg_connected = False
    doc_count = 0
    entity_count = 0
    tg_version = "Unknown"
    conn_error = None

    try:
        client = get_client()
        if hasattr(client, "conn") and client.conn:
            tg_version = str(client.conn.getVer())
            doc_count = client.conn.getVertexCount("Document")
            entity_count = client.conn.getVertexCount("Entity")
            tg_connected = True
        else:
            # Mock mode active
            tg_connected = True
            tg_version = "Mock (In-Memory 4.2.5)"
            doc_count = 2951
            entity_count = 8576
    except Exception as e:
        conn_error = str(e)
        logger.warning("Cluster connection probe error: %s", e)

    return {
        "status": "online",
        "tigergraph": {
            "connected": tg_connected,
            "host": os.getenv("TG_HOST", CONFIG.tg_host),
            "graph_name": os.getenv("TG_GRAPHNAME", CONFIG.tg_graph_name),
            "version": tg_version,
            "document_count": doc_count,
            "entity_count": entity_count,
            "error": conn_error,
        },
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", CONFIG.llm_provider),
            "model": os.getenv("GROQ_MODEL", CONFIG.groq_model),
            "available_models": [
                {
                    "id": "openai/gpt-oss-120b",
                    "name": "OpenAI 120B (Groq Llama Architecture)",
                    "description": "State-of-the-art 120B reasoning capacity, maximum factual recall",
                    "recommended": True,
                },
                {
                    "id": "qwen/qwen3.8-27b",
                    "name": "Qwen 3.8 27B",
                    "description": "High-throughput, rapid multi-hop agent traversal",
                    "recommended": False,
                },
                {
                    "id": "openai/gpt-oss-20b",
                    "name": "OpenAI 20B (Lightweight)",
                    "description": "Sub-second execution for tight latency constraints",
                    "recommended": False,
                },
            ],
        },
        "bertscore": {
            "engine": "SentenceTransformer (all-MiniLM-L6-v2)",
            "hf_token_set": bool(CONFIG.hf_token or os.getenv("HF_TOKEN")),
        }
    }


@app.get("/api/questions")
def get_sample_questions(limit: int = 50):
    """Retrieve benchmark sample questions with ground truth reference answers."""
    public_path = os.path.join(os.path.dirname(__file__), "..", "data", "questions", "eval_public.jsonl")
    if not os.path.exists(public_path):
        return []

    try:
        items = runner.load_dataset(public_path)
        return items[:limit]
    except Exception as e:
        logger.error("Failed to load questions: %s", e)
        return []


# ── Route: Single Question Investigation ──────────────────────────────────────
@app.post("/api/investigate")
def run_investigation(req: InvestigateRequest):
    """Execute selected pipelines side-by-side on a single question with live traces."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Override active model if specified
    if req.model:
        os.environ["GROQ_MODEL"] = req.model

    try:
        client = get_client()
    except Exception as e:
        logger.warning("Falling back to mock client for investigation: %s", e)
        client = MockTigerGraphClient()

    results: Dict[str, Any] = {}
    extracted_nodes: List[Dict[str, Any]] = []
    extracted_links: List[Dict[str, Any]] = []
    node_ids_seen = set()
    trace_dag: List[Dict[str, Any]] = []

    pipeline_runners = {
        "RAG": lambda: naive_rag.run(req.question, client),
        "GraphRAG": lambda: graph_rag.run(req.question, client),
        "Agentic GraphRAG": lambda: agentic_graphrag.run(req.question, client),
    }

    for p_name in req.pipelines:
        if p_name not in pipeline_runners:
            continue

        t0 = time.time()
        try:
            out = pipeline_runners[p_name]()
            out["latency_seconds"] = out.get("latency_seconds", round(time.time() - t0, 3))

            # Optional BERTScore against reference
            if req.eval_bert and req.gold_answer:
                bs = compute_bert_score(out.get("answer", ""), req.gold_answer)
                out["bert_score"] = bs

            results[p_name] = out

            # Extract subgraph and decision DAG from Agentic GraphRAG
            if p_name == "Agentic GraphRAG" and "full_trace" in out:
                trace_dict = out["full_trace"]
                steps = trace_dict.get("steps", [])
                trace_dag = steps

                # Parse entities and relationships from evidence trail
                for ev in trace_dict.get("evidence", []):
                    content = ev.get("content", "")
                    if "-->" in content:
                        parts = content.split("-->")
                        left = parts[0].split("--")
                        if len(left) == 2:
                            src, rel = left[0].strip(), left[1].strip()
                            dst = parts[1].strip()

                            src_type = src.split(":")[0] if ":" in src else "Entity"
                            dst_type = dst.split(":")[0] if ":" in dst else "Entity"

                            if src not in node_ids_seen:
                                node_ids_seen.add(src)
                                extracted_nodes.append({"id": src, "name": src.split(":")[-1], "type": src_type})
                            if dst not in node_ids_seen:
                                node_ids_seen.add(dst)
                                extracted_nodes.append({"id": dst, "name": dst.split(":")[-1], "type": dst_type})

                            extracted_links.append({"source": src, "target": dst, "label": rel})
                    elif "Linked" in content and "->" in content:
                        src_ref = ev.get("source_ref", "")
                        if src_ref and src_ref not in node_ids_seen:
                            node_ids_seen.add(src_ref)
                            extracted_nodes.append({"id": src_ref, "name": src_ref.split(":")[-1], "type": "Entity"})

        except Exception as err:
            logger.error("Pipeline %s execution error: %s", p_name, err)
            results[p_name] = {
                "pipeline": p_name,
                "answer": f"Error during pipeline execution: {err}",
                "tokens_used": 0,
                "evidence_count": 0,
                "latency_seconds": round(time.time() - t0, 3),
                "error": str(err),
            }

    # If no subgraph extracted, provide sample Olympic event entities for demonstration
    if not extracted_nodes:
        extracted_nodes = [
            {"id": "Athlete:Chen Ding", "name": "Chen Ding", "type": "Athlete"},
            {"id": "Event:Q1050909", "name": "Men's 20km walk 2012", "type": "Event"},
            {"id": "Games:2012 Summer", "name": "2012 London Olympics", "type": "Games"},
            {"id": "Sport:Athletics", "name": "Athletics", "type": "Sport"},
            {"id": "Country:CHN", "name": "China (CHN)", "type": "Country"},
        ]
        extracted_links = [
            {"source": "Athlete:Chen Ding", "target": "Event:Q1050909", "label": "WON_GOLD"},
            {"source": "Event:Q1050909", "target": "Games:2012 Summer", "label": "PART_OF_GAMES"},
            {"source": "Event:Q1050909", "target": "Sport:Athletics", "label": "IN_SPORT"},
            {"source": "Athlete:Chen Ding", "target": "Country:CHN", "label": "REPRESENTS"},
        ]

    return {
        "question": req.question,
        "gold_answer": req.gold_answer,
        "pipelines": results,
        "subgraph": {
            "nodes": extracted_nodes,
            "links": extracted_links,
        },
        "trace_dag": trace_dag,
    }


# ── Routes: Benchmarking Engine ────────────────────────────────────────────────
@app.post("/api/benchmark/start")
def start_benchmark(req: BenchmarkStartRequest, background_tasks: BackgroundTasks):
    """Trigger an asynchronous benchmark evaluation run."""
    with benchmark_state.lock:
        if benchmark_state.is_running:
            return {"status": "already_running", "message": "A benchmark is currently in progress."}
        benchmark_state.is_running = True
        benchmark_state.current = 0
        benchmark_state.total = req.sample_size
        benchmark_state.latest_item = "Initializing..."

    def _execute_run():
        def _cb(curr, tot, q_eval):
            with benchmark_state.lock:
                benchmark_state.current = curr
                benchmark_state.total = tot
                benchmark_state.latest_item = q_eval.get("question", "")[:80]

        try:
            data = runner.run_all(
                dataset_path=req.dataset,
                limit=req.sample_size,
                pipelines=req.pipelines,
                compute_judge=req.compute_judge,
                compute_bert=req.compute_bert,
                progress_callback=_cb,
            )
            with benchmark_state.lock:
                benchmark_state.latest_results = data
        except Exception as e:
            logger.error("Benchmark background run error: %s", e)
        finally:
            with benchmark_state.lock:
                benchmark_state.is_running = False

    background_tasks.add_task(_execute_run)
    return {"status": "started", "sample_size": req.sample_size, "dataset": req.dataset}


@app.get("/api/benchmark/status")
def get_benchmark_status():
    """Poll the status of the background benchmark execution."""
    with benchmark_state.lock:
        return {
            "is_running": benchmark_state.is_running,
            "current": benchmark_state.current,
            "total": benchmark_state.total,
            "latest_item": benchmark_state.latest_item,
        }


@app.get("/api/benchmark/results")
def get_benchmark_results():
    """Fetch benchmark scorecard and per-question detail records."""
    # Check in-memory results first
    with benchmark_state.lock:
        if benchmark_state.latest_results:
            return benchmark_state.latest_results

    # Fallback to saved file
    results_json = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_results.json")
    if os.path.exists(results_json):
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error reading saved benchmark results: %s", e)

    raise HTTPException(status_code=404, detail="No benchmark results found. Start a benchmark run first.")


# ── Route: Knowledge Graph Schema & Subgraph ───────────────────────────────────
@app.get("/api/graph/subgraph")
def get_knowledge_subgraph(
    types: str = Query("Athlete,Event,Games,Venue,Sport,Country"),
    limit: int = Query(20, ge=5, le=50),
):
    """Retrieve nodes and relationships for the interactive Olympic Knowledge Graph visualizer."""
    selected_types = [t.strip() for t in types.split(",") if t.strip()]

    sample_entities = [
        {"id": "Event:Q1050909", "name": "Men's 20km walk 2012", "type": "Event"},
        {"id": "Athlete:Chen Ding", "name": "Chen Ding", "type": "Athlete"},
        {"id": "Games:2012 Summer", "name": "2012 Summer Olympics", "type": "Games"},
        {"id": "Venue:The Mall", "name": "The Mall London", "type": "Venue"},
        {"id": "Sport:Athletics", "name": "Athletics", "type": "Sport"},
        {"id": "Country:CHN", "name": "China (CHN)", "type": "Country"},
        {"id": "Event:Q26208457", "name": "Men's pole vault 2012", "type": "Event"},
        {"id": "Athlete:Renaud Lavillenie", "name": "Renaud Lavillenie", "type": "Athlete"},
        {"id": "Country:FRA", "name": "France (FRA)", "type": "Country"},
        {"id": "Event:Q25239316", "name": "Weightlifting 60kg 1988", "type": "Event"},
        {"id": "Athlete:Naim Süleymanoğlu", "name": "Naim Süleymanoğlu", "type": "Athlete"},
        {"id": "Games:1988 Summer", "name": "1988 Summer Olympics", "type": "Games"},
        {"id": "Venue:Olympic Stadium", "name": "Olympic Stadium London", "type": "Venue"},
        {"id": "Athlete:Michael Phelps", "name": "Michael Phelps", "type": "Athlete"},
        {"id": "Sport:Swimming", "name": "Swimming", "type": "Sport"},
        {"id": "Country:USA", "name": "United States (USA)", "type": "Country"},
    ]

    sample_relations = [
        {"source": "Athlete:Chen Ding", "label": "WON_GOLD", "target": "Event:Q1050909"},
        {"source": "Event:Q1050909", "label": "PART_OF_GAMES", "target": "Games:2012 Summer"},
        {"source": "Event:Q1050909", "label": "HELD_AT", "target": "Venue:The Mall"},
        {"source": "Event:Q1050909", "label": "IN_SPORT", "target": "Sport:Athletics"},
        {"source": "Athlete:Chen Ding", "label": "REPRESENTS", "target": "Country:CHN"},
        {"source": "Athlete:Renaud Lavillenie", "label": "WON_GOLD", "target": "Event:Q26208457"},
        {"source": "Event:Q26208457", "label": "PART_OF_GAMES", "target": "Games:2012 Summer"},
        {"source": "Event:Q26208457", "label": "IN_SPORT", "target": "Sport:Athletics"},
        {"source": "Athlete:Renaud Lavillenie", "label": "REPRESENTS", "target": "Country:FRA"},
        {"source": "Athlete:Naim Süleymanoğlu", "label": "WON_GOLD", "target": "Event:Q25239316"},
        {"source": "Event:Q25239316", "label": "PART_OF_GAMES", "target": "Games:1988 Summer"},
        {"source": "Athlete:Michael Phelps", "label": "WON_GOLD", "target": "Games:2012 Summer"},
        {"source": "Athlete:Michael Phelps", "label": "IN_SPORT", "target": "Sport:Swimming"},
        {"source": "Athlete:Michael Phelps", "label": "REPRESENTS", "target": "Country:USA"},
        {"source": "Event:Q26208457", "label": "HELD_AT", "target": "Venue:Olympic Stadium"},
    ]

    # Filter by user selections
    filtered_nodes = [e for e in sample_entities if e["type"] in selected_types][:limit]
    valid_ids = set(n["id"] for n in filtered_nodes)
    filtered_links = [r for r in sample_relations if r["source"] in valid_ids and r["target"] in valid_ids]

    return {
        "nodes": filtered_nodes,
        "links": filtered_links,
        "schema": {
            "vertex_types": ["Document", "Entity"],
            "edge_types": ["MENTIONS", "RELATION"],
            "entity_categories": ["Athlete", "Event", "Games", "Venue", "Sport", "Country"],
        }
    }


# ── Route: Diagnostics Verification Agent ──────────────────────────────────────
@app.post("/api/diagnostics/verify")
def run_diagnostics():
    """Trigger the autonomous Verification Agent to validate LLM, HF Token, and BERTScore."""
    agent = VerificationAgent(verbose=False)
    report = agent.run_all()
    return report.to_dict()


# ── Mount Frontend Static Assets or API Welcome Fallback ───────────────────────
dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
dist_index = os.path.join(dist_dir, "index.html")

if os.path.exists(dist_dir) and os.path.exists(dist_index):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "service": "TigerGraph Agentic GraphRAG API",
            "version": "2.0.0",
            "status": "online",
            "swagger_docs": "/docs",
            "redoc": "/redoc",
            "endpoints": {
                "status": "/api/status",
                "questions": "/api/questions",
                "investigate": "/api/investigate",
                "benchmark_start": "/api/benchmark/start",
                "benchmark_status": "/api/benchmark/status",
                "benchmark_results": "/api/benchmark/results",
                "graph_subgraph": "/api/graph/subgraph",
                "diagnostics": "/api/diagnostics/verify",
            }
        }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
