"""
Smoke tests. These run entirely in mock mode (no API keys, no real
TigerGraph) so they should pass in a bare checkout — that's the point.
Run with: pytest
"""
import os
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shared.tigergraph_client import MockTigerGraphClient
from src.rag import pipeline as naive_rag
from src.graphrag import pipeline as graph_rag
from src.agentic_graphrag import pipeline as agentic_graphrag
from src.shared.state import InvestigationState
from src.agentic_graphrag.agents.orchestrator import run_investigation


def test_mock_client_has_toy_graph():
    client = MockTigerGraphClient()
    results = client.entity_search("Jane")
    assert any("Jane" in r["name"] for r in results)


def test_naive_rag_runs_end_to_end():
    client = MockTigerGraphClient()
    out = naive_rag.run("Who leads Project Helios?", client)
    assert out["pipeline"] == "naive_rag"
    assert isinstance(out["answer"], str)


def test_graph_rag_runs_end_to_end():
    client = MockTigerGraphClient()
    out = graph_rag.run("Who leads Project Helios?", client)
    assert out["pipeline"] == "graph_rag"
    assert out["evidence_count"] >= 0


def test_agentic_pipeline_runs_end_to_end():
    client = MockTigerGraphClient()
    out = agentic_graphrag.run("Who leads Project Helios and where do they work?", client)
    assert out["pipeline"] == "agentic_graphrag"
    assert "full_trace" in out
    assert out["num_steps"] >= 1


def test_investigation_state_stops_within_max_steps():
    client = MockTigerGraphClient()
    state = InvestigationState(question="Who funded Project Helios?")
    state = run_investigation(state, client)
    assert state.stopped is True
    assert len(state.steps) <= 8
