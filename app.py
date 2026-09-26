"""
Streamlit app — Agentic GraphRAG Hackathon Demo
================================================
A three-panel dashboard that:
  1. Lets you ask a question
  2. Runs all three pipelines (RAG / GraphRAG / Agentic GraphRAG) and streams results
  3. Shows the side-by-side metrics dashboard

Run locally:
  streamlit run app.py

Deploy to Streamlit Cloud:
  Push to GitHub, connect repo at https://share.streamlit.io
  Set secrets in the Streamlit Cloud dashboard (see .env.example)
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic GraphRAG | TigerGraph Hackathon",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Streamlit Secrets into env (for Cloud deployment) ───────────────────
for key in ["TG_HOST", "TG_GRAPHNAME", "TG_SECRET", "TG_USERNAME", "TG_PASSWORD",
            "TG_TGCLOUD", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TG_USE_MOCK"]:
    if hasattr(st, "secrets") and key in st.secrets:
        os.environ[key] = st.secrets[key]

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: #1a1d27; border-radius: 12px; padding: 1rem 1.25rem;
    border: 1px solid #2a2e3a; margin-bottom: 0.75rem;
}
.pipeline-badge-rag    { background:#374151; color:#d1d5db; padding:2px 8px; border-radius:6px; font-size:0.78rem; }
.pipeline-badge-graph  { background:#1d4ed8; color:#bfdbfe; padding:2px 8px; border-radius:6px; font-size:0.78rem; }
.pipeline-badge-agent  { background:#166534; color:#bbf7d0; padding:2px 8px; border-radius:6px; font-size:0.78rem; }
.answer-box {
    background:#111827; border-left:3px solid #22c55e; border-radius:6px;
    padding:1rem 1.25rem; font-size:0.95rem; line-height:1.6; margin-top:0.5rem;
}
.trace-step {
    background:#0f172a; border-radius:6px; padding:0.5rem 0.75rem;
    margin-bottom:0.35rem; font-size:0.82rem; color:#94a3b8; font-family:monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.tigergraph.com/wp-content/uploads/2021/06/TigerGraph-Logo.png", width=160)
    st.title("🐯 Agentic GraphRAG")
    st.caption("TigerGraph Hackathon — Round 1")
    st.divider()

    st.subheader("⚙️ Settings")
    use_mock = st.toggle("Mock mode (no API keys needed)", value=os.getenv("TG_USE_MOCK", "true").lower() == "true")
    os.environ["TG_USE_MOCK"] = "true" if use_mock else "false"

    max_steps = st.slider("Max agentic steps", 2, 12, 6)
    os.environ["MAX_STEPS"] = str(max_steps)

    pipelines_to_run = st.multiselect(
        "Pipelines to run",
        ["RAG", "GraphRAG", "Agentic GraphRAG"],
        default=["RAG", "GraphRAG", "Agentic GraphRAG"]
    )

    st.divider()
    st.subheader("📊 Database Status")
    if st.button("Check TigerGraph Connection", use_container_width=True):
        try:
            from src.shared.tigergraph_client import TigerGraphClient
            client = TigerGraphClient()
            vc = client.conn.getVertexCount("Document")
            ec = client.conn.getVertexCount("Entity")
            st.success(f"✅ Connected!\n- Documents: {vc:,}\n- Entities: {ec:,}")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

    st.divider()
    st.caption("Graph: **AgenticGraphRag** | TG v4.2.5")

# ── Main area ───────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Question Investigation")
st.markdown(
    "Ask any question answerable from the **Olympic events corpus**. "
    "All three pipelines run in parallel and results are compared side-by-side."
)

# Sample questions
SAMPLE_QUESTIONS = [
    "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?",
    "Which country won the most gold medals at the 2012 Summer Olympics in Canoeing?",
    "Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?",
    "According to the corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
]

selected_sample = st.selectbox("📌 Pick a sample question", ["(type your own below...)"] + SAMPLE_QUESTIONS)
question = st.text_area(
    "✏️ Your question",
    value="" if selected_sample.startswith("(") else selected_sample,
    height=80,
    placeholder="Ask something about Olympic events from the corpus..."
)

run_clicked = st.button("🚀 Run Investigation", type="primary", use_container_width=True, disabled=not question.strip())

# ── Results ─────────────────────────────────────────────────────────────────────
if run_clicked and question.strip():
    from src.shared.tigergraph_client import get_client
    import src.rag.pipeline as naive_rag
    import src.graphrag.pipeline as graph_rag
    import src.agentic_graphrag.pipeline as agentic_graphrag

    client = get_client()

    results = {}
    col1, col2, col3 = st.columns(3)
    cols = {
        "RAG": col1,
        "GraphRAG": col2,
        "Agentic GraphRAG": col3,
    }
    placeholders = {}

    for pipeline_name in ["RAG", "GraphRAG", "Agentic GraphRAG"]:
        with cols[pipeline_name]:
            badge_cls = {"RAG": "pipeline-badge-rag", "GraphRAG": "pipeline-badge-graph", "Agentic GraphRAG": "pipeline-badge-agent"}[pipeline_name]
            st.markdown(f'<span class="{badge_cls}">{pipeline_name}</span>', unsafe_allow_html=True)
            placeholders[pipeline_name] = st.empty()
            placeholders[pipeline_name].info("⏳ Waiting...")

    # Run selected pipelines
    pipeline_map = {
        "RAG": lambda q, c: naive_rag.run(q, c),
        "GraphRAG": lambda q, c: graph_rag.run(q, c),
        "Agentic GraphRAG": lambda q, c: agentic_graphrag.run(q, c),
    }

    for pipeline_name in pipelines_to_run:
        with cols[pipeline_name]:
            placeholders[pipeline_name].info(f"🔄 Running {pipeline_name}...")

        t0 = time.time()
        out = pipeline_map[pipeline_name](question, client)
        results[pipeline_name] = out

        with cols[pipeline_name]:
            placeholders[pipeline_name].empty()
            with placeholders[pipeline_name].container():
                st.markdown(f'<div class="answer-box">{out["answer"]}</div>', unsafe_allow_html=True)
                st.caption(f"⏱ {out.get('latency_seconds', time.time()-t0):.2f}s · 🪙 {out.get('tokens_used', 0):,} tokens")

                # Show agentic trace
                if pipeline_name == "Agentic GraphRAG" and "full_trace" in out:
                    with st.expander(f"🔬 Investigation Trace ({out.get('num_steps', 0)} steps)"):
                        for step in out.get("full_trace", []):
                            action = step.get("action", step.get("source_action", "step"))
                            content = step.get("content", "")[:180]
                            st.markdown(f'<div class="trace-step">→ <b>{action}</b>: {content}</div>', unsafe_allow_html=True)

    # ── Metrics Comparison ──────────────────────────────────────────────────────
    if len(results) > 1:
        st.divider()
        st.subheader("📊 Pipeline Comparison")

        pipeline_colors = {
            "RAG": "#6b7280",
            "GraphRAG": "#3b82f6",
            "Agentic GraphRAG": "#22c55e",
        }

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Fastest", min(results, key=lambda k: results[k].get("latency_seconds", 9999)),
                      delta=None)
        with m2:
            st.metric("Fewest Tokens", min(results, key=lambda k: results[k].get("tokens_used", 9999)))
        with m3:
            st.metric("Most Evidence", max(results, key=lambda k: results[k].get("evidence_count", 0)))

        # Token bar chart
        import pandas as pd
        df_tokens = pd.DataFrame([
            {"Pipeline": k, "Tokens": v.get("tokens_used", 0), "Latency (s)": v.get("latency_seconds", 0)}
            for k, v in results.items()
        ])
        st.bar_chart(df_tokens.set_index("Pipeline")["Tokens"])

    # ── Download raw outputs ────────────────────────────────────────────────────
    st.divider()
    st.download_button(
        "⬇️ Download Raw Results JSON",
        data=json.dumps({"question": question, "results": {k: {**v} for k, v in results.items()}}, indent=2, default=str),
        file_name="pipeline_results.json",
        mime="application/json",
    )

elif not run_clicked:
    # Landing state
    st.info("👆 Select or type a question above, then click **Run Investigation** to compare all three pipelines.")

    st.divider()
    st.subheader("📈 How it Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔎 RAG (Baseline)**\n\nVector similarity search → retrieve relevant text chunks → generate answer. No graph structure used.")
    with c2:
        st.markdown("**🕸 GraphRAG**\n\nEntity linking → multi-hop graph traversal → vector search → generate. Graph structure adds relational context.")
    with c3:
        st.markdown("**🤖 Agentic GraphRAG**\n\nOrchestrator decides next action based on evidence state. Iterates until confident or max steps reached. Proves *when* agents add value.")
