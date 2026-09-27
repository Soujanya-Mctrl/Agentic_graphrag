"""
Agentic GraphRAG — TigerGraph Hackathon Interactive Dashboard
=============================================================
A white-themed editorial Streamlit application featuring:
    1. Single Question Deep-Dive with Live Graph Subgraphs & LangGraph Trace DAGs
    2. Integrated Benchmarking & Evaluation Engine (LLM-as-a-Judge + BERTScore)
    3. Time (Latency) vs. Token Usage Scatter Plots & Statistical Charts
    4. Interactive Knowledge Graph Explorer (Plotly + NetworkX)
    5. Architecture & Metric Methodology Reference

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load env and append project root
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.shared.config import CONFIG
from src.shared.tigergraph_client import get_client
from src.benchmark import metrics
from src.benchmark import runner
from src.benchmark.bert_scorer import compute_bert_score
from src.benchmark.visualizer import (
        plot_time_vs_tokens,
        plot_metric_bars,
        plot_category_breakdown,
        plot_knowledge_subgraph,
        plot_agentic_trace_dag,
)
import src.rag.pipeline as naive_rag
import src.graphrag.pipeline as graph_rag
import src.agentic_graphrag.pipeline as agentic_graphrag

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic GraphRAG | TigerGraph Hackathon",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Streamlit Secrets into environment ──────────────────────────────────
for key in [
    "TG_HOST", "TG_GRAPHNAME", "TG_SECRET", "TG_USERNAME", "TG_PASSWORD",
    "TG_TGCLOUD", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TG_USE_MOCK",
    "GROQ_MODEL", "LLM_PROVIDER", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN",
]:
    if hasattr(st, "secrets") and key in st.secrets:
        os.environ[key] = str(st.secrets[key])

# ── Custom CSS: Editorial White Theme (Serif + Sans-Serif Font Pairing) ────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,600&family=Playfair+Display:ital,wght@0,600;0,700;1,400;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base canvas */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Serif Headings */
h1, h2, h3, .serif-title, .serif-quote {
    font-family: 'Newsreader', 'Playfair Display', Georgia, serif !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
}

h1 { font-size: 2.35rem !important; margin-bottom: 0.25rem !important; }
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
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="serif-title" style="font-size:1.6rem; color:#ea580c; margin-bottom:0;">🐯 Agentic GraphRAG</div>', unsafe_allow_html=True)
    st.caption("TigerGraph Hackathon — Evaluation & Investigation Suite")
    st.divider()

    st.subheader("⚙️ System Configuration")
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    any_key = has_groq or has_anthropic or has_openai

    # Auto-disable mock if a real key is present
    default_mock = not any_key
    use_mock = st.toggle("Mock mode (Zero API keys needed)", value=os.getenv("TG_USE_MOCK", str(default_mock)).lower() == "true")
    os.environ["TG_USE_MOCK"] = "true" if use_mock else "false"

    max_steps = st.slider("Max agentic investigation steps", min_value=2, max_value=12, value=int(os.getenv("MAX_STEPS", "8")))
    os.environ["MAX_STEPS"] = str(max_steps)

    # Determine active provider
    llm_provider = os.getenv("LLM_PROVIDER", "groq" if has_groq else "anthropic" if has_anthropic else "openai")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if has_groq:
        st.markdown(
            f"<div class='white-card' style='padding:0.75rem 1rem; margin-bottom:0.5rem;'>"
            f"<div class='kpi-title'>⚡ LLM Gateway — Active</div>"
            f"<div style='font-family:Newsreader,serif; font-size:1.1rem; font-weight:700; color:#059669;'>Groq</div>"
            f"<div class='kpi-sub'><code>{groq_model}</code></div>"
            f"</div>",
            unsafe_allow_html=True
        )
    elif has_anthropic:
        st.markdown(f"**LLM Gateway**: `Anthropic`")
    elif has_openai:
        st.markdown(f"**LLM Gateway**: `OpenAI`")
    else:
        st.warning("⚠️ No LLM API key detected. Switch to Mock mode or configure `.env`.")

    st.divider()
    st.subheader("🌐 TigerGraph Savanna")
    tg_host = os.getenv("TG_HOST", "https://your-instance.i.tgcloud.io")
    st.markdown(f"<span class='badge badge-tg'>{os.getenv('TG_GRAPHNAME', 'AgenticGraphRag')}</span>", unsafe_allow_html=True)
    st.caption(f"Host: `{tg_host[:35]}...`" if len(tg_host) > 35 else f"Host: `{tg_host}`")

    if st.button("🔌 Test Savanna Connection", use_container_width=True):
        try:
            client = get_client()
            if hasattr(client, "conn"):
                vc = client.conn.getVertexCount("Document")
                ec = client.conn.getVertexCount("Entity")
                st.success(f"✅ Connected to Savanna!\n- Documents: {vc:,}\n- Entities: {ec:,}")
            else:
                st.info("ℹ️ Running in Mock Mode (In-Memory Toy Graph Active)")
        except Exception as e:
            st.error(f"❌ Connection error: {e}")

    st.divider()
    st.markdown("""
    <div style="font-size:0.8rem; color:#64748b; line-height:1.5;">
    <b>TigerGraph Round 1 Deliverables:</b><br>
    - Working Agentic GraphRAG<br>
    - 3-Way Comparative Benchmark<br>
    - Accuracy, Completeness & BERTScore<br>
    - Interactive Subgraph Visualization
    </div>
    """, unsafe_allow_html=True)


# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div>
  <span class="badge badge-tg">TigerGraph Hackathon 2026</span>
  <h1 style="margin-top:0.4rem;">Autonomous Graph Investigation & Benchmark Engine</h1>
  <p class="hero-subtitle">
    Benchmarking <b>Agentic GraphRAG</b> against <b>Fixed-Sequence GraphRAG</b> and <b>Naive RAG</b> on the Olympic Knowledge Corpus.
    Auditing when dynamic, multi-step agentic planning justifies additional token investment.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation Tabs ────────────────────────────────────────────────────────────
tab_explore, tab_benchmark, tab_graph, tab_arch = st.tabs([
    "🔍 Single Question Investigation",
    "🏆 Benchmarking & Evaluation Engine",
    "🕸 Knowledge Graph & Subgraph Explorer",
    "📐 Architecture & Methodology",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE QUESTION INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### Interactive Single-Question Investigation")
    st.markdown("Run all three retrieval pipelines side-by-side on any question and inspect the autonomous investigation trace.")

    # Load sample questions from eval_public.jsonl
    sample_dataset = []
    public_path = os.path.join(os.path.dirname(__file__), "data", "questions", "eval_public.jsonl")
    if os.path.exists(public_path):
        try:
            sample_dataset = runner.load_dataset(public_path)[:25]
        except Exception:
            sample_dataset = []

    sample_options = ["(Custom Question...)"]
    sample_map = {}
    for item in sample_dataset:
        lbl = f"[{item.get('qtype', 'general').upper()}] {item['question'][:85]}..."
        sample_options.append(lbl)
        sample_map[lbl] = item

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        chosen_sample = st.selectbox("📌 Pick from Benchmark Dataset (or type below)", sample_options)
    with col_q2:
        gold_display = sample_map[chosen_sample]["gold_answer"] if chosen_sample in sample_map else ""
        st.markdown("**Gold Reference Answer**")
        st.caption(gold_display or "(Type below)")

    default_text = sample_map[chosen_sample]["question"] if chosen_sample in sample_map else "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?"
    user_question = st.text_area(
        "✏️ Question to Investigate",
        value=default_text,
        height=70,
        placeholder="Enter your question regarding Olympic events...",
    )

    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 2, 2])
    with c_btn1:
        run_single_btn = st.button("🚀 Investigate Question", type="primary", use_container_width=True, disabled=not user_question.strip())
    with c_btn2:
        eval_with_bert = st.checkbox("Compute BERTScore against Gold Answer", value=True)
    with c_btn3:
        target_pipelines = st.multiselect(
            "Pipelines to execute",
            ["RAG", "GraphRAG", "Agentic GraphRAG"],
            default=["RAG", "GraphRAG", "Agentic GraphRAG"],
        )

    if run_single_btn and user_question.strip():
        client = get_client()
        gold_ans = sample_map.get(chosen_sample, {}).get("gold_answer", "")

        st.divider()
        st.markdown(f"#### Results for: *\"{user_question}\"*")

        res_cols = st.columns(len(target_pipelines))
        outputs = {}

        pipeline_runners = {
            "RAG": lambda: naive_rag.run(user_question, client),
            "GraphRAG": lambda: graph_rag.run(user_question, client),
            "Agentic GraphRAG": lambda: agentic_graphrag.run(user_question, client),
        }

        # Progress indicators
        for idx, p_name in enumerate(target_pipelines):
            with res_cols[idx]:
                badge_class = {"RAG": "badge-rag", "GraphRAG": "badge-graph", "Agentic GraphRAG": "badge-agent"}[p_name]
                st.markdown(f'<span class="badge {badge_class}">{p_name}</span>', unsafe_allow_html=True)
                with st.spinner(f"Running {p_name}..."):
                    t0 = time.time()
                    out = pipeline_runners[p_name]()
                    out["latency_seconds"] = out.get("latency_seconds", time.time() - t0)
                    outputs[p_name] = out

                # Answer Card
                box_class = {"RAG": "answer-box-rag", "GraphRAG": "answer-box-graph", "Agentic GraphRAG": "answer-box-agent"}[p_name]
                st.markdown(f'<div class="answer-box {box_class}">{out["answer"]}</div>', unsafe_allow_html=True)

                # Metrics under answer
                bert_info = ""
                if eval_with_bert and gold_ans:
                    bs = compute_bert_score(out["answer"], gold_ans)
                    bert_info = f" · 🎯 <b>BERT F1:</b> {bs['bert_f1']:.2f}"

                st.markdown(
                    f"<div class='kpi-sub' style='margin-top:0.6rem;'>"
                    f"⏱ <b>{out['latency_seconds']:.2f}s</b> · "
                    f"🪙 <b>{out.get('tokens_used', 0):,}</b> tokens · "
                    f"📦 <b>{out.get('evidence_count', 0)}</b> items{bert_info}"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Visual Investigation Trace & Subgraph ──────────────────────────────
        st.divider()
        st.markdown("### 🔬 Deep Dive: Subgraph & Investigation Trail")
        trace_col1, trace_col2 = st.columns([1, 1])

        # Subgraph visualization
        with trace_col1:
            st.markdown("##### 🕸 Retrieved Knowledge Subgraph")
            extracted_entities = []
            extracted_relations = []

            # If Agentic or GraphRAG has graph entities, extract them
            if "Agentic GraphRAG" in outputs and "full_trace" in outputs["Agentic GraphRAG"]:
                trace_dict = outputs["Agentic GraphRAG"]["full_trace"]
                for ev in trace_dict.get("evidence", []):
                    content = ev.get("content", "")
                    if "-->" in content:
                        parts = content.split("-->")
                        left = parts[0].split("--")
                        if len(left) == 2:
                            src, rel = left[0].strip(), left[1].strip()
                            dst = parts[1].strip()
                            extracted_relations.append({"from": src, "relation": rel, "to": dst})
                    elif "Linked" in content and "->" in content:
                        src_ref = ev.get("source_ref", "")
                        extracted_entities.append({"id": src_ref, "name": src_ref, "type": "Entity"})

            subgraph_fig = plot_knowledge_subgraph(extracted_entities, extracted_relations, title="Retrieved Subgraph for Query")
            st.plotly_chart(subgraph_fig, use_container_width=True)

        # LangGraph Trace DAG
        with trace_col2:
            st.markdown("##### 🤖 Agentic Decision Trail (LangGraph DAG)")
            agentic_out = outputs.get("Agentic GraphRAG")
            if agentic_out and "full_trace" in agentic_out:
                trace_steps = agentic_out["full_trace"].get("steps", [])
                trace_fig = plot_agentic_trace_dag(trace_steps)
                st.plotly_chart(trace_fig, use_container_width=True)

                with st.expander(f"📋 Step-by-Step Audit Log ({len(trace_steps)} steps)", expanded=False):
                    for s in trace_steps:
                        st.markdown(
                            f"<div class='trace-row'>"
                            f"<b>Step {s['step_index']+1}</b>: <span class='trace-mono'>{s['action']}</span> "
                            f"— {s.get('rationale', '')}<br>"
                            f"<span style='color:#64748b; font-size:0.8rem;'>Input: {s.get('action_input', {})} · Tokens: {s.get('tokens_used', 0)}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.info("Run Agentic GraphRAG to view dynamic decision steps.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BENCHMARKING & EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown("### 🏆 Comprehensive Benchmark & Evaluation Engine")
    st.markdown(
        "Execute automated evaluations across public or hidden datasets. "
        "Scores each pipeline using **LLM-as-a-Judge** (Accuracy & Completeness) alongside **BERTScore** (Precision, Recall, F1)."
    )

    b_col1, b_col2, b_col3, b_col4 = st.columns([1.5, 1.2, 1.2, 1.2])
    with b_col1:
        dataset_choice = st.selectbox(
            "Select Evaluation Dataset",
            [
                "data/questions/eval_public.jsonl (100 Questions)",
                "data/questions/eval_hidden.jsonl (50 Questions)",
                "data/sample_questions.json (Toy Mock)",
            ],
            index=0,
        )
        dataset_file = dataset_choice.split()[0]

    with b_col2:
        eval_sample_size = st.slider("Sample Questions", min_value=1, max_value=100, value=5)

    with b_col3:
        b_pipelines = st.multiselect(
            "Pipelines to benchmark",
            ["RAG", "GraphRAG", "Agentic GraphRAG"],
            default=["RAG", "GraphRAG", "Agentic GraphRAG"],
        )

    with b_col4:
        st.markdown("**Evaluator Suite**")
        eval_judge = st.checkbox("LLM-as-a-Judge", value=True)
        eval_bert = st.checkbox("BERTScore F1", value=True)

    c_action1, c_action2 = st.columns([2, 2])
    with c_action1:
        start_benchmark_btn = st.button("🚀 Start Benchmark Run", type="primary", use_container_width=True)
    with c_action2:
        load_existing_btn = st.button("📂 Load Saved Benchmark Results", use_container_width=True)

    benchmark_data = None
    results_json_path = os.path.join(os.path.dirname(__file__), "results", "benchmark_results.json")

    # ── Running Live Benchmark ─────────────────────────────────────────────────
    if start_benchmark_btn:
        progress_bar = st.progress(0, text="Initializing benchmark run...")
        status_box = st.empty()

        def update_progress(current, total, q_eval):
            pct = int((current / total) * 100)
            q_snippet = q_eval['question'][:65]
            progress_bar.progress(pct, text=f"Evaluating {current}/{total}: {q_snippet}...")
            status_box.caption(f"Completed Question {current}/{total} — {q_eval.get('qtype', '')}")

        with st.spinner("Executing benchmark across selected pipelines..."):
            benchmark_data = runner.run_all(
                dataset_path=dataset_file,
                limit=eval_sample_size,
                pipelines=b_pipelines,
                compute_judge=eval_judge,
                compute_bert=eval_bert,
                progress_callback=update_progress,
            )
        progress_bar.empty()
        status_box.empty()
        st.success(f"✅ Benchmark completed successfully over {eval_sample_size} questions!")

    # ── Loading Existing Benchmark ─────────────────────────────────────────────
    elif load_existing_btn:
        if os.path.exists(results_json_path):
            with open(results_json_path, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
            st.success(f"✅ Loaded saved benchmark results from `{results_json_path}`")
        else:
            st.warning(f"No saved results found at `{results_json_path}`. Run a benchmark first.")

    # ── Display Benchmark Dashboard ────────────────────────────────────────────
    if benchmark_data or os.path.exists(results_json_path):
        if not benchmark_data:
            with open(results_json_path, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)

        summary = benchmark_data.get("summary", {})
        per_q_results = benchmark_data.get("per_question_results", [])
        df_results = pd.DataFrame(per_q_results)

        st.divider()
        st.markdown("### 📊 Benchmark Scorecard")

        # Top KPI Cards
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            agent_acc = summary.get("agentic_graphrag", {}).get("mean_accuracy", 0.0)
            rag_acc = summary.get("naive_rag", {}).get("mean_accuracy", 0.0)
            delta_acc = agent_acc - rag_acc
            st.markdown(
                f"<div class='white-card'>"
                f"<div class='kpi-title'>Agentic Accuracy (LLM)</div>"
                f"<div class='kpi-value'>{agent_acc:.1%}</div>"
                f"<div class='kpi-sub'>Δ vs Naive RAG: <b>{'+' if delta_acc>=0 else ''}{delta_acc:.1%}</b></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with kpi_cols[1]:
            agent_bert = summary.get("agentic_graphrag", {}).get("mean_bert_f1", 0.0)
            rag_bert = summary.get("naive_rag", {}).get("mean_bert_f1", 0.0)
            delta_bert = agent_bert - rag_bert
            st.markdown(
                f"<div class='white-card'>"
                f"<div class='kpi-title'>Agentic BERTScore F1</div>"
                f"<div class='kpi-value'>{agent_bert:.1%}</div>"
                f"<div class='kpi-sub'>Δ vs Naive RAG: <b>{'+' if delta_bert>=0 else ''}{delta_bert:.1%}</b></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with kpi_cols[2]:
            agent_tokens = summary.get("agentic_graphrag", {}).get("mean_tokens", 0)
            rag_tokens = summary.get("naive_rag", {}).get("mean_tokens", 0)
            eff_ratio = metrics.token_efficiency(agent_tokens, rag_tokens) if rag_tokens > 0 else 1.0
            st.markdown(
                f"<div class='white-card'>"
                f"<div class='kpi-title'>Token Efficiency Ratio</div>"
                f"<div class='kpi-value'>{eff_ratio:.2f}x</div>"
                f"<div class='kpi-sub'>Baseline / Agentic Token Spend</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with kpi_cols[3]:
            agent_lat = summary.get("agentic_graphrag", {}).get("mean_latency_seconds", 0.0)
            st.markdown(
                f"<div class='white-card'>"
                f"<div class='kpi-title'>Avg Agent Latency</div>"
                f"<div class='kpi-value'>{agent_lat:.2f}s</div>"
                f"<div class='kpi-sub'>Evaluated on {benchmark_data.get('total_questions', len(df_results))} items</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ── Statistical Visualizations ─────────────────────────────────────────
        st.markdown("#### Statistical Visualizations & Comparisons")
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            # 1. Time (Latency s) vs Token Usage Scatter Plot
            fig_scatter = plot_time_vs_tokens(df_results)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with chart_col2:
            # 2. Accuracy, Completeness & BERTScore Bar Chart
            fig_bars = plot_metric_bars(summary)
            st.plotly_chart(fig_bars, use_container_width=True)

        chart_col3, chart_col4 = st.columns([1, 1])
        with chart_col3:
            # 3. Performance by Question Category
            if "qtype" in df_results.columns:
                metric_to_plot = st.selectbox(
                    "Category Metric Axis",
                    ["accuracy", "bert_f1", "tokens_used", "latency_seconds"],
                    index=0,
                )
                fig_cat = plot_category_breakdown(df_results, metric=metric_to_plot)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Category breakdown available when evaluating on questions with 'qtype' tags.")

        with chart_col4:
            # 4. Token Consumption Comparison
            st.markdown("##### 🪙 Token Consumption by Pipeline")
            token_summary_df = pd.DataFrame([
                {"Pipeline": p, "Mean Tokens": summary[p].get("mean_tokens", 0), "Total Tokens": summary[p].get("total_tokens", 0)}
                for p in summary
            ])
            if not token_summary_df.empty:
                import plotly.express as px
                fig_tokens = px.bar(
                    token_summary_df,
                    x="Pipeline",
                    y="Mean Tokens",
                    color="Pipeline",
                    color_discrete_map={"naive_rag": "#64748b", "graph_rag": "#2563eb", "agentic_graphrag": "#059669"},
                    text_auto=True,
                )
                fig_tokens.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#fcfdfd", font={"family": "Plus Jakarta Sans"})
                st.plotly_chart(fig_tokens, use_container_width=True)

        # ── Tabular Drill-Down ─────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 📋 Detailed Per-Question Evaluation Log")

        filter_pipe = st.multiselect("Filter Pipelines", df_results["pipeline"].unique() if "pipeline" in df_results else [], default=df_results["pipeline"].unique() if "pipeline" in df_results else [])
        filtered_df = df_results[df_results["pipeline"].isin(filter_pipe)] if "pipeline" in df_results and filter_pipe else df_results

        cols_to_show = ["question_id", "pipeline", "qtype", "accuracy", "completeness", "bert_f1", "tokens_used", "latency_seconds", "question", "answer"]
        display_cols = [c for c in cols_to_show if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols], use_container_width=True, height=280)

        # Download Buttons
        dl_c1, dl_c2 = st.columns([1, 1])
        with dl_c1:
            st.download_button(
                "⬇️ Download Benchmark JSON",
                data=json.dumps(benchmark_data, indent=2, default=str),
                file_name="benchmark_results.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_c2:
            st.download_button(
                "⬇️ Download Results CSV",
                data=df_results.to_csv(index=False),
                file_name="benchmark_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: KNOWLEDGE GRAPH & SUBGRAPH EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.markdown("### 🕸 Olympic Knowledge Graph & Schema Explorer")
    st.markdown(
        "Explore the structured graph schema derived from the 2,951 Olympic documents. "
        "Entities are modeled as **Athletes, Events, Games, Venues, Sports, and Countries**, connected by typed relations."
    )

    g_col1, g_col2 = st.columns([1, 3])
    with g_col1:
        st.markdown("##### Filter Entities")
        selected_types = st.multiselect(
            "Entity Types",
            ["Athlete", "Event", "Games", "Venue", "Sport", "Country"],
            default=["Athlete", "Event", "Games", "Venue", "Sport"],
        )
        max_nodes = st.slider("Maximum Graph Nodes", min_value=5, max_value=30, value=12)

    with g_col2:
        # Generate sample Olympic subgraph
        sample_entities = [
            {"id": "Event:Q1050909", "name": "Men's 20km walk 2012", "type": "Event"},
            {"id": "Athlete:Chen Ding", "name": "Chen Ding", "type": "Athlete"},
            {"id": "Games:2012 Summer", "name": "2012 Summer Olympics", "type": "Games"},
            {"id": "Venue:Olympic Stadium", "name": "Olympic Stadium London", "type": "Venue"},
            {"id": "Sport:Athletics", "name": "Athletics", "type": "Sport"},
            {"id": "Country:CHN", "name": "China (CHN)", "type": "Country"},
            {"id": "Event:Q26208457", "name": "Men's pole vault 2012", "type": "Event"},
            {"id": "Athlete:Renaud Lavillenie", "name": "Renaud Lavillenie", "type": "Athlete"},
            {"id": "Country:FRA", "name": "France (FRA)", "type": "Country"},
            {"id": "Event:Q25239316", "name": "Weightlifting 60kg 1988", "type": "Event"},
            {"id": "Athlete:Naim Süleymanoğlu", "name": "Naim Süleymanoğlu", "type": "Athlete"},
            {"id": "Games:1988 Summer", "name": "1988 Summer Olympics", "type": "Games"},
        ]
        sample_relations = [
            {"from": "Athlete:Chen Ding", "relation": "WON_GOLD", "to": "Event:Q1050909"},
            {"from": "Event:Q1050909", "relation": "PART_OF_GAMES", "to": "Games:2012 Summer"},
            {"from": "Event:Q1050909", "relation": "HELD_AT", "to": "Venue:Olympic Stadium"},
            {"from": "Event:Q1050909", "relation": "IN_SPORT", "to": "Sport:Athletics"},
            {"from": "Athlete:Chen Ding", "relation": "REPRESENTS", "to": "Country:CHN"},
            {"from": "Athlete:Renaud Lavillenie", "relation": "WON_GOLD", "to": "Event:Q26208457"},
            {"from": "Event:Q26208457", "relation": "PART_OF_GAMES", "to": "Games:2012 Summer"},
            {"from": "Event:Q26208457", "relation": "IN_SPORT", "to": "Sport:Athletics"},
            {"from": "Athlete:Renaud Lavillenie", "relation": "REPRESENTS", "to": "Country:FRA"},
            {"from": "Athlete:Naim Süleymanoğlu", "relation": "WON_GOLD", "to": "Event:Q25239316"},
            {"from": "Event:Q25239316", "relation": "PART_OF_GAMES", "to": "Games:1988 Summer"},
        ]

        # Filter
        filtered_ents = [e for e in sample_entities if e["type"] in selected_types][:max_nodes]
        valid_ids = set(e["id"] for e in filtered_ents)
        filtered_rels = [r for r in sample_relations if r["from"] in valid_ids and r["to"] in valid_ids]

        full_graph_fig = plot_knowledge_subgraph(filtered_ents, filtered_rels, title="Interactive Olympic Knowledge Graph")
        st.plotly_chart(full_graph_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ARCHITECTURE & METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### 📐 System Architecture & Evaluation Methodology")

    st.markdown("""
    #### 1. Why Three Pipelines?
    To establish whether an agentic control loop genuinely adds value over static retrieval:
    - **Naive RAG** measures flat semantic search over documents without graph structure.
    - **Fixed GraphRAG** adds graph structure (1-hop traversal) in a hardcoded sequence. This isolates the value of the graph alone.
    - **Agentic GraphRAG** implements dynamic, evidence-driven planning where the LLM selects the retrieval and reasoning operations.
    """)

    st.markdown("#### 2. LangGraph Orchestration Topology")
    st.markdown("""
```mermaid
flowchart TD
    Q[User Question] --> INIT[InvestigationState Initialized]
    INIT --> ORCH{orchestrate: LLM next_action}
    ORCH -->|entity_link| EL[Entity Linking Specialist]
    ORCH -->|graph_traverse| GT[Graph Traversal Specialist]
    ORCH -->|vector_search| VS[Vector Search Specialist]
    ORCH -->|document_retrieve| DR[Document Retrieval Specialist]
    ORCH -->|aggregate| AG[Aggregation Specialist]
    ORCH -->|multi_hop_reason| MH[Multi-hop Reasoning Specialist]
    ORCH -->|evaluate_evidence| EV[Evidence Evaluation Specialist]
    ORCH -->|answer / budget reached| FIN[finalize: Final Answer + Inline Citations]

    EL --> STATE[(InvestigationState)]
    GT --> STATE
    VS --> STATE
    DR --> STATE
    AG --> STATE
    MH --> STATE
    EV --> STATE
    STATE --> ORCH
```
    """)

    st.markdown("#### 3. Scoring Formulas")
    st.latex(r"\text{Token Efficiency} = \frac{\text{Baseline Naive RAG Tokens}}{\text{Candidate Pipeline Tokens}}")
    st.latex(r"\text{BERTScore F1} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}")

    st.markdown("#### 4. Round 2 Extension Hooks")
    st.markdown("""
    - **Temporal Attribution**: `EvidenceItem` carries `valid_from`, `valid_until`, and `as_of` metadata.
    - **Conflict Resolution**: When `evidence_evaluation_agent` flags conflicting claims across sources, the orchestrator triggers `resolve_conflict` to audit recency and source authority.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
f_c1, f_c2 = st.columns([3, 1])
with f_c1:
    st.caption("TigerGraph Agentic GraphRAG Hackathon — Round 1 & Round 2 Submission Engine")
with f_c2:
    st.caption("Connected to TigerGraph Cloud Savanna v4.2.5")
