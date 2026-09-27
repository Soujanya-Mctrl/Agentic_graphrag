"""
Visualizer Module for Agentic GraphRAG Dashboard
================================================
Generates Plotly network graphs and statistical visualizations:
  1. Time (Latency s) vs. Token Usage Scatter Plot
  2. Accuracy, Completeness & BERTScore Comparison Charts
  3. Question Category Breakdown (Aggregation, Temporal, Multi-hop, etc.)
  4. Interactive Knowledge Graph / Subgraph Network (NetworkX + Plotly)
  5. Agentic LangGraph Execution Path / Trace Flow DAG

Configured with high-aesthetic styling for a White-Themed Serif + Sans-Serif dashboard.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Color Palettes (White Theme / Editorial) ──────────────────────────────────
PIPELINE_COLORS = {
    "RAG": "#64748b",              # Slate Gray
    "naive_rag": "#64748b",
    "GraphRAG": "#2563eb",         # Royal Sapphire
    "graph_rag": "#2563eb",
    "Agentic GraphRAG": "#059669", # Vivid Emerald
    "agentic_graphrag": "#059669",
}

ENTITY_COLORS = {
    "Athlete": "#10b981",    # Emerald
    "Event": "#ea580c",      # TigerGraph Orange
    "Games": "#0284c7",      # Sky Blue
    "Venue": "#8b5cf6",      # Purple
    "Sport": "#f59e0b",      # Amber
    "Country": "#f43f5e",    # Rose
    "Document": "#64748b",   # Slate
    "Unknown": "#94a3b8",
}

FONT_SERIF = "Newsreader, Playfair Display, Georgia, serif"
FONT_SANS = "Plus Jakarta Sans, Inter, -apple-system, sans-serif"


def _apply_white_theme(fig: go.Figure, title_text: str = "") -> go.Figure:
    """Applies clean white theme with serif titles and sans-serif labels."""
    fig.update_layout(
        title={
            "text": title_text,
            "font": {"family": FONT_SERIF, "size": 19, "color": "#0f172a"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fcfdfd",
        font={"family": FONT_SANS, "color": "#334155", "size": 12},
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
        legend={
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#e2e8f0",
            "borderwidth": 1,
            "font": {"family": FONT_SANS, "size": 12},
        },
    )
    fig.update_xaxes(
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        tickfont={"family": FONT_SANS, "size": 11, "color": "#475569"},
        title_font={"family": FONT_SANS, "size": 12, "color": "#1e293b"},
    )
    fig.update_yaxes(
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        tickfont={"family": FONT_SANS, "size": 11, "color": "#475569"},
        title_font={"family": FONT_SANS, "size": 12, "color": "#1e293b"},
    )
    return fig


# ── 1. Time (Latency) vs. Token Usage Scatter Plot ─────────────────────────────
def plot_time_vs_tokens(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive scatter plot of Latency (seconds) vs Tokens Used,
    colored by pipeline, with bubble size proportional to evidence count or accuracy.
    """
    if df.empty:
        fig = go.Figure()
        return _apply_white_theme(fig, "Time vs. Token Usage (No Data)")

    plot_df = df.copy()
    if "pipeline_name" not in plot_df.columns and "pipeline" in plot_df.columns:
        plot_df["pipeline_name"] = plot_df["pipeline"]

    fig = go.Figure()

    for p_name in plot_df["pipeline_name"].unique():
        sub = plot_df[plot_df["pipeline_name"] == p_name]
        color = PIPELINE_COLORS.get(p_name, "#475569")

        hover_texts = []
        sizes = []
        for _, row in sub.iterrows():
            q_text = row.get("question", "")[:80] + ("..." if len(row.get("question", "")) > 80 else "")
            acc = row.get("accuracy", 0.0)
            bert = row.get("bert_f1", 0.0)
            acc_str = f"{acc:.2f}" if acc is not None else "N/A"
            bert_str = f"{bert:.2f}" if bert is not None else "N/A"
            ev_count = row.get("evidence_count", 0)

            hover_texts.append(
                f"<b>{p_name}</b><br>"
                f"Question: {q_text}<br>"
                f"Latency: {row.get('latency_seconds', 0):.2f}s<br>"
                f"Tokens: {row.get('tokens_used', 0):,}<br>"
                f"Accuracy: {acc_str} | BERT F1: {bert_str}<br>"
                f"Evidence Items: {ev_count}"
            )
            # Size mapping (min 12, max 30)
            sizes.append(max(12, min(30, 12 + ev_count * 2)))

        fig.add_trace(
            go.Scatter(
                x=sub["latency_seconds"],
                y=sub["tokens_used"],
                mode="markers",
                name=p_name,
                marker={
                    "size": sizes,
                    "color": color,
                    "opacity": 0.85,
                    "line": {"width": 1.5, "color": "#ffffff"},
                },
                text=hover_texts,
                hoverinfo="text",
            )
        )

    _apply_white_theme(fig, "Time (Latency s) vs. Total Token Usage")
    fig.update_xaxes(title="Execution Time (Seconds)")
    fig.update_yaxes(title="Tokens Used")
    return fig


# ── 2. Comparative Metric Bars (Accuracy, Completeness, BERTScore) ─────────────
def plot_metric_bars(summary: Dict[str, Any]) -> go.Figure:
    """
    Renders grouped bar chart for Mean Accuracy, Mean Completeness,
    and Mean BERTScore F1 across the pipelines.
    """
    pipelines = list(summary.keys())
    if not pipelines:
        fig = go.Figure()
        return _apply_white_theme(fig, "Benchmark Metrics (No Data)")

    display_names = {
        "naive_rag": "Naive RAG",
        "graph_rag": "Fixed GraphRAG",
        "agentic_graphrag": "Agentic GraphRAG",
        "RAG": "Naive RAG",
        "GraphRAG": "Fixed GraphRAG",
        "Agentic GraphRAG": "Agentic GraphRAG",
    }

    labels = [display_names.get(p, p) for p in pipelines]
    accuracies = [summary[p].get("mean_accuracy", 0.0) for p in pipelines]
    completenesses = [summary[p].get("mean_completeness", 0.0) for p in pipelines]
    bert_f1s = [summary[p].get("mean_bert_f1", 0.0) for p in pipelines]

    fig = go.Figure(
        data=[
            go.Bar(
                name="LLM Accuracy",
                x=labels,
                y=accuracies,
                marker_color="#ea580c",
                text=[f"{v:.1%}" if v > 0 else "-" for v in accuracies],
                textposition="auto",
            ),
            go.Bar(
                name="LLM Completeness",
                x=labels,
                y=completenesses,
                marker_color="#0284c7",
                text=[f"{v:.1%}" if v > 0 else "-" for v in completenesses],
                textposition="auto",
            ),
            go.Bar(
                name="BERTScore F1",
                x=labels,
                y=bert_f1s,
                marker_color="#10b981",
                text=[f"{v:.1%}" if v > 0 else "-" for v in bert_f1s],
                textposition="auto",
            ),
        ]
    )

    _apply_white_theme(fig, "Evaluation Metrics: LLM-as-a-Judge vs. BERTScore")
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
    fig.update_yaxes(title="Score (0.0 – 1.0)", tickformat=".0%")
    return fig


# ── 3. Category-wise Performance Breakdown ─────────────────────────────────────
def plot_category_breakdown(df: pd.DataFrame, metric: str = "accuracy") -> go.Figure:
    """
    Compares pipelines across question categories (aggregation, temporal, multi_hop, etc.).
    """
    if df.empty or "qtype" not in df.columns:
        fig = go.Figure()
        return _apply_white_theme(fig, "Category Performance (No Data)")

    p_col = "pipeline_name" if "pipeline_name" in df.columns else "pipeline"
    agg = df.groupby([p_col, "qtype"])[metric].mean().reset_index()

    metric_title = {
        "accuracy": "Mean LLM Accuracy",
        "completeness": "Mean LLM Completeness",
        "bert_f1": "Mean BERTScore F1",
        "tokens_used": "Mean Tokens Consumed",
        "latency_seconds": "Mean Latency (s)",
    }.get(metric, metric)

    fig = px.bar(
        agg,
        x="qtype",
        y=metric,
        color=p_col,
        barmode="group",
        color_discrete_map=PIPELINE_COLORS,
        labels={"qtype": "Question Category", metric: metric_title, p_col: "Pipeline"},
    )
    _apply_white_theme(fig, f"Performance by Question Category ({metric_title})")
    return fig


# ── 4. Interactive Knowledge Graph / Subgraph Network ──────────────────────────
def plot_knowledge_subgraph(
    entities: List[Dict[str, Any]],
    relations: Optional[List[Dict[str, Any]]] = None,
    title: str = "Retrieved Knowledge Graph Subgraph",
) -> go.Figure:
    """
    Builds an interactive 2D network diagram using NetworkX and Plotly.
    Visualizes entities, types, and typed relationships discovered during investigation.
    """
    G = nx.DiGraph()

    # Add default entities if empty
    if not entities and not relations:
        entities = [
            {"id": "Event:Q1050909", "name": "Men's 20km walk 2012", "type": "Event"},
            {"id": "Athlete:Chen Ding", "name": "Chen Ding", "type": "Athlete"},
            {"id": "Games:2012 Summer", "name": "2012 Summer Olympics", "type": "Games"},
            {"id": "Venue:Olympic Stadium", "name": "Olympic Stadium London", "type": "Venue"},
            {"id": "Sport:Athletics", "name": "Athletics", "type": "Sport"},
        ]
        relations = [
            {"from": "Athlete:Chen Ding", "relation": "WON_GOLD", "to": "Event:Q1050909"},
            {"from": "Event:Q1050909", "relation": "PART_OF_GAMES", "to": "Games:2012 Summer"},
            {"from": "Event:Q1050909", "relation": "HELD_AT", "to": "Venue:Olympic Stadium"},
            {"from": "Event:Q1050909", "relation": "IN_SPORT", "to": "Sport:Athletics"},
        ]

    for e in entities:
        e_id = e.get("id") or e.get("name")
        e_name = e.get("name") or e_id
        e_type = e.get("type") or e.get("entity_type") or "Unknown"
        G.add_node(e_id, label=e_name, type=e_type)

    if relations:
        for r in relations:
            src = r.get("from") or r.get("source")
            dst = r.get("to") or r.get("target")
            rel = r.get("relation") or r.get("rel_type") or "RELATED"
            if src and dst:
                if not G.has_node(src):
                    G.add_node(src, label=src, type="Entity")
                if not G.has_node(dst):
                    G.add_node(dst, label=dst, type="Entity")
                G.add_edge(src, dst, relation=rel)

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        return _apply_white_theme(fig, "Knowledge Graph (Empty)")

    # Compute layout coordinates
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    edge_annotations = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Midpoint annotation for relation label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        rel_label = edge[2].get("relation", "")
        if rel_label:
            edge_annotations.append({
                "x": mid_x,
                "y": mid_y,
                "xref": "x",
                "yref": "y",
                "text": f"<b>{rel_label}</b>",
                "showarrow": False,
                "font": {"size": 9, "family": FONT_SANS, "color": "#64748b"},
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#e2e8f0",
                "borderwidth": 1,
                "borderpad": 2,
            })

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1.5, "color": "#cbd5e1"},
        hoverinfo="none",
        mode="lines",
    )

    # Node traces grouped by entity type
    node_traces = []
    node_types = set(G.nodes[n].get("type", "Unknown") for n in G.nodes())

    for ntype in sorted(node_types):
        nx_list = []
        ny_list = []
        labels = []
        tooltips = []

        for node in G.nodes():
            if G.nodes[node].get("type", "Unknown") == ntype:
                x, y = pos[node]
                nx_list.append(x)
                ny_list.append(y)
                lbl = G.nodes[node].get("label", node)
                deg = G.degree(node)
                labels.append(lbl)
                tooltips.append(
                    f"<b>{lbl}</b><br>Type: {ntype}<br>ID: {node}<br>Degree: {deg} connections"
                )

        color = ENTITY_COLORS.get(ntype, "#64748b")
        node_traces.append(
            go.Scatter(
                x=nx_list,
                y=ny_list,
                mode="markers+text",
                name=ntype,
                text=labels,
                textposition="top center",
                textfont={"family": FONT_SANS, "size": 11, "color": "#0f172a"},
                marker={
                    "size": 22,
                    "color": color,
                    "line": {"width": 2, "color": "#ffffff"},
                    "opacity": 0.95,
                },
                hoverinfo="text",
                hovertext=tooltips,
            )
        )

    fig = go.Figure(data=[edge_trace] + node_traces)
    _apply_white_theme(fig, title)
    fig.update_layout(
        showlegend=True,
        annotations=edge_annotations,
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
    )
    return fig


# ── 5. Agentic LangGraph Execution Path / Trace Flow DAG ───────────────────────
def plot_agentic_trace_dag(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders an interactive DAG showing the orchestrator's decision trail:
    Step 0 (orchestrate) -> Step 1 (specialist) -> Step 2 (orchestrate) -> Finalize
    """
    if not steps:
        fig = go.Figure()
        return _apply_white_theme(fig, "Investigation Path (No Steps Logged)")

    fig = go.Figure()
    n_steps = len(steps)

    x_vals = list(range(n_steps))
    y_vals = [0] * n_steps

    action_colors = {
        "entity_link": "#ea580c",
        "graph_traverse": "#0284c7",
        "vector_search": "#8b5cf6",
        "document_retrieve": "#f59e0b",
        "aggregate": "#10b981",
        "multi_hop_reason": "#059669",
        "evaluate_evidence": "#3b82f6",
        "answer": "#047857",
        "finalize": "#047857",
    }

    # Connector line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line={"width": 3, "color": "#e2e8f0"},
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Nodes
    colors = []
    labels = []
    tooltips = []

    for idx, s in enumerate(steps):
        act = s.get("action") or s.get("source_action") or "step"
        tokens = s.get("tokens_used", 0)
        rationale = s.get("rationale", "")
        inp = s.get("action_input", {})

        colors.append(action_colors.get(act, "#64748b"))
        labels.append(f"Step {idx+1}: {act}")
        tooltips.append(
            f"<b>Step {idx+1}: {act}</b><br>"
            f"Input: {str(inp)[:60]}<br>"
            f"Rationale: {rationale[:90]}<br>"
            f"Tokens: {tokens:,}"
        )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont={"family": FONT_SERIF, "size": 12, "color": "#0f172a"},
            marker={
                "size": 26,
                "color": colors,
                "line": {"width": 2, "color": "#ffffff"},
            },
            hoverinfo="text",
            hovertext=tooltips,
            showlegend=False,
        )
    )

    _apply_white_theme(fig, f"Agentic LangGraph Execution Path ({n_steps} Steps)")
    fig.update_layout(
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "range": [-0.5, 0.8]},
        height=260,
    )
    return fig
