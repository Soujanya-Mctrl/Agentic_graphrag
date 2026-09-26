"""
Generates results/dashboard.html — the metrics dashboard the submission
checklist requires (tokens, accuracy, completeness across all three
pipelines). Run after runner.py. Self-contained except for one Chart.js
CDN script tag, fine for a local/repo artifact (not a published web page).
"""
from __future__ import annotations

import json
import os

from ..shared.config import CONFIG

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Agentic GraphRAG Benchmark Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; background: #0f1117; color: #e6e6e6; }}
  h1 {{ font-size: 1.4rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }}
  .card {{ background: #1a1d27; border-radius: 10px; padding: 1rem 1.25rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #2a2e3a; }}
  th {{ color: #9aa; font-weight: 600; }}
  canvas {{ max-height: 280px; }}
</style>
</head>
<body>
<h1>RAG vs GraphRAG vs Agentic GraphRAG — Benchmark Results</h1>
<div class="grid">
  <div class="card"><canvas id="accuracyChart"></canvas></div>
  <div class="card"><canvas id="completenessChart"></canvas></div>
  <div class="card"><canvas id="tokensChart"></canvas></div>
</div>
<div class="card" style="margin-top:1.5rem;">
  <h3>Per-question results</h3>
  <table>
    <thead><tr><th>Question</th><th>Pipeline</th><th>Accuracy</th><th>Completeness</th><th>Tokens</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>
<script>
const summary = {summary_json};
const rows = {rows_json};

const pipelines = Object.keys(summary);
const colors = {{ naive_rag: '#6b7280', graph_rag: '#3b82f6', agentic_graphrag: '#22c55e' }};

function mkChart(id, label, key) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{
      labels: pipelines,
      datasets: [{{
        label,
        data: pipelines.map(p => summary[p][key]),
        backgroundColor: pipelines.map(p => colors[p] || '#888'),
      }}]
    }},
    options: {{ plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: label, color: '#e6e6e6' }} }},
                scales: {{ x: {{ ticks: {{ color: '#e6e6e6' }} }}, y: {{ ticks: {{ color: '#e6e6e6' }} }} }} }}
  }});
}}

mkChart('accuracyChart', 'Mean Accuracy', 'mean_accuracy');
mkChart('completenessChart', 'Mean Completeness', 'mean_completeness');
mkChart('tokensChart', 'Mean Tokens Used', 'mean_tokens');

const tbody = document.getElementById('rows');
rows.forEach(r => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{r.question}}</td><td>${{r.pipeline}}</td><td>${{r.accuracy ?? '-'}}</td><td>${{r.completeness ?? '-'}}</td><td>${{r.tokens_used}}</td>`;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>
"""


def generate(results_dir: str | None = None) -> str:
    results_dir = results_dir or CONFIG.results_dir
    with open(os.path.join(results_dir, "benchmark_results.json")) as f:
        data = json.load(f)

    html = TEMPLATE.format(
        summary_json=json.dumps(data["summary"]),
        rows_json=json.dumps(data["per_question_results"]),
    )
    out_path = os.path.join(results_dir, "dashboard.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


if __name__ == "__main__":
    print(generate())
