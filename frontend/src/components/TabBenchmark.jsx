import React, { useState, useEffect } from 'react';
import { Play, FolderOpen, Download, BarChart2, TrendingUp, Zap, Clock, ShieldCheck, CheckCircle } from 'lucide-react';
import { API_BASE } from '../config';

export default function TabBenchmark() {
  const [dataset, setDataset] = useState("data/questions/eval_public.jsonl");
  const [sampleSize, setSampleSize] = useState(5);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 5, latest_item: "" });
  const [filterPipeline, setFilterPipeline] = useState("all");
  const [error, setError] = useState(null);

  // Automatically try loading existing saved benchmark on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/benchmark/results`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) setBenchmarkData(data);
      })
      .catch(() => {});
  }, []);

  // Poll progress when running
  useEffect(() => {
    let interval = null;
    if (isRunning) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/benchmark/status`);
          const status = await res.json();
          setProgress(status);

          if (status.error) {
            setError(status.error);
            setIsRunning(false);
          } else if (!status.is_running) {
            setIsRunning(false);
            if (status.current > 0) {
              const rRes = await fetch(`${API_BASE}/api/benchmark/results`);
              if (rRes.ok) {
                const rData = await rRes.json();
                setBenchmarkData(rData);
              }
            }
          }
        } catch (e) {
          console.error("Progress poll error:", e);
        }
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const handleStartBenchmark = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_size: sampleSize,
          dataset: dataset,
          pipelines: ["RAG", "GraphRAG", "Agentic GraphRAG"],
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to start benchmark`);
    } catch (err) {
      setError(err.message);
      setIsRunning(false);
    }
  };

  const handleLoadSaved = async () => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark/results`);
      if (!res.ok) throw new Error("No saved benchmark results found. Run a benchmark first.");
      const data = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      setError(err.message);
    }
  };

  const summary = benchmarkData?.summary || {};
  const perQuestion = benchmarkData?.per_question_results || [];

  // KPIs
  const agenticAcc = summary.agentic_graphrag?.mean_accuracy || 0;
  const naiveAcc = summary.naive_rag?.mean_accuracy || 0;
  const deltaAcc = agenticAcc - naiveAcc;

  const agenticBert = summary.agentic_graphrag?.mean_bert_f1 || 0;
  const naiveBert = summary.naive_rag?.mean_bert_f1 || 0;
  const deltaBert = agenticBert - naiveBert;

  const agenticTokens = summary.agentic_graphrag?.mean_tokens || 0;
  const naiveTokens = summary.naive_rag?.mean_tokens || 0;
  const tokenEfficiency = agenticTokens > 0 ? (naiveTokens / agenticTokens).toFixed(2) : "1.00";

  const agenticLatency = summary.agentic_graphrag?.mean_latency_seconds || 0;

  // Filtered rows for table
  const filteredRows = filterPipeline === "all"
    ? perQuestion
    : perQuestion.filter(r => r.pipeline === filterPipeline);

  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(benchmarkData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'benchmark_results.json';
    a.click();
  };

  return (
    <div>
      {/* Benchmark Controls Card */}
      <div className="card" style={{ marginBottom: '1.75rem' }}>
        <h3 className="card-title">🏆 Comparative Benchmarking & Evaluation Engine</h3>
        <p className="card-description">
          Execute automated batch evaluations across public and hidden test splits. Each pipeline is scored using <b>LLM-as-a-Judge</b> (Accuracy & Completeness) paired with <b>BERTScore F1</b>.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr auto auto', gap: '1rem', alignItems: 'end' }}>
          <div>
            <label className="input-label">Select Evaluation Dataset</label>
            <select
              className="input-select"
              value={dataset}
              onChange={e => setDataset(e.target.value)}
              disabled={isRunning}
            >
              <option value="data/questions/eval_public.jsonl">data/questions/eval_public.jsonl (100 Questions)</option>
              <option value="data/questions/eval_hidden.jsonl">data/questions/eval_hidden.jsonl (50 Questions)</option>
              <option value="data/sample_questions.json">data/sample_questions.json (Toy Mock)</option>
            </select>
          </div>

          <div>
            <label className="input-label">Sample Size: <b>{sampleSize} Questions</b></label>
            <input
              type="range"
              min="1"
              max="50"
              value={sampleSize}
              onChange={e => setSampleSize(parseInt(e.target.value, 10))}
              disabled={isRunning}
              style={{ width: '100%', accentColor: 'var(--agentic-emerald)' }}
            />
          </div>

          <button
            className="btn btn-primary"
            onClick={handleStartBenchmark}
            disabled={isRunning}
            style={{ padding: '0.7rem 1.4rem' }}
          >
            {isRunning ? <div className="spinner" /> : <Play size={16} />}
            {isRunning ? "Running Benchmark..." : "Start Benchmark"}
          </button>

          <button
            className="btn btn-secondary"
            onClick={handleLoadSaved}
            disabled={isRunning}
            style={{ padding: '0.7rem 1.2rem' }}
          >
            <FolderOpen size={16} /> Load Saved Results
          </button>
        </div>

        {error && (
          <div style={{ marginTop: '1.2rem', padding: '0.8rem 1.2rem', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 'var(--radius-md)', color: '#991b1b', fontSize: '0.88rem' }}>
            ⚠️ <b>Benchmark Alert:</b> {error}
          </div>
        )}

        {/* Live Progress Bar */}
        {isRunning && (
          <div style={{ marginTop: '1.5rem', background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              <span>Evaluating Question {progress.current} of {progress.total}...</span>
              <span className="mono">{Math.round((progress.current / Math.max(progress.total, 1)) * 100)}%</span>
            </div>
            <div style={{ background: '#e2e8f0', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                background: 'linear-gradient(90deg, #059669, #10b981)',
                width: `${(progress.current / Math.max(progress.total, 1)) * 100}%`,
                height: '100%',
                transition: 'width 0.3s ease',
              }} />
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
              Current question: <i>"{progress.latest_item}"</i>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 'var(--radius-md)', padding: '1rem', color: '#991b1b', marginBottom: '1.5rem' }}>
          {error}
        </div>
      )}

      {/* Benchmark Scorecard KPIs */}
      {benchmarkData && (
        <div>
          <div className="grid-4" style={{ marginBottom: '2rem' }}>
            <div className="kpi-card">
              <div className="kpi-title">Agentic Accuracy (LLM)</div>
              <div className="kpi-value">{(agenticAcc * 100).toFixed(1)}%</div>
              <div className="kpi-sub">
                Δ vs Naive RAG: <span className="kpi-delta-positive">{deltaAcc >= 0 ? '+' : ''}{(deltaAcc * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-title">Agentic BERTScore F1</div>
              <div className="kpi-value">{(agenticBert * 100).toFixed(1)}%</div>
              <div className="kpi-sub">
                Δ vs Naive RAG: <span className="kpi-delta-positive">{deltaBert >= 0 ? '+' : ''}{(deltaBert * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-title">Token Efficiency Ratio</div>
              <div className="kpi-value">{tokenEfficiency}x</div>
              <div className="kpi-sub">
                Baseline / Agentic Spend
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-title">Avg Agent Latency</div>
              <div className="kpi-value">{agenticLatency.toFixed(2)}s</div>
              <div className="kpi-sub">
                Across {benchmarkData.total_questions || perQuestion.length} questions
              </div>
            </div>
          </div>

          {/* Statistical Comparisons */}
          <div className="grid-2" style={{ marginBottom: '2rem' }}>
            {/* Metric Comparison Bar Chart */}
            <div className="card">
              <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.15rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <BarChart2 size={18} color="var(--tigergraph-orange)" /> Accuracy, Completeness & BERT F1
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1rem' }}>
                {[
                  { name: "Naive RAG", key: "naive_rag", color: "var(--naive-slate)" },
                  { name: "Fixed GraphRAG", key: "graph_rag", color: "var(--fixed-blue)" },
                  { name: "Agentic GraphRAG (Ours)", key: "agentic_graphrag", color: "var(--agentic-emerald)" },
                ].map(p => {
                  const acc = summary[p.key]?.mean_accuracy || 0;
                  const bert = summary[p.key]?.mean_bert_f1 || 0;
                  const comp = summary[p.key]?.mean_completeness || 0;
                  return (
                    <div key={p.key}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                        <span>{p.name}</span>
                        <span className="mono">Acc: {(acc * 100).toFixed(1)}% · BERT: {(bert * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ display: 'flex', height: '18px', background: '#f1f5f9', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                        <div style={{ width: `${acc * 100}%`, background: p.color, height: '100%' }} title={`Accuracy: ${(acc * 100).toFixed(1)}%`} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Token Consumption Comparison */}
            <div className="card">
              <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.15rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Zap size={18} color="var(--agentic-emerald)" /> Average Token Spend per Inquiry
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1rem' }}>
                {[
                  { name: "Naive RAG", key: "naive_rag", color: "var(--naive-slate)" },
                  { name: "Fixed GraphRAG", key: "graph_rag", color: "var(--fixed-blue)" },
                  { name: "Agentic GraphRAG (Ours)", key: "agentic_graphrag", color: "var(--agentic-emerald)" },
                ].map(p => {
                  const tokens = summary[p.key]?.mean_tokens || 0;
                  const maxTokens = Math.max(
                    summary.naive_rag?.mean_tokens || 1,
                    summary.graph_rag?.mean_tokens || 1,
                    summary.agentic_graphrag?.mean_tokens || 1
                  );
                  const pct = Math.min(100, Math.round((tokens / maxTokens) * 100));

                  return (
                    <div key={p.key}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                        <span>{p.name}</span>
                        <span className="mono">{tokens.toLocaleString()} tokens</span>
                      </div>
                      <div style={{ display: 'flex', height: '18px', background: '#f1f5f9', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                        <div style={{ width: `${pct}%`, background: p.color, height: '100%' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Per-Question Detailed Evaluation Table */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div>
                <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.15rem', margin: 0 }}>
                  📋 Per-Question Evaluation Detail Log
                </h4>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Showing {filteredRows.length} recorded runs
                </span>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <select
                  className="input-select"
                  style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
                  value={filterPipeline}
                  onChange={e => setFilterPipeline(e.target.value)}
                >
                  <option value="all">All Pipelines</option>
                  <option value="naive_rag">Naive RAG</option>
                  <option value="graph_rag">Fixed GraphRAG</option>
                  <option value="agentic_graphrag">Agentic GraphRAG</option>
                </select>

                <button className="btn btn-secondary" style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem' }} onClick={downloadJSON}>
                  <Download size={14} /> Export JSON
                </button>
              </div>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border-subtle)', textAlign: 'left', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.75rem 0.5rem' }}>ID</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Pipeline</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Type</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Accuracy</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>BERT F1</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Tokens</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Latency</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Question</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.slice(0, 30).map((row, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '0.65rem 0.5rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{row.question_id}</td>
                      <td style={{ padding: '0.65rem 0.5rem' }}>
                        <span className={`badge ${row.pipeline === 'agentic_graphrag' ? 'badge-agent' : row.pipeline === 'graph_rag' ? 'badge-graph' : 'badge-rag'}`}>
                          {row.pipeline}
                        </span>
                      </td>
                      <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-muted)' }}>{row.qtype || 'general'}</td>
                      <td style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>{row.accuracy != null ? (row.accuracy * 100).toFixed(0) + '%' : '-'}</td>
                      <td style={{ padding: '0.65rem 0.5rem', fontWeight: 700, color: 'var(--agentic-emerald)' }}>
                        {row.bert_f1 != null ? row.bert_f1.toFixed(2) : '-'}
                      </td>
                      <td style={{ padding: '0.65rem 0.5rem', fontFamily: 'var(--font-mono)' }}>{row.tokens_used}</td>
                      <td style={{ padding: '0.65rem 0.5rem' }}>{row.latency_seconds?.toFixed(2)}s</td>
                      <td style={{ padding: '0.65rem 0.5rem', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {row.question}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
