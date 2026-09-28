import React, { useState, useEffect } from 'react';
import { Play, Sparkles, Clock, Coins, Layers, Target, CheckSquare, Square, AlertCircle } from 'lucide-react';
import SubgraphVisualizer from './SubgraphVisualizer';
import DecisionTrailDag from './DecisionTrailDag';
import { API_BASE } from '../config';

export default function TabInvestigate({ activeModel }) {
  const [questions, setQuestions] = useState([]);
  const [selectedQuestionIdx, setSelectedQuestionIdx] = useState(0);
  const [customQuestion, setCustomQuestion] = useState("");
  const [goldAnswer, setGoldAnswer] = useState("");
  const [evalBert, setEvalBert] = useState(true);
  const [selectedPipelines, setSelectedPipelines] = useState(["RAG", "GraphRAG", "Agentic GraphRAG"]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Fetch sample questions from backend on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/questions?limit=25`)
      .then(res => res.json())
      .then(data => {
        setQuestions(data);
        if (data.length > 0) {
          setCustomQuestion(data[0].question);
          setGoldAnswer(data[0].gold_answer || "");
        }
      })
      .catch(err => console.error("Error loading sample questions:", err));
  }, []);

  const handleSelectQuestion = (e) => {
    const idx = parseInt(e.target.value, 10);
    setSelectedQuestionIdx(idx);
    if (idx >= 0 && questions[idx]) {
      setCustomQuestion(questions[idx].question);
      setGoldAnswer(questions[idx].gold_answer || "");
    }
  };

  const togglePipeline = (p) => {
    if (selectedPipelines.includes(p)) {
      if (selectedPipelines.length > 1) {
        setSelectedPipelines(selectedPipelines.filter(item => item !== p));
      }
    } else {
      setSelectedPipelines([...selectedPipelines, p]);
    }
  };

  const handleRunInvestigation = async () => {
    if (!customQuestion.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/investigate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: customQuestion,
          pipelines: selectedPipelines,
          model: activeModel,
          eval_bert: evalBert,
          gold_answer: goldAnswer,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Question Selection Card */}
      <div className="card" style={{ marginBottom: '1.75rem' }}>
        <h3 className="card-title">🔍 Single Question Autonomous Deep-Dive</h3>
        <p className="card-description">
          Execute all three retrieval pipelines side-by-side on any Olympic inquiry to inspect factual recall, token economy, and the dynamic LangGraph multi-hop reasoning trail.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.25rem', marginBottom: '1rem' }}>
          <div>
            <label className="input-label">📌 Select from Benchmark Dataset</label>
            <select
              className="input-select"
              value={selectedQuestionIdx}
              onChange={handleSelectQuestion}
            >
              {questions.map((q, idx) => (
                <option key={q.question_id || idx} value={idx}>
                  [{q.qtype ? q.qtype.toUpperCase() : 'GENERAL'}] {q.question.length > 90 ? q.question.slice(0, 87) + '...' : q.question}
                </option>
              ))}
              <option value={-1}>✏️ (Custom Question...)</option>
            </select>
          </div>

          <div>
            <label className="input-label">🎯 Gold Reference Answer</label>
            <input
              type="text"
              className="input-text"
              placeholder="Ground truth answer..."
              value={goldAnswer}
              onChange={e => setGoldAnswer(e.target.value)}
            />
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label className="input-label">✏️ Question to Investigate</label>
          <textarea
            className="input-textarea"
            rows={3}
            value={customQuestion}
            onChange={e => setCustomQuestion(e.target.value)}
            placeholder="Type your Olympic question here..."
          />
        </div>

        {/* Options & Action Row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Target Pipelines:</span>
            {['RAG', 'GraphRAG', 'Agentic GraphRAG'].map(p => {
              const checked = selectedPipelines.includes(p);
              return (
                <label key={p} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', fontSize: '0.88rem' }}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => togglePipeline(p)}
                    style={{ accentColor: 'var(--agentic-emerald)', cursor: 'pointer' }}
                  />
                  <b>{p}</b>
                </label>
              );
            })}

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', fontSize: '0.88rem', marginLeft: '0.5rem', color: 'var(--text-secondary)' }}>
              <input
                type="checkbox"
                checked={evalBert}
                onChange={e => setEvalBert(e.target.checked)}
                style={{ accentColor: 'var(--agentic-emerald)', cursor: 'pointer' }}
              />
              Compute BERTScore F1
            </label>
          </div>

          <button
            className="btn btn-primary"
            disabled={loading || !customQuestion.trim()}
            onClick={handleRunInvestigation}
            style={{ padding: '0.75rem 1.75rem' }}
          >
            {loading ? (
              <>
                <div className="spinner" /> Investigating Question...
              </>
            ) : (
              <>
                <Play size={16} /> Run Comparative Investigation
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 'var(--radius-md)', padding: '1rem', color: '#991b1b', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={18} /> Error: {error}
        </div>
      )}

      {/* Results Section */}
      {results && results.pipelines && (
        <div>
          <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.35rem', marginBottom: '1rem' }}>
            Comparative Pipeline Outputs
          </h3>

          <div className="grid-3" style={{ marginBottom: '2rem' }}>
            {/* Naive RAG Card */}
            {results.pipelines["RAG"] && (
              <div className="pipeline-card rag">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="badge badge-rag">Naive RAG (Baseline)</span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Flat Vector Search</span>
                </div>
                <div className="answer-text">
                  {results.pipelines["RAG"].answer}
                </div>
                <div className="pipeline-metrics-bar">
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Clock size={13} /> {results.pipelines["RAG"].latency_seconds}s
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Coins size={13} /> {results.pipelines["RAG"].tokens_used?.toLocaleString()} tokens
                  </span>
                  {results.pipelines["RAG"].bert_score && (
                    <span style={{ fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Target size={13} color="var(--naive-slate)" /> BERT F1: {results.pipelines["RAG"].bert_score.bert_f1?.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Fixed GraphRAG Card */}
            {results.pipelines["GraphRAG"] && (
              <div className="pipeline-card graph">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="badge badge-graph">Fixed GraphRAG</span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Hardcoded 1-Hop</span>
                </div>
                <div className="answer-text">
                  {results.pipelines["GraphRAG"].answer}
                </div>
                <div className="pipeline-metrics-bar">
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Clock size={13} /> {results.pipelines["GraphRAG"].latency_seconds}s
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Coins size={13} /> {results.pipelines["GraphRAG"].tokens_used?.toLocaleString()} tokens
                  </span>
                  {results.pipelines["GraphRAG"].bert_score && (
                    <span style={{ fontWeight: 700, color: 'var(--fixed-blue)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Target size={13} color="var(--fixed-blue)" /> BERT F1: {results.pipelines["GraphRAG"].bert_score.bert_f1?.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Agentic GraphRAG Card */}
            {results.pipelines["Agentic GraphRAG"] && (
              <div className="pipeline-card agentic">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="badge badge-agent">Agentic GraphRAG (Ours)</span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--agentic-emerald)', fontWeight: 600 }}>LangGraph Orchestrated</span>
                </div>
                <div className="answer-text">
                  {results.pipelines["Agentic GraphRAG"].answer}
                </div>
                <div className="pipeline-metrics-bar">
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Clock size={13} /> {results.pipelines["Agentic GraphRAG"].latency_seconds}s
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Coins size={13} /> {results.pipelines["Agentic GraphRAG"].tokens_used?.toLocaleString()} tokens
                  </span>
                  {results.pipelines["Agentic GraphRAG"].bert_score && (
                    <span style={{ fontWeight: 700, color: 'var(--agentic-emerald)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Target size={13} color="var(--agentic-emerald)" /> BERT F1: {results.pipelines["Agentic GraphRAG"].bert_score.bert_f1?.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Deep-Dive Subgraph & Decision DAG Row */}
          <div className="grid-2">
            <SubgraphVisualizer
              nodes={results.subgraph?.nodes || []}
              links={results.subgraph?.links || []}
              title="Retrieved Knowledge Subgraph (TigerGraph)"
            />
            <DecisionTrailDag steps={results.trace_dag || []} />
          </div>
        </div>
      )}
    </div>
  );
}
