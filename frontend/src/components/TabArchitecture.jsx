import React from 'react';
import { GitBranch, Shield, Zap, Target, BookOpen, Clock, AlertTriangle } from 'lucide-react';

export default function TabArchitecture() {
  return (
    <div>
      <div className="card" style={{ marginBottom: '1.75rem' }}>
        <h3 className="card-title">📐 System Architecture & Evaluation Methodology</h3>
        <p className="card-description">
          Architectural principles, state-machine orchestration diagrams, and rigorous mathematical evaluation formulas for the TigerGraph Agentic GraphRAG Hackathon.
        </p>
      </div>

      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        {/* Pipeline Comparative Rationale */}
        <div className="card">
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.2rem', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <GitBranch size={18} color="var(--tigergraph-orange)" /> 1. Why Three Pipelines?
          </h4>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
            To establish whether an autonomous, agentic control loop genuinely adds value over static retrieval:
          </p>
          <ul style={{ paddingLeft: '1.25rem', fontSize: '0.88rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            <li>
              <b>Naive RAG (Baseline Floor):</b> Measures flat semantic search over documents without knowledge graph structure. Provides the token baseline.
            </li>
            <li>
              <b>Fixed GraphRAG:</b> Adds graph structure (entity linking $\rightarrow$ 1-hop traversal) in a hardcoded sequence. Isolates the value of the knowledge graph alone without dynamic planning.
            </li>
            <li>
              <b>Agentic GraphRAG:</b> Implements dynamic, evidence-driven planning where the LLM selects retrieval and reasoning operations conditioned on intermediate findings.
            </li>
          </ul>
        </div>

        {/* Scoring Formulations */}
        <div className="card">
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.2rem', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Target size={18} color="var(--agentic-emerald)" /> 2. Evaluation Formulations
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontSize: '0.88rem' }}>
            <div style={{ padding: '0.85rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 700, color: 'var(--text-main)', marginBottom: '0.25rem' }}>Token Efficiency Ratio</div>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--agentic-emerald)', fontSize: '0.9rem' }}>
                Token Efficiency = Baseline Naive RAG Tokens / Candidate Pipeline Tokens
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                Values &gt; 1.0 indicate the agent extracted the answer using fewer tokens than flat document chunk dumping.
              </p>
            </div>

            <div style={{ padding: '0.85rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 700, color: 'var(--text-main)', marginBottom: '0.25rem' }}>BERTScore F1 Harmonic Mean</div>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--fixed-blue)', fontSize: '0.9rem' }}>
                BERT F1 = 2 · (Precision_BERT · Recall_BERT) / (Precision_BERT + Recall_BERT)
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                Evaluates semantic alignment against ground truth without being fooled by superficial phrasing variations.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* LangGraph State Machine Topology */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.2rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Zap size={18} color="var(--accent-purple)" /> 3. LangGraph Orchestration Topology
        </h4>

        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.5rem',
          textAlign: 'center',
          overflowX: 'auto',
        }}>
          <svg viewBox="0 0 900 280" style={{ maxWidth: '100%', height: 'auto', margin: '0 auto' }}>
            {/* User Question Node */}
            <rect x="20" y="110" width="120" height="50" rx="8" fill="#ffffff" stroke="#64748b" strokeWidth="2" />
            <text x="80" y="140" textAnchor="middle" fontSize="12" fontWeight="700" fill="#0f172a">User Question</text>

            <line x1="140" y1="135" x2="200" y2="135" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrow)" />

            {/* Central Orchestrator */}
            <rect x="200" y="95" width="160" height="80" rx="10" fill="#f5f3ff" stroke="#7c3aed" strokeWidth="2.5" />
            <text x="280" y="130" textAnchor="middle" fontSize="13" fontWeight="800" fill="#5b21b6">Orchestrator</text>
            <text x="280" y="150" textAnchor="middle" fontSize="10" fill="#7c3aed">(State Inspection)</text>

            {/* Specialist Nodes */}
            {/* 1. Entity Link */}
            <line x1="360" y1="115" x2="480" y2="40" stroke="#059669" strokeWidth="1.8" />
            <rect x="480" y="20" width="160" height="40" rx="6" fill="#ecfdf5" stroke="#059669" strokeWidth="1.5" />
            <text x="560" y="45" textAnchor="middle" fontSize="11" fontWeight="700" fill="#065f46">Entity Linking Specialist</text>

            {/* 2. Graph Traverse */}
            <line x1="360" y1="125" x2="480" y2="90" stroke="#2563eb" strokeWidth="1.8" />
            <rect x="480" y="70" width="160" height="40" rx="6" fill="#eff6ff" stroke="#2563eb" strokeWidth="1.5" />
            <text x="560" y="95" textAnchor="middle" fontSize="11" fontWeight="700" fill="#1e40af">Graph Traversal Specialist</text>

            {/* 3. Vector Search */}
            <line x1="360" y1="140" x2="480" y2="140" stroke="#7c3aed" strokeWidth="1.8" />
            <rect x="480" y="120" width="160" height="40" rx="6" fill="#f5f3ff" stroke="#7c3aed" strokeWidth="1.5" />
            <text x="560" y="145" textAnchor="middle" fontSize="11" fontWeight="700" fill="#5b21b6">Vector Search Specialist</text>

            {/* 4. Multi-hop Reasoning */}
            <line x1="360" y1="155" x2="480" y2="190" stroke="#ea580c" strokeWidth="1.8" />
            <rect x="480" y="170" width="160" height="40" rx="6" fill="#fff7ed" stroke="#ea580c" strokeWidth="1.5" />
            <text x="560" y="195" textAnchor="middle" fontSize="11" fontWeight="700" fill="#9a3412">Multi-hop Reasoner</text>

            {/* 5. Evidence Evaluator */}
            <line x1="360" y1="165" x2="480" y2="240" stroke="#0891b2" strokeWidth="1.8" />
            <rect x="480" y="220" width="160" height="40" rx="6" fill="#ecfeff" stroke="#0891b2" strokeWidth="1.5" />
            <text x="560" y="245" textAnchor="middle" fontSize="11" fontWeight="700" fill="#155e75">Evidence Evaluator</text>

            {/* Final Answer Node */}
            <line x1="360" y1="135" x2="720" y2="135" stroke="#10b981" strokeWidth="2.5" strokeDasharray="4 4" />
            <rect x="720" y="105" width="150" height="60" rx="8" fill="#10b981" stroke="#047857" strokeWidth="2" />
            <text x="795" y="135" textAnchor="middle" fontSize="12" fontWeight="800" fill="#ffffff">Final Answer</text>
            <text x="795" y="152" textAnchor="middle" fontSize="9.5" fill="#ecfdf5">+ Inline Citations</text>
          </svg>
        </div>
      </div>

      {/* Round 2 Extension Hooks */}
      <div className="card">
        <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.2rem', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Shield size={18} color="var(--fixed-blue)" /> 4. Round 2 Extension Hooks (Designed-In)
        </h4>
        <div className="grid-2">
          <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, marginBottom: '0.35rem' }}>
              <Clock size={16} color="var(--tigergraph-orange)" /> Temporal Attribution
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <code>EvidenceItem</code> data structures carry explicit <code>valid_from</code>, <code>valid_until</code>, and <code>as_of</code> attributes to track evolving athletic records and multi-year Olympic timelines.
            </p>
          </div>

          <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, marginBottom: '0.35rem' }}>
              <AlertTriangle size={16} color="#d97706" /> Conflict Resolution Engine
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              When the Evidence Evaluation specialist detects conflicting claims across documents or revisions, the orchestrator triggers an audit step prioritizing primary source documents and authoritative timestamps.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
