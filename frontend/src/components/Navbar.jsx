import React from 'react';
import { Database, Cpu, Beaker, CheckCircle, AlertCircle } from 'lucide-react';

export default function Navbar({ systemStatus, onOpenDiagnostics, activeModel, onSelectModel }) {
  const tg = systemStatus?.tigergraph || {};
  const llm = systemStatus?.llm || {};
  const isTgConnected = tg.connected;

  return (
    <header className="top-navbar">
      <div className="navbar-inner">
        <div className="brand-section">
          <div className="brand-logo-icon">
            🐯
          </div>
          <div>
            <div className="brand-title">
              Agentic GraphRAG
              <span className="brand-badge">TigerGraph Hackathon</span>
            </div>
          </div>
        </div>

        <div className="navbar-metrics">
          {/* TigerGraph Cluster Status */}
          <div className="status-chip" title={tg.host || "TigerGraph Savanna Cluster"}>
            <span className={`status-dot ${isTgConnected ? '' : 'warning'}`} />
            <Database size={14} color="#ea580c" />
            <span>
              <b>Savanna {tg.version || 'v4.2.5'}</b>: {tg.document_count ? tg.document_count.toLocaleString() : '2,951'} Docs · {tg.entity_count ? tg.entity_count.toLocaleString() : '8,576'} Ent
            </span>
          </div>

          {/* Active LLM & Model Switcher */}
          <div className="status-chip" style={{ padding: '0.2rem 0.5rem' }}>
            <Cpu size={14} color="#059669" />
            <select
              value={activeModel || llm.model || "openai/gpt-oss-120b"}
              onChange={(e) => onSelectModel(e.target.value)}
              style={{
                border: 'none',
                background: 'transparent',
                fontWeight: 600,
                fontSize: '0.82rem',
                color: 'var(--text-main)',
                cursor: 'pointer',
                outline: 'none',
              }}
            >
              <option value="openai/gpt-oss-120b">Groq: openai/gpt-oss-120b (120B Flagship)</option>
              <option value="qwen/qwen3.8-27b">Groq: qwen/qwen3.8-27b (Fast 27B)</option>
              <option value="openai/gpt-oss-20b">Groq: openai/gpt-oss-20b (Sub-second)</option>
            </select>
          </div>

          {/* Diagnostics Verification Button */}
          <button className="btn btn-secondary" style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem' }} onClick={onOpenDiagnostics}>
            <Beaker size={14} color="#7c3aed" /> Verify All Systems
          </button>
        </div>
      </div>
    </header>
  );
}
