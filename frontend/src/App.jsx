import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import TabInvestigate from './components/TabInvestigate';
import TabBenchmark from './components/TabBenchmark';
import TabGraphExplorer from './components/TabGraphExplorer';
import TabArchitecture from './components/TabArchitecture';
import DiagnosticsModal from './components/DiagnosticsModal';
import { Search, Trophy, Share2, Layers } from 'lucide-react';
import { API_BASE } from './config';

export default function App() {
  const [activeTab, setActiveTab] = useState('investigate');
  const [systemStatus, setSystemStatus] = useState(null);
  const [activeModel, setActiveModel] = useState('openai/gpt-oss-120b');
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState(false);

  // Fetch status on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/status`)
      .then(res => res.json())
      .then(data => {
        setSystemStatus(data);
        if (data.llm?.model) {
          setActiveModel(data.llm.model);
        }
      })
      .catch(err => console.error("Error fetching status:", err));
  }, []);

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <Navbar
        systemStatus={systemStatus}
        activeModel={activeModel}
        onSelectModel={setActiveModel}
        onOpenDiagnostics={() => setIsDiagnosticsOpen(true)}
      />

      <main className="main-content">
        {/* Editorial Hero Header */}
        <div className="hero-header">
          <div style={{ display: 'inline-block', marginBottom: '0.4rem' }}>
            <span className="brand-badge" style={{ background: '#ecfdf5', color: '#059669', borderColor: '#a7f3d0' }}>
              Autonomous Multi-Agent Control Loop
            </span>
          </div>
          <h1 style={{ fontSize: '2.4rem', lineHeight: 1.2 }}>
            Autonomous Graph Investigation & Benchmark Engine
          </h1>
          <p className="hero-subtitle">
            Benchmarking <b>Agentic GraphRAG</b> against <b>Fixed-Sequence GraphRAG</b> and <b>Naive RAG</b> on the 2,951-document Olympic Knowledge Corpus. Auditing when dynamic, multi-step agentic planning justifies additional token investment.
          </p>
        </div>

        {/* Navigation Tabs Bar */}
        <div className="nav-tabs-wrapper">
          <button
            className={`tab-btn ${activeTab === 'investigate' ? 'active' : ''}`}
            onClick={() => setActiveTab('investigate')}
          >
            <Search size={16} /> Single Question Investigation
          </button>

          <button
            className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            <Trophy size={16} /> Benchmarking & Scorecard
          </button>

          <button
            className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
            onClick={() => setActiveTab('graph')}
          >
            <Share2 size={16} /> Knowledge Graph Explorer
          </button>

          <button
            className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
            onClick={() => setActiveTab('architecture')}
          >
            <Layers size={16} /> Architecture & Methodology
          </button>
        </div>

        {/* Tab Content Views */}
        {activeTab === 'investigate' && <TabInvestigate activeModel={activeModel} />}
        {activeTab === 'benchmark' && <TabBenchmark />}
        {activeTab === 'graph' && <TabGraphExplorer />}
        {activeTab === 'architecture' && <TabArchitecture />}
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '1.5rem 2rem',
        background: '#ffffff',
        fontSize: '0.82rem',
        color: 'var(--text-muted)',
      }}>
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <b>TigerGraph Agentic GraphRAG Hackathon 2026</b> · Round 1 & Round 2 Submission Engine
          </div>
          <div>
            Connected to <b>TigerGraph Savanna Cloud v4.2.5</b> · LLM Gateway: <b>Groq ({activeModel})</b>
          </div>
        </div>
      </footer>

      {/* Diagnostics Verification Modal */}
      <DiagnosticsModal
        isOpen={isDiagnosticsOpen}
        onClose={() => setIsDiagnosticsOpen(false)}
      />
    </div>
  );
}
