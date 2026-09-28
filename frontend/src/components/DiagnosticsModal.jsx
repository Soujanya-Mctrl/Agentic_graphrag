import React, { useState } from 'react';
import { X, CheckCircle2, AlertTriangle, XCircle, Play, Loader2 } from 'lucide-react';
import { API_BASE } from '../config';

export default function DiagnosticsModal({ isOpen, onClose }) {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const runVerification = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/diagnostics/verify`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to run verification`);
      const data = await res.json();
      setReport(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.35rem', margin: 0 }}>
              🧪 System Verification Agent
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              End-to-End Validation of LLM Gateway, Hugging Face Token & BERTScore
            </p>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>
            <X size={20} />
          </button>
        </div>

        {!report && !loading && (
          <div style={{ textAlign: 'center', padding: '2rem 1rem' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.92rem' }}>
              Execute autonomous system diagnostics to audit environment credentials, LLM Gateway generation speeds, and BERTScore semantic scoring.
            </p>
            <button className="btn btn-primary" onClick={runVerification}>
              <Play size={16} /> Run Diagnostics Agent
            </button>
          </div>
        )}

        {loading && (
          <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
            <div className="spinner" style={{ margin: '0 auto 1rem auto', borderColor: 'rgba(5, 150, 105, 0.3)', borderTopColor: 'var(--agentic-emerald)', width: '32px', height: '32px' }}></div>
            <p style={{ fontWeight: 600, color: 'var(--text-main)' }}>Verification Agent running checks...</p>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Auditing LLM Gateway & BERTScore engine</p>
          </div>
        )}

        {error && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 'var(--radius-md)', padding: '1rem', color: '#991b1b', marginBottom: '1rem', fontSize: '0.88rem' }}>
            ❌ Error running diagnostics: {error}
          </div>
        )}

        {report && (
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.85rem 1.25rem',
              borderRadius: 'var(--radius-md)',
              background: report.overall_status === 'HEALTHY' ? 'var(--agentic-emerald-bg)' : '#fffbeb',
              border: `1px solid ${report.overall_status === 'HEALTHY' ? 'var(--agentic-emerald-border)' : '#fde68a'}`,
              marginBottom: '1.25rem',
            }}>
              <div>
                <span style={{
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  color: report.overall_status === 'HEALTHY' ? 'var(--agentic-emerald)' : '#b45309',
                }}>
                  Status: {report.overall_status}
                </span>
                <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-main)', marginTop: '0.2rem' }}>
                  {report.summary}
                </div>
              </div>
            </div>

            <h4 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.6rem' }}>
              Check Results
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginBottom: '1.5rem' }}>
              {report.checks?.map((chk, idx) => (
                <div key={idx} style={{
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.75rem 1rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '0.75rem',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    {chk.status === 'PASSED' ? (
                      <CheckCircle2 size={16} color="var(--agentic-emerald)" />
                    ) : (
                      <AlertTriangle size={16} color="#f59e0b" />
                    )}
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.88rem' }}>{chk.name}</div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{chk.message}</div>
                    </div>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                    {chk.duration_ms?.toFixed(1)}ms
                  </span>
                </div>
              ))}
            </div>

            {report.trial_result && (
              <div style={{
                background: '#ffffff',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '1rem',
                fontSize: '0.85rem',
              }}>
                <div style={{ fontWeight: 700, marginBottom: '0.4rem', color: 'var(--text-main)' }}>
                  🎯 End-to-End Trial Result:
                </div>
                <div style={{ color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                  BERTScore F1: <b style={{ color: 'var(--agentic-emerald)' }}>{report.trial_result.bert_f1?.toFixed(4)}</b> (Precision: {report.trial_result.bert_precision?.toFixed(4)}, Recall: {report.trial_result.bert_recall?.toFixed(4)})
                </div>
                <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
                  Latency: {report.trial_result.latency_seconds?.toFixed(2)}s · Tokens: {report.trial_result.tokens_used}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.5rem', gap: '0.75rem' }}>
              <button className="btn btn-secondary" onClick={runVerification}>
                Re-Run
              </button>
              <button className="btn btn-primary" onClick={onClose}>
                Done
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
