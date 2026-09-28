import React, { useState } from 'react';
import { Cpu, ArrowDown, ChevronDown, ChevronRight, CheckCircle2, Coins, Clock } from 'lucide-react';

const ACTION_COLORS = {
  entity_link: '#059669',
  graph_traverse: '#2563eb',
  vector_search: '#7c3aed',
  document_retrieve: '#d97706',
  multi_hop_reason: '#ea580c',
  evaluate_evidence: '#0891b2',
  aggregate: '#4f46e5',
  finalize: '#10b981',
};

export default function DecisionTrailDag({ steps = [] }) {
  const [expandedStep, setExpandedStep] = useState(null);

  if (!steps || steps.length === 0) {
    return (
      <div className="card" style={{ padding: '1.5rem', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Execute an investigation with <b>Agentic GraphRAG</b> to view the live dynamic LangGraph decision trail.
        </p>
      </div>
    );
  }

  const toggleExpand = (idx) => {
    setExpandedStep(expandedStep === idx ? null : idx);
  };

  return (
    <div className="card" style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Cpu size={16} color="var(--accent-purple)" /> LangGraph Decision Trail (DAG)
        </h4>
        <span className="badge badge-agent">
          {steps.length} Autonomous Steps
        </span>
      </div>

      <div style={{ position: 'relative' }}>
        {steps.map((step, idx) => {
          const actionColor = ACTION_COLORS[step.action] || '#64748b';
          const isExpanded = expandedStep === idx;
          const isLast = idx === steps.length - 1;

          return (
            <div key={`step-${idx}`} style={{ position: 'relative' }}>
              <div 
                className="dag-step-node"
                onClick={() => toggleExpand(idx)}
                style={{
                  borderLeft: `4px solid ${actionColor}`,
                  cursor: 'pointer',
                  background: isExpanded ? 'var(--bg-secondary)' : '#ffffff',
                }}
              >
                <div className="dag-step-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <span style={{
                      width: '22px',
                      height: '22px',
                      borderRadius: '50%',
                      background: 'var(--bg-tertiary)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                    }}>
                      {step.step_index != null ? step.step_index + 1 : idx + 1}
                    </span>
                    <span className="dag-step-action" style={{ color: actionColor, borderColor: actionColor }}>
                      {step.action}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {step.tokens_used != null && (
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                        <Coins size={12} /> {step.tokens_used}
                      </span>
                    )}
                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </div>
                </div>

                <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {step.rationale || 'Autonomous step executed by the orchestrator.'}
                </div>

                {isExpanded && (
                  <div style={{
                    marginTop: '0.75rem',
                    paddingTop: '0.75rem',
                    borderTop: '1px solid var(--border-subtle)',
                    fontSize: '0.8rem',
                  }}>
                    <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                      Input Parameters:
                    </div>
                    <pre style={{
                      background: '#ffffff',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '0.5rem',
                      fontSize: '0.75rem',
                      fontFamily: 'var(--font-mono)',
                      overflowX: 'auto',
                    }}>
                      {JSON.stringify(step.action_input || {}, null, 2)}
                    </pre>
                  </div>
                )}
              </div>

              {!isLast && (
                <div style={{
                  display: 'flex',
                  justifyContent: 'center',
                  margin: '-0.35rem 0',
                  color: '#94a3b8',
                }}>
                  <ArrowDown size={14} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
