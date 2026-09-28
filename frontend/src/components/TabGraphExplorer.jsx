import React, { useState, useEffect } from 'react';
import { Share2, Filter, Layers, Database } from 'lucide-react';
import SubgraphVisualizer from './SubgraphVisualizer';
import { API_BASE } from '../config';

const ALL_TYPES = ["Athlete", "Event", "Games", "Venue", "Sport", "Country"];

export default function TabGraphExplorer() {
  const [selectedTypes, setSelectedTypes] = useState(["Athlete", "Event", "Games", "Venue", "Sport"]);
  const [maxNodes, setMaxNodes] = useState(16);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/graph/subgraph?types=${selectedTypes.join(',')}&limit=${maxNodes}`)
      .then(res => res.json())
      .then(data => {
        setGraphData(data);
      })
      .catch(err => console.error("Error loading graph:", err))
      .finally(() => setLoading(false));
  }, [selectedTypes, maxNodes]);

  const toggleType = (t) => {
    if (selectedTypes.includes(t)) {
      if (selectedTypes.length > 1) {
        setSelectedTypes(selectedTypes.filter(item => item !== t));
      }
    } else {
      setSelectedTypes([...selectedTypes, t]);
    }
  };

  return (
    <div>
      <div className="card" style={{ marginBottom: '1.75rem' }}>
        <h3 className="card-title">🕸 Olympic Knowledge Graph & Schema Explorer</h3>
        <p className="card-description">
          Explore the typed heterogeneous graph schema derived from 2,951 Olympic documents. Entities represent <b>Athletes, Events, Games, Venues, Sports, and Countries</b> connected by explicit semantic relationships.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem', alignItems: 'center' }}>
          <div>
            <label className="input-label">Filter Entity Categories</label>
            <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
              {ALL_TYPES.map(t => {
                const active = selectedTypes.includes(t);
                return (
                  <button
                    key={t}
                    onClick={() => toggleType(t)}
                    className="btn btn-secondary"
                    style={{
                      padding: '0.4rem 0.9rem',
                      fontSize: '0.82rem',
                      background: active ? 'var(--bg-tertiary)' : '#ffffff',
                      borderColor: active ? 'var(--border-focus)' : 'var(--border-subtle)',
                      fontWeight: active ? 700 : 500,
                    }}
                  >
                    {t}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label className="input-label">Maximum Graph Vertices: <b>{maxNodes} Nodes</b></label>
            <input
              type="range"
              min="6"
              max="30"
              value={maxNodes}
              onChange={e => setMaxNodes(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: 'var(--tigergraph-orange)' }}
            />
          </div>
        </div>
      </div>

      {/* Graph Visualizer Canvas */}
      <div style={{ marginBottom: '2rem' }}>
        <SubgraphVisualizer
          nodes={graphData.nodes || []}
          links={graphData.links || []}
          title="Interactive Olympic Knowledge Graph (TigerGraph Cluster)"
        />
      </div>

      {/* Schema Description & Edge Types */}
      <div className="grid-2">
        <div className="card">
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.15rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Layers size={18} color="var(--tigergraph-orange)" /> Typed Relationships (Edge Schema)
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.88rem' }}>
            <div style={{ padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid #10b981' }}>
              <code>(Athlete) -[WON_GOLD | WON_SILVER | WON_BRONZE]&rarr; (Event)</code>
            </div>
            <div style={{ padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid #3b82f6' }}>
              <code>(Event) -[PART_OF_GAMES]&rarr; (Games)</code>
            </div>
            <div style={{ padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid #8b5cf6' }}>
              <code>(Event) -[HELD_AT]&rarr; (Venue)</code>
            </div>
            <div style={{ padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid #ef4444' }}>
              <code>(Event) -[IN_SPORT]&rarr; (Sport)</code>
            </div>
            <div style={{ padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid #eab308' }}>
              <code>(Athlete) -[REPRESENTS]&rarr; (Country)</code>
            </div>
            <div style={{ padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid #64748b' }}>
              <code>(Document) -[MENTIONS]&rarr; (Entity)</code>
            </div>
          </div>
        </div>

        <div className="card">
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.15rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Database size={18} color="var(--agentic-emerald)" /> TigerGraph Savanna Database Totals
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
            <div className="kpi-card" style={{ padding: '1rem' }}>
              <div className="kpi-title">Document Vertices</div>
              <div className="kpi-value" style={{ fontSize: '1.6rem' }}>2,951</div>
              <div className="kpi-sub">Full Olympic event text</div>
            </div>
            <div className="kpi-card" style={{ padding: '1rem' }}>
              <div className="kpi-title">Entity Vertices</div>
              <div className="kpi-value" style={{ fontSize: '1.6rem' }}>8,576</div>
              <div className="kpi-sub">Athletes, Venues, Games</div>
            </div>
            <div className="kpi-card" style={{ padding: '1rem' }}>
              <div className="kpi-title">MENTIONS Edges</div>
              <div className="kpi-value" style={{ fontSize: '1.6rem' }}>21,962</div>
              <div className="kpi-sub">Document-to-Entity links</div>
            </div>
            <div className="kpi-card" style={{ padding: '1rem' }}>
              <div className="kpi-title">RELATION Edges</div>
              <div className="kpi-value" style={{ fontSize: '1.6rem' }}>13,059</div>
              <div className="kpi-sub">Medal and venue relations</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
