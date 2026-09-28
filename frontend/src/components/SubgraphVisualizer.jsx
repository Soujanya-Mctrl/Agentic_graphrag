import React, { useState, useEffect, useRef } from 'react';
import { Eye, ZoomIn, ZoomOut, RotateCcw, Share2, Layers } from 'lucide-react';

const ENTITY_COLORS = {
  Athlete: { bg: '#ecfdf5', border: '#10b981', text: '#065f46', fill: '#059669' },
  Event: { bg: '#eff6ff', border: '#3b82f6', text: '#1e40af', fill: '#2563eb' },
  Games: { bg: '#fff7ed', border: '#f97316', text: '#9a3412', fill: '#ea580c' },
  Venue: { bg: '#f5f3ff', border: '#8b5cf6', text: '#5b21b6', fill: '#7c3aed' },
  Sport: { bg: '#fef2f2', border: '#ef4444', text: '#991b1b', fill: '#dc2626' },
  Country: { bg: '#fefce8', border: '#eab308', text: '#854d0e', fill: '#ca8a04' },
  Document: { bg: '#f8fafc', border: '#64748b', text: '#334155', fill: '#475569' },
  Entity: { bg: '#ecfdf5', border: '#10b981', text: '#065f46', fill: '#059669' },
};

export default function SubgraphVisualizer({ nodes = [], links = [], title = "Retrieved Knowledge Subgraph" }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Calculate layout coordinates in a circular/bipartite configuration
  const width = 800;
  const height = 480;
  const centerX = width / 2;
  const centerY = height / 2;

  const nodePositions = React.useMemo(() => {
    const posMap = {};
    if (!nodes || nodes.length === 0) return posMap;

    const n = nodes.length;
    const radius = Math.min(width, height) * 0.35;

    nodes.forEach((node, idx) => {
      // Position center node if it's the main subject
      if (idx === 0) {
        posMap[node.id] = { x: centerX, y: centerY };
      } else {
        const angle = ((idx - 1) / (n - 1)) * 2 * Math.PI - Math.PI / 2;
        posMap[node.id] = {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        };
      }
    });
    return posMap;
  }, [nodes]);

  const handleMouseDown = (e) => {
    if (e.target.tagName === 'svg' || e.target.tagName === 'rect') {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="card" style={{ padding: '1rem', position: 'relative' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <div>
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Share2 size={16} color="var(--agentic-emerald)" /> {title}
          </h4>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            {nodes.length} Vertices · {links.length} Relations
          </span>
        </div>

        {/* Zoom & Control Bar */}
        <div style={{ display: 'flex', gap: '0.25rem' }}>
          <button className="btn btn-secondary" style={{ padding: '0.35rem' }} onClick={() => setZoom(z => Math.min(2, z + 0.15))} title="Zoom In">
            <ZoomIn size={14} />
          </button>
          <button className="btn btn-secondary" style={{ padding: '0.35rem' }} onClick={() => setZoom(z => Math.max(0.5, z - 0.15))} title="Zoom Out">
            <ZoomOut size={14} />
          </button>
          <button className="btn btn-secondary" style={{ padding: '0.35rem' }} onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} title="Reset View">
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      <div 
        className="graph-viewport"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`}>
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
            </marker>
          </defs>

          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
            {/* Background Grid */}
            <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1" fill="#e2e8f0" />
            </pattern>
            <rect x="-1000" y="-1000" width="3000" height="3000" fill="url(#grid)" />

            {/* Edge Lines */}
            {links.map((link, idx) => {
              const srcPos = nodePositions[link.source];
              const dstPos = nodePositions[link.target];
              if (!srcPos || !dstPos) return null;

              const midX = (srcPos.x + dstPos.x) / 2;
              const midY = (srcPos.y + dstPos.y) / 2;

              return (
                <g key={`edge-${idx}`}>
                  <line
                    x1={srcPos.x}
                    y1={srcPos.y}
                    x2={dstPos.x}
                    y2={dstPos.y}
                    stroke="#cbd5e1"
                    strokeWidth="1.8"
                    strokeDasharray={link.label === 'MENTIONS' ? '4 3' : 'none'}
                    markerEnd="url(#arrow)"
                  />
                  {/* Edge Label Badge */}
                  <rect
                    x={midX - (link.label.length * 3.5 + 8)}
                    y={midY - 9}
                    width={link.label.length * 7 + 16}
                    height="18"
                    rx="4"
                    fill="#ffffff"
                    stroke="#e2e8f0"
                    strokeWidth="1"
                  />
                  <text
                    x={midX}
                    y={midY + 3}
                    textAnchor="middle"
                    fontSize="9"
                    fontWeight="700"
                    fill="#475569"
                    fontFamily="var(--font-mono)"
                  >
                    {link.label}
                  </text>
                </g>
              );
            })}

            {/* Node Circles */}
            {nodes.map((node) => {
              const pos = nodePositions[node.id] || { x: centerX, y: centerY };
              const colorTheme = ENTITY_COLORS[node.type] || ENTITY_COLORS.Entity;
              const isSelected = selectedNode?.id === node.id;

              return (
                <g
                  key={node.id}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  onClick={() => setSelectedNode(node)}
                  style={{ cursor: 'pointer' }}
                >
                  <circle
                    r={isSelected ? "22" : "18"}
                    fill={colorTheme.bg}
                    stroke={isSelected ? "#0f172a" : colorTheme.border}
                    strokeWidth={isSelected ? "3" : "2"}
                    style={{ transition: 'all 0.2s ease' }}
                  />
                  <circle r="6" fill={colorTheme.fill} />
                  <text
                    y="32"
                    textAnchor="middle"
                    fontSize="11"
                    fontWeight="600"
                    fill="var(--text-main)"
                    fontFamily="var(--font-sans)"
                    style={{ pointerEvents: 'none' }}
                  >
                    {node.name.length > 20 ? node.name.slice(0, 18) + '...' : node.name}
                  </text>
                  <text
                    y="44"
                    textAnchor="middle"
                    fontSize="8.5"
                    fontWeight="600"
                    fill={colorTheme.text}
                    fontFamily="var(--font-mono)"
                    style={{ pointerEvents: 'none' }}
                  >
                    {node.type}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Selected Node Details Overlay */}
        {selectedNode && (
          <div style={{
            position: 'absolute',
            bottom: '12px',
            right: '12px',
            background: '#ffffff',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: '0.85rem 1rem',
            boxShadow: 'var(--shadow-md)',
            maxWidth: '260px',
            fontSize: '0.82rem',
          }}>
            <div style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '0.2rem' }}>{selectedNode.name}</div>
            <div style={{ color: 'var(--text-muted)', marginBottom: '0.4rem' }}>Type: <b>{selectedNode.type}</b></div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: '#64748b' }}>ID: {selectedNode.id}</div>
            <button
              onClick={() => setSelectedNode(null)}
              style={{ marginTop: '0.5rem', background: 'none', border: 'none', color: 'var(--tigergraph-orange)', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600 }}
            >
              Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
