import React, { useEffect, useRef, useState, useMemo } from 'react';
import cytoscape from 'cytoscape';
import { 
  ZoomIn, ZoomOut, Maximize2, RotateCcw, Download, 
  Search, Share2, Layers, Info, X, ExternalLink, Filter
} from 'lucide-react';

const ENTITY_CONFIG = {
  Athlete:  { color: '#10b981', border: '#059669', bg: '#ecfdf5', shape: 'ellipse', label: 'Athlete' },
  Event:    { color: '#3b82f6', border: '#1d4ed8', bg: '#eff6ff', shape: 'round-rectangle', label: 'Event' },
  Games:    { color: '#f97316', border: '#c2410c', bg: '#fff7ed', shape: 'hexagon', label: 'Games' },
  Venue:    { color: '#8b5cf6', border: '#6d28d9', bg: '#f5f3ff', shape: 'barrel', label: 'Venue' },
  Sport:    { color: '#ef4444', border: '#b91c1c', bg: '#fef2f2', shape: 'diamond', label: 'Sport' },
  Country:  { color: '#f59e0b', border: '#d97706', bg: '#fefce8', shape: 'round-triangle', label: 'Country' },
  Document: { color: '#64748b', border: '#334155', bg: '#f8fafc', shape: 'rectangle', label: 'Document' },
  Entity:   { color: '#10b981', border: '#059669', bg: '#ecfdf5', shape: 'ellipse', label: 'Entity' },
};

const CY_STYLESHEET = [
  // Base Node Style
  {
    selector: 'node',
    style: {
      'label': 'data(label)',
      'font-family': 'Outfit, Inter, system-ui, -apple-system, sans-serif',
      'font-size': '10px',
      'font-weight': 600,
      'color': '#1e293b',
      'text-valign': 'bottom',
      'text-margin-y': 5,
      'text-background-color': '#ffffff',
      'text-background-opacity': 0.88,
      'text-background-padding': '3px',
      'text-background-shape': 'roundrectangle',
      'border-width': 2,
      'border-color': '#94a3b8',
      'background-color': '#e2e8f0',
      'width': 34,
      'height': 34,
      'overlay-opacity': 0,
      'transition-property': 'background-color, border-color, width, height, opacity',
      'transition-duration': '0.2s',
    },
  },
  // Entity Types
  {
    selector: 'node[type = "Athlete"]',
    style: {
      'background-color': '#10b981',
      'border-color': '#059669',
      'shape': 'ellipse',
    },
  },
  {
    selector: 'node[type = "Event"]',
    style: {
      'background-color': '#3b82f6',
      'border-color': '#1d4ed8',
      'shape': 'round-rectangle',
      'width': 38,
      'height': 38,
    },
  },
  {
    selector: 'node[type = "Games"]',
    style: {
      'background-color': '#f97316',
      'border-color': '#c2410c',
      'shape': 'hexagon',
      'width': 42,
      'height': 42,
    },
  },
  {
    selector: 'node[type = "Venue"]',
    style: {
      'background-color': '#8b5cf6',
      'border-color': '#6d28d9',
      'shape': 'barrel',
      'width': 36,
      'height': 36,
    },
  },
  {
    selector: 'node[type = "Sport"]',
    style: {
      'background-color': '#ef4444',
      'border-color': '#b91c1c',
      'shape': 'diamond',
      'width': 38,
      'height': 38,
    },
  },
  {
    selector: 'node[type = "Country"]',
    style: {
      'background-color': '#f59e0b',
      'border-color': '#d97706',
      'shape': 'round-triangle',
      'width': 36,
      'height': 36,
    },
  },
  {
    selector: 'node[type = "Document"]',
    style: {
      'background-color': '#64748b',
      'border-color': '#334155',
      'shape': 'rectangle',
      'width': 40,
      'height': 30,
    },
  },
  // Hover & Active States
  {
    selector: 'node:selected, node.selected',
    style: {
      'border-width': 4,
      'border-color': '#ff5000', // TigerGraph Orange
      'width': 42,
      'height': 42,
      'text-background-color': '#fff7ed',
      'text-background-opacity': 1,
      'color': '#c2410c',
      'font-weight': 700,
    },
  },
  {
    selector: 'node.highlighted',
    style: {
      'border-width': 3,
      'border-color': '#059669',
      'width': 40,
      'height': 40,
    },
  },
  {
    selector: 'node.dimmed',
    style: {
      'opacity': 0.2,
    },
  },
  // Base Edge Style
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#cbd5e1',
      'target-arrow-color': '#94a3b8',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'arrow-scale': 1.1,
      'label': 'data(label)',
      'font-family': 'Outfit, Inter, system-ui, sans-serif',
      'font-size': '8.5px',
      'font-weight': 600,
      'color': '#475569',
      'text-background-color': '#ffffff',
      'text-background-opacity': 0.9,
      'text-background-padding': '2px',
      'text-background-shape': 'roundrectangle',
      'text-rotation': 'autorotate',
      'text-margin-y': -4,
      'transition-property': 'line-color, target-arrow-color, width, opacity',
      'transition-duration': '0.2s',
    },
  },
  // Relation-specific edge styling
  {
    selector: 'edge[label = "WON_GOLD"]',
    style: {
      'line-color': '#eab308',
      'target-arrow-color': '#ca8a04',
      'width': 2.8,
      'color': '#854d0e',
    },
  },
  {
    selector: 'edge[label = "WON_SILVER"]',
    style: {
      'line-color': '#94a3b8',
      'target-arrow-color': '#64748b',
      'width': 2.4,
    },
  },
  {
    selector: 'edge[label = "WON_BRONZE"]',
    style: {
      'line-color': '#d97706',
      'target-arrow-color': '#b45309',
      'width': 2.4,
    },
  },
  {
    selector: 'edge[label = "HELD_AT"]',
    style: {
      'line-color': '#c084fc',
      'target-arrow-color': '#a855f7',
    },
  },
  {
    selector: 'edge[label = "PART_OF_GAMES"]',
    style: {
      'line-color': '#fdba74',
      'target-arrow-color': '#fb923c',
    },
  },
  {
    selector: 'edge[label = "IN_SPORT"]',
    style: {
      'line-color': '#fca5a5',
      'target-arrow-color': '#f87171',
    },
  },
  {
    selector: 'edge.highlighted',
    style: {
      'width': 3.5,
      'line-color': '#059669',
      'target-arrow-color': '#059669',
      'opacity': 1,
      'z-index': 999,
    },
  },
  {
    selector: 'edge.dimmed',
    style: {
      'opacity': 0.12,
    },
  },
];

export default function SubgraphVisualizer({ 
  nodes = [], 
  links = [], 
  title = "Retrieved Knowledge Subgraph (TigerGraph)",
  height = 500
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  const [selectedNode, setSelectedNode] = useState(null);
  const [activeLayout, setActiveLayout] = useState('cose');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('ALL');

  // Convert incoming nodes & links to Cytoscape elements
  const elements = useMemo(() => {
    if (!nodes || nodes.length === 0) return [];

    const nodeElements = nodes.map(n => {
      const cleanLabel = n.name || n.label || String(n.id).replace(/^(Athlete|Event|Games|Venue|Sport|Country):/, '');
      return {
        data: {
          id: String(n.id),
          label: cleanLabel.length > 24 ? cleanLabel.slice(0, 22) + '...' : cleanLabel,
          fullName: cleanLabel,
          type: n.type || 'Entity',
          raw: n,
        },
      };
    });

    const validIds = new Set(nodes.map(n => String(n.id)));
    const edgeElements = (links || [])
      .filter(l => validIds.has(String(l.source || l.from)) && validIds.has(String(l.target || l.to)))
      .map((l, idx) => ({
        data: {
          id: l.id || `edge-${idx}-${l.source || l.from}-${l.target || l.to}`,
          source: String(l.source || l.from),
          target: String(l.target || l.to),
          label: l.label || l.relation || '',
        },
      }));

    return [...nodeElements, ...edgeElements];
  }, [nodes, links]);

  // Compute category counts for the interactive legend
  const categoryCounts = useMemo(() => {
    const counts = {};
    nodes.forEach(n => {
      const t = n.type || 'Entity';
      counts[t] = (counts[t] || 0) + 1;
    });
    return counts;
  }, [nodes]);

  // Initialize or update Cytoscape instance
  useEffect(() => {
    if (!containerRef.current) return;

    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }

    if (elements.length === 0) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements: elements,
      style: CY_STYLESHEET,
      layout: {
        name: activeLayout,
        animate: true,
        animationDuration: 500,
        nodeDimensionsIncludeLabels: true,
        fit: true,
        padding: 40,
        ...(activeLayout === 'cose' ? {
          nodeRepulsion: () => 450000,
          idealEdgeLength: () => 90,
          edgeElasticity: () => 100,
          gravity: 0.25,
          numIter: 800,
        } : {}),
        ...(activeLayout === 'concentric' ? {
          concentric: (node) => node.degree(),
          levelWidth: () => 2,
        } : {}),
      },
      minZoom: 0.3,
      maxZoom: 3.0,
    });

    // Node Click -> Open Inspector Card
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const rawData = node.data();
      const connectedEdges = node.connectedEdges().map(e => ({
        label: e.data('label'),
        targetName: e.target().data('label'),
        sourceName: e.source().data('label'),
        isOutbound: e.source().id() === node.id(),
      }));

      setSelectedNode({
        id: rawData.id,
        name: rawData.fullName || rawData.label,
        type: rawData.type,
        degree: node.degree(),
        edges: connectedEdges,
        attributes: rawData.raw?.attributes || {},
      });
    });

    // Hover Highlight Behavior
    cy.on('mouseover', 'node', (evt) => {
      if (!cy || cy.destroyed()) return;
      const node = evt.target;
      const neighborhood = node.neighborhood().add(node);
      cy.elements().addClass('dimmed');
      neighborhood.removeClass('dimmed').addClass('highlighted');
    });

    cy.on('mouseout', 'node', () => {
      if (!cy || cy.destroyed()) return;
      cy.elements().removeClass('dimmed highlighted');
    });

    // Background Click -> Deselect
    cy.on('tap', (evt) => {
      if (!cy || cy.destroyed()) return;
      if (evt.target === cy) {
        setSelectedNode(null);
        cy.elements().removeClass('selected dimmed highlighted');
      }
    });

    cyRef.current = cy;

    return () => {
      if (cyRef.current) {
        try {
          cyRef.current.removeAllListeners();
          cyRef.current.destroy();
        } catch (e) {
          // ignore cleanup race
        }
        cyRef.current = null;
      }
    };
  }, [elements, activeLayout]);

  // Search Filter Effect
  useEffect(() => {
    if (!cyRef.current) return;
    const cy = cyRef.current;

    if (!searchQuery.trim() && filterType === 'ALL') {
      cy.elements().removeClass('dimmed highlighted selected');
      return;
    }

    const q = searchQuery.toLowerCase().trim();
    cy.batch(() => {
      cy.elements().addClass('dimmed');

      const matches = cy.nodes().filter(node => {
        const matchesQuery = !q || node.data('fullName').toLowerCase().includes(q) || node.data('id').toLowerCase().includes(q);
        const matchesType = filterType === 'ALL' || node.data('type') === filterType;
        return matchesQuery && matchesType;
      });

      matches.removeClass('dimmed').addClass('highlighted');
      matches.connectedEdges().removeClass('dimmed');

      if (matches.length > 0 && q) {
        cy.animate({
          center: { eles: matches },
          zoom: 1.3,
          duration: 400,
        });
      }
    });
  }, [searchQuery, filterType]);

  // Layout Switching
  const handleLayoutChange = (layoutName) => {
    setActiveLayout(layoutName);
    if (cyRef.current) {
      cyRef.current.layout({
        name: layoutName,
        animate: true,
        animationDuration: 500,
        fit: true,
        padding: 40,
        ...(layoutName === 'cose' ? {
          nodeRepulsion: () => 450000,
          idealEdgeLength: () => 90,
          edgeElasticity: () => 100,
          gravity: 0.25,
        } : {}),
      }).run();
    }
  };

  // Zoom / Fit Controls
  const handleZoomIn = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current && cyRef.current.fit(undefined, 35);
  const handleReset = () => {
    if (cyRef.current) {
      cyRef.current.reset();
      cyRef.current.fit(undefined, 35);
      setSelectedNode(null);
      setSearchQuery('');
      setFilterType('ALL');
    }
  };

  // Export PNG Functionality
  const handleExportPNG = () => {
    if (!cyRef.current) return;
    const png64 = cyRef.current.png({ full: true, scale: 2.0, bg: '#ffffff' });
    const link = document.createElement('a');
    link.href = png64;
    link.download = `olympic_graph_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Center on Selected Node
  const handleFocusSelectedNode = () => {
    if (!cyRef.current || !selectedNode) return;
    const target = cyRef.current.getElementById(selectedNode.id);
    if (target.length > 0) {
      cyRef.current.animate({
        center: { eles: target },
        zoom: 1.6,
        duration: 400,
      });
    }
  };

  return (
    <div className="card" style={{ padding: '1rem', position: 'relative' }}>
      {/* Top Header & Interactive Toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.85rem' }}>
        <div>
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.15rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <Share2 size={18} color="var(--tigergraph-orange)" /> {title}
          </h4>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Powered by <b>Cytoscape.js</b> · {nodes.length} Vertices · {links.length} Relations
          </span>
        </div>

        {/* Toolbar: Layouts, Search, Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          {/* Quick Search */}
          <div style={{ position: 'relative', width: '170px' }}>
            <Search size={13} style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Find entity..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                padding: '0.35rem 0.5rem 0.35rem 1.65rem',
                fontSize: '0.78rem',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                background: '#ffffff',
                outline: 'none',
              }}
            />
            {searchQuery && (
              <button 
                onClick={() => setSearchQuery('')}
                style={{ position: 'absolute', right: '6px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
              >
                <X size={12} />
              </button>
            )}
          </div>

          {/* Layout Selector */}
          <div style={{ display: 'flex', background: 'var(--bg-secondary)', padding: '2px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            {[
              { id: 'cose', label: 'Force (Cose)' },
              { id: 'concentric', label: 'Radial' },
              { id: 'breadthfirst', label: 'Tree' },
              { id: 'circle', label: 'Circle' },
            ].map(l => (
              <button
                key={l.id}
                onClick={() => handleLayoutChange(l.id)}
                style={{
                  padding: '0.25rem 0.6rem',
                  fontSize: '0.74rem',
                  fontWeight: activeLayout === l.id ? 700 : 500,
                  background: activeLayout === l.id ? '#ffffff' : 'transparent',
                  color: activeLayout === l.id ? 'var(--tigergraph-orange)' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  boxShadow: activeLayout === l.id ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                  transition: 'all 0.15s ease',
                }}
              >
                {l.label}
              </button>
            ))}
          </div>

          {/* Zoom & Action Buttons */}
          <div style={{ display: 'flex', gap: '0.2rem' }}>
            <button className="btn btn-secondary" style={{ padding: '0.35rem 0.45rem' }} onClick={handleZoomIn} title="Zoom In">
              <ZoomIn size={14} />
            </button>
            <button className="btn btn-secondary" style={{ padding: '0.35rem 0.45rem' }} onClick={handleZoomOut} title="Zoom Out">
              <ZoomOut size={14} />
            </button>
            <button className="btn btn-secondary" style={{ padding: '0.35rem 0.45rem' }} onClick={handleFit} title="Fit to Screen">
              <Maximize2 size={14} />
            </button>
            <button className="btn btn-secondary" style={{ padding: '0.35rem 0.45rem' }} onClick={handleReset} title="Reset View">
              <RotateCcw size={14} />
            </button>
            <button className="btn btn-secondary" style={{ padding: '0.35rem 0.45rem' }} onClick={handleExportPNG} title="Export High-Res PNG">
              <Download size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Cytoscape Canvas Container */}
      <div 
        ref={containerRef}
        className="graph-viewport"
        style={{ 
          height: `${height}px`, 
          width: '100%', 
          position: 'relative',
          background: 'radial-gradient(circle at center, #ffffff 0%, #f8fafc 100%)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
        }}
      >
        {elements.length === 0 && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            No graph entities available for current query selection.
          </div>
        )}
      </div>

      {/* Interactive Floating Node Inspector Card */}
      {selectedNode && (
        <div style={{
          position: 'absolute',
          bottom: '55px',
          right: '18px',
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          border: '1px solid var(--border-medium)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem',
          boxShadow: 'var(--shadow-lg)',
          maxWidth: '310px',
          width: '90%',
          zIndex: 1000,
          animation: 'fadeIn 0.2s ease',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
            <span 
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                fontSize: '0.72rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                padding: '0.15rem 0.5rem',
                borderRadius: 'var(--radius-sm)',
                background: ENTITY_CONFIG[selectedNode.type]?.bg || '#f1f5f9',
                color: ENTITY_CONFIG[selectedNode.type]?.border || '#334155',
                border: `1px solid ${ENTITY_CONFIG[selectedNode.type]?.border || '#cbd5e1'}`,
              }}
            >
              {selectedNode.type}
            </span>
            <button 
              onClick={() => setSelectedNode(null)} 
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
            >
              <X size={15} />
            </button>
          </div>

          <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-main)', marginBottom: '0.3rem' }}>
            {selectedNode.name}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', color: '#64748b', marginBottom: '0.6rem' }}>
            ID: {selectedNode.id}
          </div>

          {/* Connected Edges Breakdown */}
          {selectedNode.edges && selectedNode.edges.length > 0 && (
            <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                Relations ({selectedNode.edges.length}):
              </div>
              <div style={{ maxHeight: '110px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                {selectedNode.edges.map((e, idx) => (
                  <div 
                    key={idx} 
                    style={{ 
                      fontSize: '0.72rem', 
                      background: 'var(--bg-secondary)', 
                      padding: '0.25rem 0.4rem', 
                      borderRadius: 'var(--radius-xs)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                    }}
                  >
                    <b style={{ color: 'var(--tigergraph-orange)' }}>{e.label}</b>
                    <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {e.isOutbound ? e.targetName : e.sourceName}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.4rem' }}>
            <button 
              onClick={handleFocusSelectedNode} 
              className="btn btn-secondary" 
              style={{ flex: 1, padding: '0.3rem', fontSize: '0.74rem', justifyContent: 'center' }}
            >
              Focus View
            </button>
            <button 
              onClick={() => navigator.clipboard.writeText(selectedNode.id)} 
              className="btn btn-secondary" 
              style={{ padding: '0.3rem 0.6rem', fontSize: '0.74rem' }}
              title="Copy Node ID"
            >
              Copy ID
            </button>
          </div>
        </div>
      )}

      {/* Bottom Interactive Legend */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between', 
        flexWrap: 'wrap', 
        gap: '0.5rem', 
        marginTop: '0.75rem', 
        paddingTop: '0.5rem', 
        borderTop: '1px solid var(--border-subtle)',
        fontSize: '0.78rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
          <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Filter Category:</span>
          <button
            onClick={() => setFilterType('ALL')}
            style={{
              background: filterType === 'ALL' ? 'var(--bg-tertiary)' : 'transparent',
              border: `1px solid ${filterType === 'ALL' ? 'var(--border-medium)' : 'transparent'}`,
              borderRadius: 'var(--radius-xs)',
              padding: '0.15rem 0.45rem',
              cursor: 'pointer',
              fontWeight: filterType === 'ALL' ? 700 : 500,
              fontSize: '0.74rem',
            }}
          >
            All ({nodes.length})
          </button>
          {Object.entries(categoryCounts).map(([type, count]) => {
            const cfg = ENTITY_CONFIG[type] || ENTITY_CONFIG.Entity;
            const active = filterType === type;
            return (
              <button
                key={type}
                onClick={() => setFilterType(active ? 'ALL' : type)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  background: active ? cfg.bg : 'transparent',
                  border: `1px solid ${active ? cfg.border : 'transparent'}`,
                  borderRadius: 'var(--radius-xs)',
                  padding: '0.15rem 0.45rem',
                  cursor: 'pointer',
                  fontSize: '0.74rem',
                  fontWeight: active ? 700 : 500,
                }}
              >
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: cfg.color }} />
                <span>{type}</span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>({count})</span>
              </button>
            );
          })}
        </div>

        <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>
          Drag nodes to reposition · Scroll to zoom
        </div>
      </div>
    </div>
  );
}
