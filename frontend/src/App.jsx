import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [logs, setLogs] = useState([]);
  const [papers, setPapers] = useState([]);
  const [stats, setStats] = useState({ total_papers: 0, total_chunks: 0 });
  const [isProcessing, setIsProcessing] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [activeContext, setActiveContext] = useState(null); // Selected paper for follow-ups
  const [clarificationData, setClarificationData] = useState(null); // When system asks "Which paper?"

  const scrollRef = useRef(null);
  const logScrollRef = useRef(null);
  const ws = useRef(null);

  useEffect(() => {
    fetchStats();
    fetchPapers();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (logScrollRef.current) {
      logScrollRef.current.scrollTop = logScrollRef.current.scrollHeight;
    }
  }, [logs]);

  const fetchStats = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/stats`);
      if (resp.ok) {
        const data = await resp.json();
        setStats(data);
        setApiStatus("online");
      } else {
        setApiStatus("offline");
      }
    } catch (err) { 
      console.error("Stats Error:", err); 
      setApiStatus("offline");
    }
  };

  const fetchPapers = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/papers`);
      const data = await resp.json();
      setPapers(data);
    } catch (err) { console.error("Papers Error:", err); }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (resp.ok) {
        fetchStats();
        fetchPapers();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const startQuery = (overrideQuery = null, overridePaperId = null) => {
    const q = overrideQuery || query;
    const pId = overridePaperId || activeContext?.paper_id;

    if (!q || isProcessing) return;

    setIsProcessing(true);
    setClarificationData(null);
    setLogs([]);
    
    if (!overrideQuery) {
      setMessages(prev => [...prev, { role: 'user', content: q }]);
    }
    
    ws.current = new WebSocket(`${WS_BASE}/ws/reasoning`);
    
    ws.current.onopen = () => {
      ws.current.send(JSON.stringify({ 
        query: q,
        active_paper_id: pId 
      }));
    };

    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.error) {
        setIsProcessing(false);
        setLogs(prev => [...prev, { step: 'error', status: 'error', data: msg.error }]);
        return;
      }

      if (msg.step === "clarification_needed") {
        setIsProcessing(false);
        setClarificationData({ papers: msg.papers, query: msg.query });
        return;
      }

      if (msg.step === "final_answer") {
        setMessages(prev => [...prev, { 
          role: 'ai', 
          content: msg.answer, 
          confidence: msg.confidence,
          chunks: msg.retrieved_chunks,
          isFinal: true
        }]);
        setIsProcessing(false);
        setQuery("");
        if (msg.active_paper_id) {
          const paper = papers.find(p => p.paper_id === msg.active_paper_id);
          if (paper) setActiveContext(paper);
        }
        fetchStats();
      } else {
        setLogs(prev => {
          const filtered = prev.filter(l => l.step !== msg.step);
          return [...filtered, msg];
        });
      }
    };

    ws.current.onerror = (err) => {
      console.error("WS Error:", err);
      setIsProcessing(false);
    };
  };

  const handlePaperSelect = (paper) => {
    setActiveContext(paper);
    setClarificationData(null);
    // Automatically re-run the query with this paper context
    startQuery(clarificationData.query, paper.paper_id);
  };

  return (
    <div className="app-container">
      {/* Left Panel: Knowledge Base */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <span className="mono-label">Resembler / Intelligence</span>
            <h2>Repository</h2>
          </div>
          <div className="status-indicator">
            <div className={`pulse ${apiStatus === 'online' ? '' : 'offline'}`} />
            <span style={{ fontSize: '0.65rem' }}>API: {apiStatus}</span>
          </div>
        </div>
        <div className="panel-content">
          <div className="stats-grid">
            <div className="stats-card">
              <span className="mono-label">Papers</span>
              <h3 style={{ fontSize: '1.5rem', color: 'var(--accent)' }}>{stats.total_papers}</h3>
            </div>
            <div className="stats-card">
              <span className="mono-label">Chunks</span>
              <h3 style={{ fontSize: '1.5rem', color: 'var(--accent)' }}>{stats.total_chunks}</h3>
            </div>
          </div>

          <label className="soft-button" style={{ width: '100%', marginBottom: '1.5rem' }}>
            <input type="file" style={{ display: 'none' }} onChange={handleUpload} accept=".pdf" />
            Upload Research PDF
          </label>

          <span className="mono-label">Indexed Documents</span>
          {papers.map((p, i) => (
            <div 
              key={i} 
              className={`paper-card staggered-item ${activeContext?.paper_id === p.paper_id ? 'active-ctx' : ''}`} 
              onClick={() => setActiveContext(p)}
            >
              <h4 style={{ fontSize: '0.9rem', marginBottom: '0.25rem' }}>{p.title || p.paper_id}</h4>
              <span className="mono-label" style={{ fontSize: '0.6rem' }}>{p.total_pages} Pages | {p.total_chunks || '?'} Chunks</span>
            </div>
          ))}
        </div>
      </div>

      {/* Center Panel: Reasoning Stream */}
      <div className="panel">
        <div className="panel-header" style={{ paddingBottom: '0.5rem' }}>
          <div>
            <span className="mono-label">Stream / Reasoning</span>
            <h2>Agent Pipeline</h2>
          </div>
          <div className="status-indicator">
            <div className={`pulse ${isProcessing ? '' : 'inactive'}`} />
            <span style={{ fontSize: '0.65rem' }}>System: {isProcessing ? 'Thinking' : 'Idle'}</span>
          </div>
        </div>
        
        {activeContext && (
          <div className="active-context-bar staggered-item">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="mono-label" style={{ marginBottom: 0 }}>Focusing on:</span>
              <span className="context-chip">
                {activeContext.title || activeContext.paper_id}
                <button className="clear-ctx" onClick={() => setActiveContext(null)}>×</button>
              </span>
            </div>
          </div>
        )}

        <div className="panel-content" ref={scrollRef}>
          <div className="message-list">
            {messages.length === 0 && !isProcessing && (
              <div style={{ opacity: 0.1, marginTop: '20vh', textAlign: 'center', userSelect: 'none' }}>
                <h1 style={{ fontSize: '6rem', fontWeight: 900, marginBottom: '-1rem' }}>RESEMBLER</h1>
                <p className="mono-label" style={{ fontSize: '1rem' }}>Multi-Agent Research Intelligence</p>
              </div>
            )}
            
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role} ${m.isFinal ? 'ai-final' : ''} staggered-item`}>
                <span className="mono-label">{m.role === 'user' ? 'Scientist' : 'Resembler Core'}</span>
                <div style={{ fontSize: '1.05rem', lineHeight: 1.6 }} className="markdown-content">
                  {m.role === 'ai' ? (
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                  )}
                </div>
                {m.confidence && (
                  <div className="mono-label" style={{ marginTop: '1rem', color: m.confidence === 'high' ? 'var(--success)' : 'var(--accent)', fontWeight: 800 }}>
                    Confidence: {m.confidence}
                  </div>
                )}
                {m.chunks && m.chunks.length > 0 && (
                  <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                    <span className="mono-label">Supporting Citations:</span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
                      {m.chunks.slice(0, 3).map((c, j) => (
                        <div key={j} className="citation-chunk">
                          <span className="mono-label" style={{ color: 'var(--accent)', marginBottom: '0.25rem' }}>[{c.paper_id}] {c.section_heading}</span>
                          <p style={{ fontSize: '0.8rem', opacity: 0.9 }}>{c.content.substring(0, 180)}...</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {clarificationData && (
              <div className="message ai staggered-item" style={{ background: '#FFF7ED', border: '1px dashed var(--accent)', maxWidth: '100%' }}>
                <span className="mono-label" style={{ color: 'var(--accent)' }}>System / Ambiguity Detected</span>
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Which paper should I analyze for this request?</h3>
                <div className="selection-wall">
                  {clarificationData.papers.map((p, i) => (
                    <button key={i} className="selection-card" onClick={() => handlePaperSelect(p)}>
                      <h4 style={{ fontSize: '0.85rem' }}>{p.title || p.paper_id}</h4>
                      <span className="mono-label" style={{ fontSize: '0.6rem', marginBottom: 0 }}>Select this document</span>
                    </button>
                  ))}
                  <button className="selection-card" onClick={() => { setClarificationData(null); startQuery(); }} style={{ borderColor: 'var(--border)' }}>
                    <h4 style={{ fontSize: '0.85rem' }}>All Papers</h4>
                    <span className="mono-label" style={{ fontSize: '0.6rem', marginBottom: 0 }}>Search everything</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="chat-input-container">
          <input 
            className="chat-input"
            placeholder={activeContext ? `Ask about "${activeContext.title || activeContext.paper_id}"...` : "Ask Resembler anything..."}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && startQuery()}
            disabled={isProcessing}
          />
          <button 
            className="soft-button" 
            style={{ position: 'absolute', right: '2.5rem', top: '2rem', padding: '0.5rem 1.2rem', fontSize: '0.8rem' }}
            onClick={() => startQuery()}
            disabled={isProcessing || !query.trim()}
          >
            {isProcessing ? 'Thinking...' : 'Analyze'}
          </button>
        </div>
      </div>

      {/* Right Panel: Agent Status */}
      <div className="panel right-panel">
        <div className="panel-header">
          <div>
            <span className="mono-label">Internal / Logs</span>
            <h2>Multi-Agent</h2>
          </div>
        </div>
        <div className="panel-content" ref={logScrollRef}>
          <div className="log-list">
            <div className="log-entry active">
              <span className="mono-label">System Initialized</span>
              <p>v1.0.0-PRO-MAX</p>
            </div>
            
            {logs.map((l, i) => (
              <div key={i} className={`log-entry ${l.status === 'completed' ? 'done' : 'active'}`}>
                <span className="mono-label">{l.step} {l.duration ? `(${l.duration}s)` : ''}</span>
                <p>{l.status === 'started' ? 'Initializing agent...' : 'Process completed successfully.'}</p>
                {l.data && l.data.strategy_notes && (
                  <p style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '0.25rem' }}>Strategy: {l.data.strategy_notes}</p>
                )}
                {l.data && l.data.num_results && (
                  <p style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '0.25rem' }}>Retrieved: {l.data.num_results} Chunks</p>
                )}
              </div>
            ))}
            
            {isProcessing && (
              <div className="log-entry active pulse-log">
                <span className="mono-label">Awaiting Output...</span>
                <p>Synthesizing structured answer.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
