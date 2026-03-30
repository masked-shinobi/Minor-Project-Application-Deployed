import React, { useState, useEffect, useRef } from 'react';
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
  const [curretPlan, setCurrentPlan] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");

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

    setUploadStatus("Uploading...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (resp.ok) {
        setUploadStatus("Success");
        fetchStats();
        fetchPapers();
        setTimeout(() => setUploadStatus(null), 3000);
      } else {
        setUploadStatus("Failed");
      }
    } catch (err) {
      console.error(err);
      setUploadStatus("Error");
    }
  };

  const startQuery = () => {
    if (!query || isProcessing) return;

    setIsProcessing(true);
    setLogs([]);
    setMessages(prev => [...prev, { role: 'user', content: query }]);
    
    // Connect WebSocket
    ws.current = new WebSocket(`${WS_BASE}/ws/reasoning`);
    
    ws.current.onopen = () => {
      ws.current.send(JSON.stringify({ query }));
    };

    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.error) {
        setIsProcessing(false);
        setLogs(prev => [...prev, { step: 'error', status: 'error', data: msg.error }]);
        return;
      }

      // Handle Log Steps
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
        fetchStats();
      } else {
        setLogs(prev => {
          const filtered = prev.filter(l => l.step !== msg.step);
          return [...filtered, msg];
        });
        if (msg.step === "planning") setCurrentPlan(msg.data);
      }
    };

    ws.current.onerror = (err) => {
      console.error("WS Error:", err);
      setIsProcessing(false);
    };
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
          <div className="stats-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1px', background: 'var(--border)', marginBottom: '2rem' }}>
            <div style={{ background: 'var(--bg-layer-2)', padding: '1rem' }}>
              <span className="mono-label">Papers</span>
              <h3 style={{ fontSize: '1.5rem', color: 'var(--accent)' }}>{stats.total_papers}</h3>
            </div>
            <div style={{ background: 'var(--bg-layer-2)', padding: '1rem' }}>
              <span className="mono-label">Chunks</span>
              <h3 style={{ fontSize: '1.5rem', color: 'var(--accent)' }}>{stats.total_chunks}</h3>
            </div>
          </div>

          <label className="brutalist-button" style={{ width: '100%', justifyContent: 'center', marginBottom: '1.5rem' }}>
            <input type="file" style={{ display: 'none' }} onChange={handleUpload} accept=".pdf" />
            {uploadStatus || "Upload Research PDF"}
          </label>

          <span className="mono-label">Indexed Documents</span>
          {papers.map((p, i) => (
            <div key={i} className="paper-card staggered-item" style={{ animationDelay: `${i * 0.05}s` }}>
              <h4 style={{ fontSize: '0.9rem', marginBottom: '0.25rem' }}>{p.title || p.paper_id}</h4>
              <span className="mono-label" style={{ fontSize: '0.6rem' }}>{p.total_pages} Pages | {p.total_chunks || '?'} Chunks</span>
            </div>
          ))}
        </div>
      </div>

      {/* Center Panel: Reasoning Stream */}
      <div className="panel" style={{ background: 'var(--bg-deep)' }}>
        <div className="panel-header">
          <div>
            <span className="mono-label">Stream / Reasoning</span>
            <h2>Agent Pipeline</h2>
          </div>
          <div className="status-indicator">
            <div className={`pulse ${isProcessing ? '' : 'inactive'}`} />
            <span style={{ fontSize: '0.65rem' }}>System: {isProcessing ? 'Thinking' : 'Idle'}</span>
          </div>
        </div>
        <div className="panel-content" ref={scrollRef} style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="message-list">
            {messages.length === 0 && (
              <div style={{ opacity: 0.3, marginTop: '20vh', textAlign: 'center' }}>
                <h1 style={{ fontSize: '4rem', opacity: 0.5 }}>RESEMBLER</h1>
                <p className="mono-label">Waiting for research query...</p>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role} ${m.isFinal ? 'ai-final' : ''} staggered-item`}>
                <span className="mono-label">{m.role === 'user' ? 'Scientist' : 'Resembler Core'}</span>
                <div style={{ fontSize: '1.1rem', whiteSpace: 'pre-wrap' }}>{m.content}</div>
                {m.confidence && (
                  <div className="mono-label" style={{ marginTop: '1rem', color: m.confidence === 'high' ? 'var(--success)' : 'var(--accent)' }}>
                    Confidence: {m.confidence}
                  </div>
                )}
                {m.chunks && (
                  <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                    <span className="mono-label">Found Citations:</span>
                    {m.chunks.slice(0, 3).map((c, j) => (
                      <div key={j} className="citation-chunk">
                        <span className="mono-label" style={{ color: 'var(--accent)' }}>[{c.paper_id}] Section: {c.section_heading}</span>
                        <p style={{ marginTop: '0.5rem', opacity: 0.8 }}>{c.content.substring(0, 200)}...</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="chat-input-container">
          <input 
            className="chat-input"
            placeholder="Ask Resembler about your papers..." 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && startQuery()}
            disabled={isProcessing}
          />
          <button 
            className="brutalist-button" 
            style={{ position: 'absolute', right: '2rem', top: '2.1rem', padding: '0.4rem 1rem' }}
            onClick={startQuery}
            disabled={isProcessing}
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
