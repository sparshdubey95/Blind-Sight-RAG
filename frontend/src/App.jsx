import React, { useState, useEffect, useRef } from 'react';
import VideoPlayer from './components/VideoPlayer';
import ThreatDial from './components/ThreatDial';
import AnalysisFeed from './components/AnalysisFeed';

function App() {
  const [videoUrl, setVideoUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [filename, setFilename] = useState(null);
  const [events, setEvents] = useState([]);
  const [currentThreat, setCurrentThreat] = useState(0);
  const wsRef = useRef(null);

  // Handle file upload
  const handleUpload = async (file) => {
    setIsUploading(true);
    setEvents([]);
    setCurrentThreat(0);
    
    // Create local object URL for immediate playback
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setSessionId(data.session_id);
      setFilename(data.filename);
      setIsUploading(false);
      
      // Start WebSocket connection
      connectWebSocket(data.filename);
    } catch (error) {
      console.error("Upload failed:", error);
      setIsUploading(false);
      alert("Failed to upload video to backend.");
    }
  };

  const connectWebSocket = (fname) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const wsUrl = `ws://localhost:8000/ws/analyze/${fname}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'complete') {
        console.log("Analysis complete");
        return;
      }
      
      if (data.error) {
        console.error("WS Error:", data.error);
        return;
      }

      if (data.type === 'analysis_result') {
        setCurrentThreat(data.threat_level);
        setEvents(prev => [...prev, data]);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket Error:", error);
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
    };
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  return (
    <div className="min-h-screen flex flex-col bg-bg-base text-text-primary font-sans relative">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-accent-violet/10 via-bg-base to-bg-base pointer-events-none"></div>

      {/* Header */}
      <header className="border-b border-white/[0.06] bg-bg-surface/60 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center text-white font-black text-sm bg-gradient-to-br from-accent-cyan to-accent-violet shadow-[0_2px_12px_rgba(6,182,212,0.3)]">
              BS
            </div>
            <div>
              <h1 className="text-lg font-bold leading-tight gradient-text">Blind-Sight RAG</h1>
              <p className="text-[10px] text-text-muted font-medium tracking-widest uppercase -mt-0.5">
                Active Analysis Dashboard
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
             {wsRef.current && wsRef.current.readyState === WebSocket.OPEN && (
               <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.06] text-xs text-text-muted">
                 <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald animate-pulse" />
                 <span>Stream Active</span>
               </div>
             )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
        
        {/* Left Column (Video) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="aspect-video">
            <VideoPlayer 
              videoUrl={videoUrl} 
              onUpload={handleUpload} 
              isUploading={isUploading} 
            />
          </div>
          
          {/* Hardware Specs Panel */}
          <div className="glass-panel rounded-2xl p-4 flex gap-4 text-xs">
            <div className="flex-1 bg-white/[0.02] rounded-lg p-3 border border-white/[0.04]">
              <div className="text-text-muted uppercase tracking-wider mb-1">Embedding Engine</div>
              <div className="font-medium font-mono text-accent-cyan">DINOv2 (ViT-Small)</div>
            </div>
            <div className="flex-1 bg-white/[0.02] rounded-lg p-3 border border-white/[0.04]">
              <div className="text-text-muted uppercase tracking-wider mb-1">Vector DB</div>
              <div className="font-medium font-mono text-accent-violet">FAISS (FlatIP)</div>
            </div>
            <div className="flex-1 bg-white/[0.02] rounded-lg p-3 border border-white/[0.04]">
              <div className="text-text-muted uppercase tracking-wider mb-1">Reasoning Model</div>
              <div className="font-medium font-mono text-accent-amber">Gemini 1.5 Flash</div>
            </div>
          </div>
        </div>

        {/* Right Column (Analysis) */}
        <div 
          className="lg:col-span-5 flex flex-col gap-6 transition-all duration-500"
          style={{
            filter: currentThreat >= 7 ? 'drop-shadow(0 0 15px rgba(239, 68, 68, 0.2))' : 'none'
          }}
        >
          <ThreatDial threatLevel={currentThreat} />
          <AnalysisFeed events={events} />
        </div>

      </main>
    </div>
  );
}

export default App;
