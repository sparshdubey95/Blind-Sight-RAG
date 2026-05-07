import React, { useEffect, useRef } from 'react';
import { ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react';

const AnalysisFeed = ({ events = [] }) => {
  const feedRef = useRef(null);

  // Auto-scroll to bottom and read out loud when new events arrive
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
    
    // Audio Feedback for the visually impaired
    if (events.length > 0 && window.speechSynthesis) {
      const latestEvent = events[events.length - 1];
      // Speak only if there is a threat (avoid reading every 1s if path is clear)
      if (latestEvent.threat_level > 0) {
        window.speechSynthesis.cancel(); // Prevent pile-up
        const utterance = new SpeechSynthesisUtterance(latestEvent.warning);
        window.speechSynthesis.speak(utterance);
      }
    }
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="glass-panel rounded-2xl flex-1 flex flex-col items-center justify-center text-text-muted p-8 text-center border-dashed">
        <div className="w-12 h-12 rounded-full bg-white/[0.02] flex items-center justify-center mb-4">
          <svg className="w-6 h-6 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <p className="font-medium">Waiting for video stream...</p>
        <p className="text-xs mt-2 opacity-60">Upload a video to start the real-time RAG analysis.</p>
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-2xl flex-1 flex flex-col overflow-hidden relative">
      <div className="p-4 border-b border-white/[0.04] bg-bg-surface/50 backdrop-blur flex justify-between items-center z-10">
        <h3 className="text-sm font-semibold tracking-wide text-text-secondary flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-emerald animate-pulse" />
          LIVE ANALYSIS FEED
        </h3>
        <span className="text-xs bg-white/[0.04] px-2 py-1 rounded-md text-text-muted font-mono">
          {events.length} EVENTS
        </span>
      </div>
      
      <div 
        ref={feedRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar"
        style={{ scrollBehavior: 'smooth' }}
      >
        {events.map((event, idx) => {
          const isHighThreat = event.threat_level >= 7;
          const isMediumThreat = event.threat_level >= 4 && event.threat_level < 7;
          
          let borderColor = 'border-white/[0.04]';
          let bgColor = 'bg-white/[0.02]';
          let icon = <ShieldCheck className="w-5 h-5 text-accent-emerald" />;
          
          if (isHighThreat) {
            borderColor = 'border-accent-red/30';
            bgColor = 'bg-accent-red/[0.05]';
            icon = <ShieldAlert className="w-5 h-5 text-accent-red" />;
          } else if (isMediumThreat) {
            borderColor = 'border-accent-amber/30';
            bgColor = 'bg-accent-amber/[0.05]';
            icon = <AlertTriangle className="w-5 h-5 text-accent-amber" />;
          }

          return (
            <div 
              key={`${event.timestamp}-${idx}`} 
              className={`border rounded-xl p-4 transition-all duration-300 animate-in fade-in slide-in-from-bottom-4 ${borderColor} ${bgColor}`}
            >
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2">
                  {icon}
                  <span className="font-mono text-xs font-medium bg-white/[0.06] px-2 py-0.5 rounded text-text-secondary">
                    {event.timestamp.toFixed(1)}s
                  </span>
                </div>
                <div className="text-xs font-mono text-text-muted">
                  {(event.processing_time_ms / 1000).toFixed(2)}s inferred
                </div>
              </div>
              
              <div className={`p-3 rounded-lg mb-3 ${isHighThreat ? 'bg-accent-red/20 border-l-4 border-accent-red' : isMediumThreat ? 'bg-accent-amber/20 border-l-4 border-accent-amber' : 'bg-white/5'}`}>
                <p className={`text-lg font-bold leading-snug ${isHighThreat ? 'text-accent-red brightness-150' : 'text-white'}`}>
                  {event.warning}
                </p>
              </div>
              
              {event.hazards_detected && event.hazards_detected.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {event.hazards_detected.map((hazard, hIdx) => (
                    <span 
                      key={hIdx} 
                      className={`text-[10px] uppercase tracking-wider px-2 py-1 rounded-md font-medium border
                        ${isHighThreat ? 'bg-accent-red/10 text-accent-red border-accent-red/20' : 
                          isMediumThreat ? 'bg-accent-amber/10 text-accent-amber border-accent-amber/20' : 
                          'bg-white/[0.04] text-text-muted border-white/[0.06]'}`}
                    >
                      {hazard}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AnalysisFeed;
