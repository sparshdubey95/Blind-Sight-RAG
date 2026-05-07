import React, { useRef, useState } from 'react';
import { UploadCloud, FileVideo, Play, Pause } from 'lucide-react';

const VideoPlayer = ({ videoUrl, onUpload, isUploading }) => {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isHovering, setIsHovering] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play();
        setIsPlaying(true);
      } else {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    }
  };

  if (!videoUrl && !isUploading) {
    return (
      <div 
        className="glass-panel rounded-2xl h-full flex flex-col items-center justify-center border-dashed border-2 cursor-pointer hover:bg-white/[0.02] transition-colors"
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="w-16 h-16 rounded-full bg-white/[0.04] flex items-center justify-center mb-6">
          <UploadCloud className="w-8 h-8 text-accent-cyan" />
        </div>
        <h3 className="text-lg font-medium text-text-primary mb-2">Upload Camera Feed</h3>
        <p className="text-sm text-text-muted mb-6 text-center max-w-xs">
          Select an MP4 video to begin real-time hazard analysis.
        </p>
        <button className="bg-accent-cyan text-bg-base font-semibold px-6 py-2.5 rounded-lg hover:brightness-110 transition-all shadow-[0_0_15px_rgba(6,182,212,0.3)]">
          Select Video
        </button>
        <input 
          type="file" 
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="video/mp4,video/quicktime" 
          className="hidden" 
        />
      </div>
    );
  }

  if (isUploading) {
    return (
      <div className="glass-panel rounded-2xl h-full flex flex-col items-center justify-center">
        <div className="w-16 h-16 relative flex items-center justify-center mb-6">
          <div className="absolute inset-0 border-4 border-white/[0.04] rounded-full"></div>
          <div className="absolute inset-0 border-4 border-accent-cyan border-t-transparent rounded-full animate-spin"></div>
          <FileVideo className="w-6 h-6 text-accent-cyan animate-pulse" />
        </div>
        <h3 className="text-lg font-medium text-text-primary mb-2">Uploading and Initializing...</h3>
        <p className="text-sm text-text-muted text-center max-w-xs">
          Preparing the RAG engine for real-time analysis.
        </p>
      </div>
    );
  }

  return (
    <div 
      className="glass-panel rounded-2xl h-full overflow-hidden relative group"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <video 
        ref={videoRef}
        src={videoUrl}
        className="w-full h-full object-cover"
        autoPlay
        muted
        loop
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />
      
      {/* HUD Overlay */}
      <div className="absolute inset-0 pointer-events-none border border-white/[0.05] rounded-2xl"></div>
      
      {/* Center Crop Reticle (to visualize what DINOv2 sees) */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3/5 h-3/5 border border-dashed border-white/20 pointer-events-none flex items-center justify-center">
         <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-accent-cyan/50"></div>
         <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-accent-cyan/50"></div>
         <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-accent-cyan/50"></div>
         <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-accent-cyan/50"></div>
      </div>
      
      {/* Controls */}
      <div className={`absolute bottom-0 left-0 w-full p-4 bg-gradient-to-t from-black/80 to-transparent transition-opacity duration-300 ${isHovering ? 'opacity-100' : 'opacity-0'}`}>
        <button 
          onClick={togglePlay}
          className="w-10 h-10 rounded-full bg-white/10 backdrop-blur flex items-center justify-center hover:bg-white/20 transition-colors"
        >
          {isPlaying ? (
            <Pause className="w-5 h-5 text-white" />
          ) : (
            <Play className="w-5 h-5 text-white ml-0.5" />
          )}
        </button>
      </div>
      
      {/* Upload New Button */}
      <div className="absolute top-4 right-4 z-10">
        <button 
          onClick={() => fileInputRef.current?.click()}
          className="bg-black/50 backdrop-blur px-3 py-1.5 rounded-lg text-xs font-medium border border-white/10 hover:bg-white/10 transition-colors text-white"
        >
          Change Video
        </button>
        <input 
          type="file" 
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="video/mp4,video/quicktime" 
          className="hidden" 
        />
      </div>
    </div>
  );
};

export default VideoPlayer;
