"""
Blind-Sight RAG — FastAPI Server
================================
Provides the REST endpoints for file upload and the WebSocket endpoint
for streaming real-time analysis back to the React frontend.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Global state
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    Application lifespan context manager.
    
    Initializes the RAG Engine at startup and ensures proper cleanup on shutdown.
    If GEMINI_API_KEY is not set, displays a warning but allows the server to start
    (analysis will fail until the key is provided).
    
    Args:
        app (FastAPI): FastAPI application instance
    """
    global rag_engine
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("⚠ WARNING: GEMINI_API_KEY not found in environment.")
        print("  Please create a .env file with your Google Gemini API key.")
    
    try:
        rag_engine = RAGEngine(gemini_api_key=gemini_key or "mock_key")
    except Exception as e:
        print(f"Failed to initialize RAG Engine: {e}")
        print("Server will start, but analysis will fail until fixed.")
        
    yield
    
    if rag_engine:
        rag_engine.close()


# ── FastAPI App ───────────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Blind-Sight RAG Backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns the current health status of the server and indicates whether
    the RAG engine has been successfully initialized.
    
    Returns:
        Dict[str, Any]: Status information with keys:
            - status: str, "healthy" if server is running
            - rag_engine_loaded: bool, whether RAG engine initialized successfully
    """
    return {
        "status": "healthy",
        "rag_engine_loaded": rag_engine is not None,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a video file for analysis.
    
    Accepts MP4, MOV, or AVI video files and stores them for processing.
    Generates a unique session ID for tracking the upload.
    
    Args:
        file (UploadFile): Video file to upload (must be .mp4, .mov, or .avi)
        
    Returns:
        JSONResponse: Response containing:
            - session_id: str, unique identifier for this upload session
            - filename: str, stored filename (session_id + original name)
            - message: str, confirmation message
            
    Raises:
        HTTPException: 400 if file is not a supported video format
    """
    if not file.filename.endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(400, "Only video files are supported")
        
    session_id = os.urandom(8).hex()
    safe_filename = f"{session_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    return JSONResponse({
        "session_id": session_id,
        "filename": safe_filename,
        "message": "Upload successful"
    })


@app.get("/api/frame/{filename}/{frame_idx}")
async def get_frame(filename: str, frame_idx: int) -> Dict[str, str]:
    """
    Retrieve a specific frame from an uploaded video (placeholder).
    
    This endpoint provides a utility to fetch individual frames if the frontend
    needs to display or verify specific frames being analyzed. Currently returns
    a placeholder response pending implementation.
    
    Args:
        filename (str): Name of the uploaded video file
        frame_idx (int): Index of the frame to retrieve
        
    Returns:
        Dict[str, str]: Placeholder response
        
    Raises:
        HTTPException: 404 if video file is not found
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Video not found")
        
    return {"status": "not_implemented"}


@app.websocket("/ws/analyze/{filename}")
async def websocket_analyze(websocket: WebSocket, filename: str) -> None:
    """
    WebSocket endpoint for real-time video analysis streaming.
    
    Establishes a persistent WebSocket connection to stream real-time threat
    analysis results as the video plays. Stays synchronized with frontend playback
    by tracking elapsed time and analyzing frames accordingly.
    
    Flow:
        1. Accept WebSocket connection
        2. Validate video file and RAG engine availability
        3. Extract video properties (FPS, duration, frame count)
        4. For each elapsed time:
           - Seek to the corresponding frame
           - Analyze frame using RAG Engine
           - Send analysis result to client
        5. Send completion message when done
    
    Args:
        websocket (WebSocket): WebSocket connection from client
        filename (str): Name of the uploaded video file to analyze
        
    Message Format (sent to client):
        Analysis result: {
            "type": "analysis_result",
            "threat_level": int,
            "warning": str,
            "hazards_detected": [str],
            "timestamp": float,
            "processing_time_ms": int
        }
        
        Completion: {"type": "complete"}
        
        Error: {"error": str}
    """
    await websocket.accept()
    
    file_path = UPLOAD_DIR / filename
    if not file_path.exists() or not rag_engine:
        await websocket.send_json({"error": "Backend Engine is not initialized or file missing."})
        await websocket.close()
        return

    print(f"[WebSocket] Starting LIVE analysis for {filename}")
    cap = cv2.VideoCapture(str(file_path))
    
    if not cap.isOpened():
        await websocket.send_json({"error": "Failed to open video file"})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    try:
        start_time = time.time()
        
        while True:
            # Calculate how much time has passed since video started playing
            elapsed_time = time.time() - start_time
            
            if elapsed_time > duration:
                break
                
            # Seek video to the exact elapsed time frame
            cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_time * 1000)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame in a separate thread
            result = await asyncio.to_thread(
                rag_engine.analyze_frame, frame, elapsed_time
            )
            
            result["type"] = "analysis_result"
            await websocket.send_json(result)
            
            # Brief pause to prevent flooding WebSocket
            await asyncio.sleep(0.1)
            
        await websocket.send_json({"type": "complete"})
        
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        cap.release()