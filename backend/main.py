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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Global state
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG Engine at startup."""
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


# ── FastAPI App ───────────────────────────────────────────────────────────────

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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_engine_loaded": rag_engine is not None,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept an uploaded .mp4 video and return a session ID."""
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
async def get_frame(filename: str, frame_idx: int):
    """
    Utility endpoint: frontend can fetch a specific frame as an image
    if it wants to display the exact frame being analyzed.
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Video not found")
        
    # We could implement frame extraction here, but for now we'll rely on the 
    # frontend's <video> player to stay in sync.
    # This is a placeholder if needed later.
    return {"status": "not_implemented"}


@app.websocket("/ws/analyze/{filename}")
async def websocket_analyze(websocket: WebSocket, filename: str):
    """
    WebSocket endpoint for real-time video analysis.
    Processes the video at 1 FPS and streams results back to the client.
    """
    await websocket.accept()
    
    file_path = UPLOAD_DIR / filename
    if not file_path.exists() or not rag_engine:
        await websocket.send_json({"error": "File not found or engine not ready"})
        await websocket.close()
        return

    print(f"[WebSocket] Starting analysis for {filename}")
    
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        await websocket.send_json({"error": "Failed to open video file"})
        await websocket.close()
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"  Video: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")
    
    # We want to process roughly 1 frame per second
    frame_interval = max(1, int(fps))
    
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process 1 frame per second
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                
                # Offload heavy ML to a separate thread to keep WS responsive
                result = await asyncio.to_thread(
                    rag_engine.analyze_frame, frame, timestamp
                )
                
                result["frame_index"] = frame_idx
                result["type"] = "analysis_result"
                
                # Send to frontend
                await websocket.send_json(result)
                
                # Small delay to allow WS to flush and prevent overwhelming client
                await asyncio.sleep(0.05)
                
            frame_idx += 1
            
        # EOF
        await websocket.send_json({"type": "complete"})
        
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected from {filename}")
    except Exception as e:
        print(f"[WebSocket] Error during analysis: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        cap.release()
        print(f"[WebSocket] Analysis finished for {filename}")
