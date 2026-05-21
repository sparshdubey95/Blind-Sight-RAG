"""
Blind-Sight RAG — RAG Inference Engine
======================================
Combines DINOv2 for frame embedding, FAISS for vector similarity search,
SQLite for context retrieval, and Gemini 1.5 Flash for threat reasoning.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import faiss
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_ROOT / "hazards.index"
DB_FILE = PROJECT_ROOT / "metadata.db"

DINOV2_MODEL = "facebook/dinov2-small"
IMG_SIZE = 224


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for real-time hazard detection.
    
    Combines DINOv2 for frame embedding, FAISS for vector similarity search,
    SQLite for context retrieval, and Gemini 1.5 Flash for threat reasoning.
    
    Attributes:
        index (faiss.Index): FAISS index for efficient similarity search over embeddings
        conn (sqlite3.Connection): SQLite database connection for hazard metadata
        cur (sqlite3.Cursor): Database cursor for executing queries
        device (torch.device): Computing device (CUDA or CPU)
        processor (AutoImageProcessor): DINOv2 image preprocessor
        model (AutoModel): DINOv2 vision transformer for generating embeddings
        gemini (genai.GenerativeModel): Gemini API client for threat analysis
    """
    
    def __init__(self, gemini_api_key: str) -> None:
        """
        Initialize the RAG Engine and load all required models.
        
        Args:
            gemini_api_key (str): Google Gemini API key for threat analysis
            
        Raises:
            RuntimeError: If FAISS index or SQLite database is not found.
                         Run idd_data_prep.py first to generate these files.
            Exception: If model loading or API configuration fails
        """
        print("[RAG Engine] Initializing...")

        # 1. Setup FAISS & SQLite
        if not INDEX_FILE.exists() or not DB_FILE.exists():
            raise RuntimeError(
                f"FAISS index or SQLite DB not found.\n"
                f"  Expected: {INDEX_FILE}\n"
                f"  Expected: {DB_FILE}\n"
                f"  Run idd_data_prep.py first to generate these files."
            )

        print(f"  Loading FAISS index from {INDEX_FILE}...")
        self.index = faiss.read_index(str(INDEX_FILE))
        print(f"  ✓ Index loaded ({self.index.ntotal} vectors)")

        print(f"  Connecting to SQLite at {DB_FILE}...")
        self.conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
        self.cur = self.conn.cursor()
        print("  ✓ Database connected")

        # 2. Setup DINOv2
        print("  Loading DINOv2 ViT-Small...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
        self.model = AutoModel.from_pretrained(DINOV2_MODEL).to(self.device)
        self.model.eval()
        print(f"  ✓ DINOv2 ready on {self.device}")

        # 3. Setup Gemini
        print("  Configuring Gemini 1.5 Flash...")
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")
        print("  ✓ Gemini API ready")

    def _apply_homography_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to focus on the main road area.
        
        Applies the same homography transformation used during data preparation
        to standardize the viewing perspective and crop to the relevant region.
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV)
            
        Returns:
            np.ndarray: Warped image of size (IMG_SIZE, IMG_SIZE) in BGR format
        """
        h, w = image.shape[:2]
        y_top = int(h * 0.20)
        y_bot = int(h * 0.80)
        x_left = 0
        x_right = int(w * 0.75)

        src_pts = np.float32([
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bot],
            [x_left, y_bot],
        ])

        dst_pts = np.float32([
            [0, 0],
            [IMG_SIZE, 0],
            [IMG_SIZE, IMG_SIZE],
            [0, IMG_SIZE],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (IMG_SIZE, IMG_SIZE))
        return warped

    def embed_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a 384-dimensional DINOv2 embedding from a video frame.
        
        Processes the input frame through perspective transformation and DINOv2
        to produce a normalized embedding suitable for similarity search.
        
        Args:
            frame (np.ndarray): Input frame in BGR format (OpenCV)
            
        Returns:
            np.ndarray: Normalized 384-d float32 embedding vector
        """
        cropped = self._apply_homography_crop(frame)
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def retrieve_context(self, vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search FAISS index and retrieve context from SQLite database.
        
        Performs similarity search to find the k most similar historical scenarios
        in the FAISS index, then retrieves corresponding metadata from the database.
        
        Args:
            vector (np.ndarray): Query embedding (1 x 384 float32)
            top_k (int): Number of similar results to retrieve (default: 3)
            
        Returns:
            List[Dict[str, Any]]: List of context dictionaries with keys:
                - distance: float, similarity distance from FAISS
                - vector_id: int, ID of the vector in the index
                - image_path: str, path to the reference image
                - description: str, textual description of the hazard
        """
        # FAISS search expects 2D array
        distances, indices = self.index.search(vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            # Fetch from SQLite
            self.cur.execute("SELECT image_path, description FROM hazards WHERE vector_id = ?", (int(idx),))
            row = self.cur.fetchone()
            if row:
                results.append({
                    "distance": float(dist),
                    "vector_id": int(idx),
                    "image_path": row[0],
                    "description": row[1]
                })
        return results

    def _encode_image_for_gemini(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Encode OpenCV frame as JPEG for Gemini API transmission.
        
        Resizes the frame for efficiency and encodes it as JPEG with compression
        to reduce bandwidth usage while maintaining quality for analysis.
        
        Args:
            frame (np.ndarray): Input frame in BGR format (OpenCV)
            
        Returns:
            Dict[str, Any]: Dictionary with keys:
                - mime_type: "image/jpeg"
                - data: bytes, JPEG-encoded image data
                
        Raises:
            ValueError: If JPEG encoding fails
        """
        # Resize for faster upload (Gemini will resize anyway, but saves bandwidth)
        h, w = frame.shape[:2]
        new_w = 640
        new_h = int(h * (new_w / w))
        resized = cv2.resize(frame, (new_w, new_h))
        
        success, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            raise ValueError("Failed to encode frame as JPEG")
            
        return {
            "mime_type": "image/jpeg",
            "data": buffer.tobytes()
        }

    def analyze_frame(self, frame: np.ndarray, timestamp_sec: float) -> Dict[str, Any]:
        """
        Perform complete RAG-based analysis on a video frame.
        
        Embeds the frame, retrieves similar historical scenarios, and uses Gemini
        to analyze threats and generate warnings for the visually impaired user.
        
        Args:
            frame (np.ndarray): Input frame in BGR format (OpenCV)
            timestamp_sec (float): Video timestamp in seconds
            
        Returns:
            Dict[str, Any]: Analysis result with keys:
                - threat_level: int (0-10), severity of detected threat
                - warning: str, actionable guidance for the user
                - hazards_detected: List[str], specific hazards found
                - timestamp: float, video timestamp
                - processing_time_ms: int, frame processing duration
                
        Note:
            If Gemini API fails, returns a graceful error response with threat_level=0
        """
        t0 = time.time()
        vector = self.embed_frame(frame)
        contexts = self.retrieve_context(vector, top_k=3)
        context_descriptions = "\n".join([f"- {c['description']}" for c in contexts])
        
        prompt = f"""
You are an assistive AI for visually impaired pedestrians navigating streets.
Current frame timestamp: {timestamp_sec:.1f}s.
RETRIEVED HISTORICAL SCENARIOS:
{context_descriptions}

TASK:
1. Analyze the provided image. Detect obstacles, vehicles, uneven terrain, or hazards.
2. Determine current threat level (0-10):
   - Clear path = 0
   - Small obstacles (poles, people on sides) = 2 to 4
   - Direct path blocked or Vehicles approaching = 7 to 10
3. Warning MUST be 1 sentence. If threat > 0, tell them to "Stop" or "Change direction".

RESPOND STRICTLY IN VALID JSON FORMAT, without markdown:
{{
  "threat_level": <int>,
  "warning": "<string>",
  "hazards_detected": ["<array of strings>"]
}}
"""
        gemini_image = self._encode_image_for_gemini(frame)
        
        # Override Safety Settings so dashcam footage isn't blocked!
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        try:
            response = self.gemini.generate_content(
                [prompt, gemini_image],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
                safety_settings=safety_settings
            )
            
            # Clean markdown codeblocks if Gemini adds them
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            result = json.loads(raw_text.strip())
            
        except Exception as e:
            print(f"[RAG Engine] API Error: {e}")
            result = {
                "threat_level": 0,
                "warning": f"API Connection Error: {str(e)[:50]}...",
                "hazards_detected": []
            }
            
        result["timestamp"] = round(timestamp_sec, 2)
        result["processing_time_ms"] = round((time.time() - t0) * 1000)
        return result

    def close(self) -> None:
        """
        Clean up and release resources.
        
        Closes the SQLite database connection. Should be called during
        application shutdown to ensure proper resource cleanup.
        """
        if hasattr(self, 'conn'):
            self.conn.close()