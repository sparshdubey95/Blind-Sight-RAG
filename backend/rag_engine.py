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
    def __init__(self, gemini_api_key: str):
        """Initialize the RAG Engine."""
        print("[RAG Engine] Initializing...")

        # 1. Setup FAISS & SQLite
        if not INDEX_FILE.exists() or not DB_FILE.exists():
            raise RuntimeError("FAISS index or SQLite DB not found. Run idd_data_prep.py first.")

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
        """Apply the same perspective transform used during data prep."""
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
        """Process an OpenCV BGR frame into a 384-d normalized DINOv2 vector."""
        cropped = self._apply_homography_crop(frame)
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def retrieve_context(self, vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search FAISS and fetch context from SQLite."""
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

    def _encode_image_for_gemini(self, frame: np.ndarray) -> Dict:
        """Encode OpenCV frame as JPEG bytes for Gemini API."""
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
        Full RAG pipeline for a single frame.
        1. Embed frame
        2. Retrieve context
        3. Call Gemini
        """
        t0 = time.time()
        
        # 1. Embed
        vector = self.embed_frame(frame)
        
        # 2. Retrieve
        contexts = self.retrieve_context(vector, top_k=3)
        context_descriptions = "\n".join([f"- {c['description']}" for c in contexts])
        
        # 3. Formulate Prompt
        prompt = f"""
You are an assistive AI for visually impaired pedestrians navigating streets and environments.
I am providing you with the current wearable camera frame (at {timestamp_sec:.1f} seconds) and context from similar historical scenarios we have retrieved.

RETRIEVED SCENARIO CONTEXTS:
{context_descriptions}

TASK:
1. Analyze the provided image for any obstacles, vehicles, or hazards.
2. Consider the retrieved context to understand typical hazards in similar looking scenes.
3. Determine the current threat level to the visually impaired pedestrian using these STRICT RULES:
   - If ANY object or obstacle is detected in the path (e.g. pole, person, debris), threat_level MUST be at least 2 or 3. It cannot be 0 if there are objects.
   - If a CAR or other vehicle is spotted, the threat_level MUST be high (8-10).
   - If there is any hazard, the 'warning' MUST explicitly state to "change direction or stop" along with the hazard description.
   - Only return 0 if the path is completely clear of any potential obstacles.

RESPOND STRICTLY IN VALID JSON FORMAT:
{{
  "threat_level": <integer 0-10, based on rules above>,
  "warning": "<1-sentence concise warning. Must include 'change direction or stop' if a hazard is present>",
  "hazards_detected": ["<list>", "<of>", "<specific>", "<hazards>"]
}}
"""

        # 4. Call Gemini
        gemini_image = self._encode_image_for_gemini(frame)
        
        try:
            response = self.gemini.generate_content(
                [prompt, gemini_image],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2, # Low temp for consistency
                )
            )
            
            # Parse JSON
            try:
                result = json.loads(response.text)
                
                # --- LOGIC ENFORCEMENT ---
                # Fallback to guarantee the strict rules are met even if the LLM hallucinates
                
                hazards = result.get("hazards_detected", [])
                hazards_lower = " ".join(hazards).lower() if isinstance(hazards, list) else ""
                warning_lower = result.get("warning", "").lower()
                
                # Rule 1: Car/Vehicle -> High Threat (8-10)
                if any(v in hazards_lower for v in ["car", "vehicle", "truck", "bus", "auto", "bike", "motorcycle"]):
                    if result.get("threat_level", 0) < 8:
                        result["threat_level"] = 8
                    if "stop" not in warning_lower and "change direction" not in warning_lower:
                        result["warning"] = result.get("warning", "") + " Stop or change direction immediately."
                
                # Rule 2: Any object -> Needle Deflection (> 0)
                elif len(hazards) > 0:
                    if result.get("threat_level", 0) < 2:
                        result["threat_level"] = 2
                        
                # Rule 3: Ensure actionable warning for any hazard
                if result.get("threat_level", 0) > 0 and "stop" not in warning_lower and "change direction" not in warning_lower:
                     result["warning"] = result.get("warning", "") + " Please be cautious and change direction if necessary."
                     
            except json.JSONDecodeError:
                # Fallback if Gemini didn't return perfect JSON
                result = {
                    "threat_level": 5,
                    "warning": "Error parsing AI response, proceed with caution.",
                    "hazards_detected": ["Parsing error"]
                }
                
        except Exception as e:
            print(f"[RAG Engine] Gemini API Error: {e}")
            result = {
                "threat_level": 0,
                "warning": "Connection to reasoning engine failed.",
                "hazards_detected": []
            }
            
        # Add metadata to result
        result["timestamp"] = round(timestamp_sec, 2)
        result["processing_time_ms"] = round((time.time() - t0) * 1000)
        result["retrieved_contexts"] = contexts
        
        return result

    def close(self):
        """Cleanup resources."""
        if hasattr(self, 'conn'):
            self.conn.close()
