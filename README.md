# Blind-Sight RAG 🚗👁️

**Real-time AI-powered street hazard detection for visually impaired pedestrians.**

A Retrieval-Augmented Generation (RAG) system that combines **DINOv2** vision transformers, **FAISS** vector search, and **Google Gemini** AI to analyze dashcam footage in real-time and provide actionable audio warnings.

---

## Overview

Blind-Sight addresses a critical accessibility gap: **helping visually impaired individuals navigate busy Indian streets safely**. By processing video frames in real-time, the system:

1. **Detects hazards** using state-of-the-art vision transformers (DINOv2)
2. **Retrieves contextual information** about similar historical scenarios (FAISS + SQLite)
3. **Reasons about threats** using large language models (Google Gemini 1.5 Flash)
4. **Delivers guidance** to the user with a threat level and actionable warning

### Key Features

- ✅ **Real-time streaming analysis** via WebSocket for synchronized warnings
- ✅ **Low-latency inference** optimized for RTX 3050 (4GB VRAM)
- ✅ **RAG-augmented reasoning** using Indian traffic scenarios from IDD-Lite dataset
- ✅ **Multi-modal threat detection** (vehicles, pedestrians, road conditions)
- ✅ **React + Vite frontend** with intuitive video player interface
- ✅ **No Docker, no local LLMs** — cloud-based Gemini API

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         Frontend (React + Vite)             │
│   Video Upload & Real-time Analysis UI      │
└──────────────┬──────────────────────────────┘
               │ WebSocket
               │ (elapsed_time sync)
               ▼
┌─────────────────────────────────────────────┐
│      FastAPI Backend (main.py)              │
│  • Video upload endpoint                    │
│  • Frame extraction & preprocessing         │
│  • WebSocket streaming                      │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│      RAG Engine (rag_engine.py)             │
│                                             │
│  1. DINOv2: Frame → 384-d embedding        │
│  2. FAISS: Similarity search (top-3)        │
│  3. SQLite: Retrieve context descriptions   │
│  4. Gemini: Analyze + Reason + Warn         │
└─────────────────────────────────────────────┘
```

### Data Flow Per Frame

```
Video Frame (BGR) 
  ↓
[Homography Transform] → 224×224 image (pedestrian-focused perspective)
  ↓
[DINOv2 ViT-Small] → 384-d normalized embedding
  ↓
[FAISS Search] → Top-3 similar historical scenarios
  ↓
[SQLite Query] → Retrieve descriptions of similar hazards
  ↓
[Gemini 1.5 Flash] → Threat analysis + warning generation
  ↓
WebSocket → Real-time alert to user
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **CUDA 12.1** (RTX 3050 or similar; CPU fallback available)
- **Node.js 18+** (for frontend)
- **Google Gemini API key** (free tier available at https://makersuite.google.com/app/apikey)

### 1. Clone & Setup Backend

```bash
# Clone repository
git clone https://github.com/sparshdubey95/Blind-Sight-RAG.git
cd Blind-Sight-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

The system requires the **IDD-Lite dataset** (600 MB) for building the FAISS index:

```bash
# Create datasets folder
mkdir -p datasets

# Download idd-lite.tar.gz from:
# https://github.com/val-iisc/idd-lite
# Place the file in ./datasets/

# Or download programmatically (requires ~10-15 min):
# wget -P datasets https://drive.google.com/uc?id=DATASET_ID
```

### 3. Generate FAISS Index & Database

```bash
# Preprocess IDD-Lite images and build index
python idd_data_prep.py

# This generates:
#   - hazards.index (FAISS vector database)
#   - metadata.db (SQLite with hazard metadata)
```

**Expected output:**
```
[Data Prep] Extracting IDD-Lite dataset...
  ✓ Dataset extracted
[Data Prep] Collecting images...
  ✓ Found 600 images
[Data Prep] Building FAISS index...
  ✓ Index built: 600 vectors, 384-d
  ✓ SQLite database created with 600 entries
```

### 4. Configure Environment

```bash
# Create .env file with your Gemini API key
cat > .env << EOF
GEMINI_API_KEY=your_api_key_here
EOF
```

Get your free API key: https://makersuite.google.com/app/apikey

### 5. Start Backend Server

```bash
# Terminal 1: Backend
python -m uvicorn backend.main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`

### 6. Start Frontend

```bash
# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## Usage

### 1. Upload Video

- Click **"Choose Video"** button
- Select an MP4, MOV, or AVI file (ideally dashcam footage from Indian streets)
- System generates a **session ID** for tracking

### 2. Analyze

- Click **"Start Analysis"**
- Video plays in the player
- Real-time threat warnings stream via WebSocket
- Each frame displays:
  - **Threat Level** (0-10 scale)
  - **Warning** (actionable guidance)
  - **Detected Hazards** (list of threats)
  - **Processing Time** (latency in ms)

### 3. Export Results

- Download analysis results as JSON
- Results include frame-by-frame threat analysis with timestamps

---

## API Reference

### REST Endpoints

#### Health Check
```http
GET /api/health
```

Response:
```json
{
  "status": "healthy",
  "rag_engine_loaded": true
}
```

#### Upload Video
```http
POST /api/upload
Content-Type: multipart/form-data

file=<video_file>
```

Response:
```json
{
  "session_id": "a1b2c3d4",
  "filename": "a1b2c3d4_video.mp4",
  "message": "Upload successful"
}
```

### WebSocket Endpoint

#### Analyze Stream
```
WS /ws/analyze/{filename}
```

**Message Format:**

Receive (per frame):
```json
{
  "type": "analysis_result",
  "threat_level": 7,
  "warning": "Stop! Oncoming vehicle detected.",
  "hazards_detected": ["vehicle", "intersection"],
  "timestamp": 2.5,
  "processing_time_ms": 1250
}
```

Completion:
```json
{
  "type": "complete"
}
```

---

## System Requirements

### Hardware (Minimum)

| Component | Requirement |
|-----------|----------|
| GPU | RTX 3050 (4GB VRAM) or equivalent |
| RAM | 8GB |
| Storage | 10GB (FAISS index + dataset) |
| CPU | 6+ cores recommended |

### Software

| Package | Version |
|---------|----------|
| PyTorch | >= 2.1 |
| Transformers | >= 4.36 |
| FAISS | >= 1.7.4 |
| FastAPI | >= 0.110 |
| React | 18+ |
| Vite | 5+ |

---

## Configuration

### Performance Tuning

Edit `backend/rag_engine.py` to adjust:

```python
DINOV2_MODEL = "facebook/dinov2-small"  # Or dinov2-large for better accuracy
BATCH_SIZE = 16  # Increase if you have >6GB VRAM
TOP_K = 3  # Number of similar scenarios to retrieve
```

### Safety Settings

Gemini safety filters are disabled for dashcam footage (may contain unsafe imagery). Modify in `analyze_frame()`:

```python
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    # ... etc
}
```

---

## Dataset: Indian Driving Dataset (IDD-Lite)

The system uses **IDD-Lite** — a curated subset of the **Indian Driving Dataset**:

- **600 images** from dashcam footage on Indian streets
- **30 traffic scenarios** represented (dense traffic, potholes, pedestrians, etc.)
- **Geographically diverse** (Bangalore, Hyderabad, Mumbai)
- **High-resolution** (1920×1080)

**Citation:**
```
@article{idd,
  title={IDD: A Dataset for Autonomous Driving in India},
  author={Varma et al.},
  journal={arXiv},
  year={2021}
}
```

---

## Troubleshooting

### Issue: "FAISS index or SQLite DB not found"

**Solution:**
```bash
python idd_data_prep.py
```

Ensure `hazards.index` and `metadata.db` exist in project root.

### Issue: "GEMINI_API_KEY not found"

**Solution:**
```bash
# Check .env file exists
cat .env

# Get API key from https://makersuite.google.com/app/apikey
# Update .env with your key
```

### Issue: CUDA out of memory

**Solution:**
```python
# In backend/rag_engine.py, use CPU:
self.device = torch.device("cpu")

# Or reduce batch size in idd_data_prep.py:
BATCH_SIZE = 8
```

### Issue: Slow inference (>3s per frame)

**Solution:**
- Check GPU is being used: `nvidia-smi`
- Use smaller DINOv2 model: `dinov2-small` (already set)
- Reduce resolution in `_encode_image_for_gemini()`

---

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for other LLMs (OpenAI, Llama)
- [ ] Multi-language support for warnings
- [ ] Audio output integration
- [ ] Mobile app version
- [ ] More dataset coverage (other countries)

---

## Limitations & Future Work

### Current Limitations

1. **Requires pre-indexed dataset** — FAISS index must be pre-built
2. **Cloud-dependent** — Gemini API requires internet connection
3. **Single video processing** — No batch processing yet
4. **Indian-specific** — Designed for Indian traffic patterns
5. **Dashcam-only** — Expects vehicle-mounted camera perspective

### Roadmap

- [ ] Real-time camera feed support (live webcam)
- [ ] Offline mode with quantized local LLM
- [ ] Multi-modal input (lidar, GPS)
- [ ] Continuous dataset updates from user submissions
- [ ] Mobile app with audio-only interface
- [ ] Fine-tuned models on safety-critical scenarios

---

## License

MIT License — See LICENSE file

---

## Contact & Support

- **GitHub Issues:** https://github.com/sparshdubey95/Blind-Sight-RAG/issues
- **Email:** sparsh.dubey@example.com

---

## Acknowledgments

- **IDD Dataset Team** for the Indian Driving Dataset
- **Meta Research** for DINOv2
- **Facebook Research** for FAISS
- **Google** for Gemini 1.5 Flash API
- Accessibility community for feedback and use cases

---

**Built with ❤️ for accessibility | Tested on RTX 3050 + Windows 11**