"""
Blind-Sight RAG — IDD-Lite Data Preprocessing & FAISS Index Builder
====================================================================
Standalone CLI script that:

1. Extracts the IDD-Lite `.tar.gz` if not already extracted.
2. Collects all images from leftImg8bit/ (train/val/test).
3. Applies an OpenCV homography crop: center 60%, left-edge focus.
4. Embeds each cropped image using DINOv2 ViT-Small (384-d, CUDA).
5. Builds a FAISS IndexFlatIP on L2-normalised vectors → `hazards.index`.
6. Creates an SQLite database (`metadata.db`) mapping vector IDs to hazard
   descriptions drawn from a curated Indian-traffic vocabulary.

Usage:
    python idd_data_prep.py

Outputs (project root):
    hazards.index    FAISS binary index
    metadata.db      SQLite metadata (vector_id → description, image path)
"""

from __future__ import annotations

import os
import random
import sqlite3
import sys
import tarfile
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_TAR = PROJECT_ROOT / "datasets" / "idd-lite.tar.gz"
DATASET_DIR = PROJECT_ROOT / "datasets" / "idd20k_lite" / "leftImg8bit"
INDEX_FILE = PROJECT_ROOT / "hazards.index"
DB_FILE = PROJECT_ROOT / "metadata.db"

DINOV2_MODEL = "facebook/dinov2-small"
VECTOR_DIM = 384   # DINOv2 ViT-Small output dimension
BATCH_SIZE = 16     # Conservative for RTX 3050 4GB VRAM
IMG_SIZE = 224

# ── Hazard description vocabulary ─────────────────────────────────────────────
# Curated descriptions that map to typical Indian driving scenarios found in IDD.
# These are randomly assigned to images based on sequence characteristics.

HAZARD_DESCRIPTIONS = [
    "Dense chaotic traffic at uncontrolled intersection with mixed vehicle types",
    "Narrow lane with pedestrians walking on road edge and parked vehicles",
    "Uneven road surface with potholes and loose gravel near construction zone",
    "Heavy two-wheeler and auto-rickshaw congestion with minimal lane discipline",
    "Pedestrian crossing unmarked road section with oncoming traffic",
    "Mixed traffic flow with buses, trucks, and bicycles sharing single lane",
    "Street vendor encroachment reducing effective road width significantly",
    "Unpredictable animal crossing — stray dogs and cattle on roadway",
    "Blind curve on narrow road with no mirrors or warning signs",
    "Waterlogged road section with hidden potholes and reduced traction",
    "School zone with children crossing without traffic signals or guards",
    "Dimly lit road segment with poor visibility and no street lighting",
    "Sharp speed breaker with no warning signage on approach",
    "Overloaded commercial vehicle moving slowly in narrow lane",
    "Aggressive lane merging with no signaling at highway on-ramp",
    "Roadside market area with sudden pedestrian movement from blind spots",
    "Construction debris partially blocking driving lane",
    "Dust cloud from unpaved road section reducing forward visibility",
    "Multi-modal intersection with auto-rickshaws, cycles, and handcarts",
    "Residential area exit with children and elderly pedestrians present",
    "Highway median break with illegal U-turns and cross traffic",
    "Flooded underpass with uncertain water depth",
    "Festival or market crowd spilling onto main road",
    "Parked truck blocking sightline at T-junction",
    "Cycle rickshaw moving against traffic flow on wrong side",
    "Stalled vehicle on narrow bridge causing bottleneck",
    "Open drain alongside road with no protective barriers",
    "Freshly paved tar surface with loose stone chips",
    "Electric wires hanging low over road near utility pole",
    "Emergency vehicle approaching from behind in congested traffic",
]

# ── Homography crop ───────────────────────────────────────────────────────────


def apply_homography_crop(image: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform that:
    - Crops the vertical center 60% (removes sky and dashboard)
    - Shifts focus toward the left edge (pedestrian perspective)
    - Outputs a 224×224 image ready for DINOv2

    Args:
        image: BGR image from OpenCV (H, W, 3).

    Returns:
        Cropped and warped 224×224 BGR image.
    """
    h, w = image.shape[:2]

    # Source quadrilateral: left-biased center crop
    # Top-left is pulled inward, top-right is pulled further inward
    # to create a left-leaning perspective
    y_top = int(h * 0.20)    # skip top 20% (sky)
    y_bot = int(h * 0.80)    # skip bottom 20% (dashboard/bonnet)
    x_left = 0               # keep full left edge
    x_right = int(w * 0.75)  # crop right 25% to focus on pedestrian side

    src_pts = np.float32([
        [x_left,  y_top],           # top-left
        [x_right, y_top],           # top-right
        [x_right, y_bot],           # bottom-right
        [x_left,  y_bot],           # bottom-left
    ])

    dst_pts = np.float32([
        [0,        0],
        [IMG_SIZE, 0],
        [IMG_SIZE, IMG_SIZE],
        [0,        IMG_SIZE],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (IMG_SIZE, IMG_SIZE))
    return warped


# ── Dataset extraction ────────────────────────────────────────────────────────


def extract_dataset() -> None:
    """Extract the IDD-Lite tar.gz archive if not already extracted."""
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print("  ✓ Dataset already extracted")
        return

    if not DATASET_TAR.exists():
        print(f"  ✗ Archive not found: {DATASET_TAR}")
        print("    Please download idd-lite.tar.gz into datasets/")
        sys.exit(1)

    print(f"  Extracting {DATASET_TAR.name}...")
    with tarfile.open(DATASET_TAR, "r:gz") as tar:
        tar.extractall(path=DATASET_TAR.parent)
    print("  ✓ Extraction complete")


# ── Image collection ──────────────────────────────────────────────────────────


def collect_images() -> List[Tuple[str, str, str]]:
    """
    Recursively collect all images from the IDD-Lite dataset.

    Returns:
        List of (abs_path, split, sequence_id) tuples.
    """
    results: List[Tuple[str, str, str]] = []

    for split in ("train", "val", "test"):
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_id = seq_dir.name
            for img_file in sorted(seq_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    results.append((str(img_file), split, seq_id))

    return results


# ── DINOv2 embedding ──────────────────────────────────────────────────────────


def load_dinov2(device: torch.device):
    """Load DINOv2 ViT-Small model and processor."""
    print(f"  Loading {DINOV2_MODEL} on {device}...")
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    model = AutoModel.from_pretrained(DINOV2_MODEL).to(device)
    model.eval()
    print(f"  ✓ DINOv2 loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return processor, model


def embed_batch(
    images: List[np.ndarray],
    processor,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Embed a batch of BGR images through DINOv2.

    Returns:
        L2-normalised float32 array of shape (batch, 384).
    """
    # Convert BGR → RGB PIL images
    pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]

    inputs = processor(images=pil_images, return_tensors="pt").to(device)

    # Convert amp.autocast to device_type standard format for torch>=2.0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        outputs = model(**inputs)
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]

    # L2 normalise for cosine similarity via inner product
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    result = embeddings.cpu().numpy().astype(np.float32)

    return result


# ── SQLite metadata ───────────────────────────────────────────────────────────


def create_database(
    image_infos: List[Tuple[str, str, str]],
    descriptions: List[str],
) -> None:
    """Create the metadata SQLite database."""
    if DB_FILE.exists():
        DB_FILE.unlink()

    conn = sqlite3.connect(str(DB_FILE))
    cur = conn.cursor()

    cur.execute(\"""
        CREATE TABLE hazards (
            vector_id   INTEGER PRIMARY KEY,
            image_path  TEXT NOT NULL,
            split       TEXT,
            sequence_id TEXT,
            description TEXT
        )
    \""")

    rows = [
        (i, info[0], info[1], info[2], desc)
        for i, (info, desc) in enumerate(zip(image_infos, descriptions))
    ]
    cur.executemany(
        "INSERT INTO hazards (vector_id, image_path, split, sequence_id, description) VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    conn.commit()
    conn.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  Blind-Sight RAG  |  IDD-Lite Data Preprocessing")
    print("=" * 70)

    # ── Step 1: Extract ───────────────────────────────────────────────────
    print("\n[1/5] Checking dataset extraction...")
    extract_dataset()

    # ── Step 2: Collect images ────────────────────────────────────────────
    print("\n[2/5] Collecting images...")
    if not DATASET_DIR.exists():
        print(f"  ✗ Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    image_infos = collect_images()
    print(f"  ✓ Found {len(image_infos):,} images across {len(set(i[1] for i in image_infos))} splits")

    if not image_infos:
        print("  ✗ No images found. Aborting.")
        sys.exit(1)

    # ── Step 3: Homography crop + DINOv2 embeddings ───────────────────────
    print(f"\n[3/5] Embedding images with DINOv2 (batch_size={BATCH_SIZE})...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_dinov2(device)
    t0 = time.time()

    all_embeddings: List[np.ndarray] = []
    valid_infos: List[Tuple[str, str, str]] = []

    for start in tqdm(range(0, len(image_infos), BATCH_SIZE), desc="  Batches"):
        batch_infos = image_infos[start : start + BATCH_SIZE]
        batch_images = []
        batch_valid_infos = []

        for path, split, seq_id in batch_infos:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                cropped = apply_homography_crop(img)
                batch_images.append(cropped)
                batch_valid_infos.append((path, split, seq_id))
            except Exception as e:
                print(f"\n    ⚠ Skipping {path}: {e}")

        if not batch_images:
            continue

        try:
            embeddings = embed_batch(batch_images, processor, model, device)
            all_embeddings.append(embeddings)
            valid_infos.extend(batch_valid_infos)
        except Exception as e:
            print(f"\n    ⚠ Batch error at index {start}: {e}")
            # Fall back to one-by-one
            for img, info in zip(batch_images, batch_valid_infos):
                try:
                    emb = embed_batch([img], processor, model, device)
                    all_embeddings.append(emb)
                    valid_infos.append(info)
                except Exception as inner_e:
                    print(f"      ⚠ Skipping {info[0]}: {inner_e}")

        # Free VRAM periodically
        if device.type == "cuda":
            torch.cuda.empty_cache()

    feature_matrix = np.vstack(all_embeddings).astype(np.float32)
    elapsed = time.time() - t0
    print(f"\n  ✓ Embedded {feature_matrix.shape[0]:,} images in {elapsed:.1f}s")
    print(f"    Vector shape: {feature_matrix.shape}")

    # ── Step 4: Build FAISS index ─────────────────────────────────────────
    print(f"\n[4/5] Building FAISS IndexFlatIP (dim={VECTOR_DIM})...")
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(feature_matrix)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"  ✓ Index saved to: {INDEX_FILE}")
    print(f"    Total vectors: {index.ntotal:,}")

    # ── Step 5: Create SQLite metadata ────────────────────────────────────
    print("\n[5/5] Creating SQLite metadata database...")

    # Assign hazard descriptions — use sequence_id as seed for consistency
    descriptions = []
    for _, _, seq_id in valid_infos:
        # Deterministic-ish assignment based on sequence, but varied per image
        seed = hash(seq_id) % len(HAZARD_DESCRIPTIONS)
        # Pick 1–3 descriptions and combine for richer context
        n_desc = (hash(seq_id + "n") % 2) + 1
        indices = [(seed + i) % len(HAZARD_DESCRIPTIONS) for i in range(n_desc)]
        combined = "; ".join(HAZARD_DESCRIPTIONS[j] for j in indices)
        descriptions.append(combined)

    create_database(valid_infos, descriptions)
    print(f"  ✓ Database saved to: {DB_FILE}")
    print(f"    Total rows: {len(valid_infos):,}")

    # ── Summary ───────────────────────────────────────────────────────────
    index_size_mb = INDEX_FILE.stat().st_size / (1024 * 1024)
    db_size_kb = DB_FILE.stat().st_size / 1024
    print(f"\n{'=' * 70}")
    print(f"  DONE — {index.ntotal:,} vectors indexed")
    print(f"  Index size:    {index_size_mb:.1f} MB")
    print(f"  Database size: {db_size_kb:.0f} KB")
    print(f"  Ready for backend inference.")
    print(f"{'=' * 70}\n")

    # Cleanup GPU memory
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
