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

def apply_homography_crop(image: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform that:
    - Crops the vertical center 60% (removes sky and dashboard)
    - Shifts focus toward the left edge (pedestrian perspective)
    - Outputs a 224×224 image ready for DINOv2

    Args:
        image (np.ndarray): BGR image from OpenCV (H, W, 3).

    Returns:
        np.ndarray: Cropped and warped 224×224 BGR image.
    """
    h, w = image.shape[:2]

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
        [224, 0],
        [224, 224],
        [0,        224],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (224, 224))
    return warped

def extract_dataset(dataset_tar: Path, dataset_dir: Path) -> None:
    """
    Extract the IDD-Lite tar.gz archive if not already extracted.
    
    Checks if the dataset directory already exists and contains files.
    If not, extracts the tar.gz file from the datasets/ folder.
    
    Args:
        dataset_tar (Path): Path to the tar.gz archive
        dataset_dir (Path): Target extraction directory
        
    Raises:
        SystemExit: If the tar.gz archive is not found in datasets/
    """
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print("  ✓ Dataset already extracted")
        return

    if not dataset_tar.exists():
        print(f"  ✗ Archive not found: {dataset_tar}")
        print("    Please download idd-lite.tar.gz into datasets/")
        sys.exit(1)

    print(f"  Extracting {dataset_tar.name}...")
    with tarfile.open(dataset_tar, "r:gz") as tar:
        tar.extractall(path=dataset_tar.parent)
    print("  ✓ Extraction complete")

def collect_images(dataset_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Recursively collect all images from the IDD-Lite dataset.
    
    Scans the leftImg8bit folder (train/val/test splits) and collects paths
    to all supported image files (.jpg, .jpeg, .png).
    
    Returns:
        List[Tuple[str, str, str]]: List of tuples containing:
            - image path (absolute path to .jpg/.png file)
            - split (train/val/test)
            - sequence_id (folder name, identifies scene sequence)
    """
    results: List[Tuple[str, str, str]] = []

    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
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

def load_dinov2(device: torch.device, dinov2_model: str):
    """
    Load pre-trained DINOv2 ViT-Small model and image processor.
    
    Downloads (if not cached) and loads the Facebook Research DINOv2 model.
    The model is set to evaluation mode for inference.
    
    Args:
        device (torch.device): Target device (cuda or cpu)
        dinov2_model (str): Model identifier (e.g., 'facebook/dinov2-small')
        
    Returns:
        Tuple[AutoImageProcessor, AutoModel]: Processor and model instances
    """
    print(f"  Loading {dinov2_model} on {device}...")
    processor = AutoImageProcessor.from_pretrained(dinov2_model)
    model = AutoModel.from_pretrained(dinov2_model).to(device)
    model.eval()
    print(f"  ✓ DINOv2 loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return processor, model