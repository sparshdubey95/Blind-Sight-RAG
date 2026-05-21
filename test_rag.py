"""
Blind-Sight RAG — RAG Engine Test Script
========================================
Diagnostic script to test RAG Engine initialization and basic functionality.

Usage:
    python test_rag.py

Expected output on success:
    [RAG Engine] Initializing...
    ✓ Index loaded (600 vectors)
    ✓ Database connected
    ✓ DINOv2 ready on device
    ✓ Gemini API ready
    Success

If this script fails, check:
    1. .env file exists with GEMINI_API_KEY
    2. hazards.index and metadata.db exist in project root
    3. Internet connection (for downloading models)
    4. GPU driver (nvidia-smi should work for CUDA)
"""

import traceback
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_rag_engine() -> None:
    """Test RAG Engine initialization."""
    gemini_key = os.getenv('GEMINI_API_KEY')
    print(f'Gemini API key configured: {bool(gemini_key)}')
    
    if not gemini_key:
        print('⚠ WARNING: GEMINI_API_KEY not found in .env file')
        print('  Get a free API key at: https://makersuite.google.com/app/apikey')
    
    try:
        from backend.rag_engine import RAGEngine
        
        print("\nInitializing RAG Engine...")
        r = RAGEngine(gemini_key or "mock_key")
        print('✓ RAG Engine initialized successfully')
        
        # Test embedding a dummy frame (optional)
        print("\nTesting frame embedding...")
        import numpy as np
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = r.embed_frame(dummy_frame)
        print(f'✓ Frame embedding successful: {embedding.shape}')
        
        # Cleanup
        r.close()
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f'\n✗ Error during RAG Engine test:')
        traceback.print_exc()
        print(f'\nTroubleshooting:')
        print(f'  1. Check that hazards.index and metadata.db exist')
        print(f'  2. Run: python idd_data_prep.py')
        print(f'  3. Ensure .env file has GEMINI_API_KEY set')

if __name__ == "__main__":
    test_rag_engine()