"""
dataset_generator/run_build_index.py

Rebuilds the FAISS index from Firestore data.

Run from the dataset_generator directory:
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python3.11 run_build_index.py
"""

import asyncio
import sys
from pathlib import Path

# Point imports to the backend package
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Ensure the faiss_index directory exists BEFORE importing FAISSIndex,
# because the module resolves FAISS_INDEX_PATH at import time and
# faiss.write_index() will crash if the parent folder is missing.
FAISS_DIR = BACKEND_DIR / "faiss_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)
print(f"faiss_index dir ready: {FAISS_DIR}")

from services.firebase_service import init_firebase
from ai_modules.faiss_index import FAISSIndex


async def main() -> None:
    init_firebase()
    idx = FAISSIndex()
    await idx.rebuild()
    print("Done! Index saved to backend/faiss_index/")


asyncio.run(main())