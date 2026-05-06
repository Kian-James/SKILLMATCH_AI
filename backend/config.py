"""
config.py
=========
Centralised configuration for the AI Resume Analyzer.

All os.getenv() calls live here. Every other module imports from this file
instead of calling os.getenv() directly.

The .env file is loaded here via python-dotenv so importing this module in
any entry-point is sufficient — no need to call load_dotenv() elsewhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the directory containing this file)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# ── HuggingFace Inference API ─────────────────────────────────────────────────
HF_TOKEN: str    = os.getenv("HF_TOKEN", "")
HF_API_URL: str  = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL: str    = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# ── Firebase / Firestore ──────────────────────────────────────────────────────
FIREBASE_CREDENTIALS_PATH: str    = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase_service_account.json")
FIRESTORE_JOBS_COLLECTION: str    = os.getenv("FIRESTORE_JOBS_COLLECTION",    "jobs")
FIRESTORE_RESULTS_COLLECTION: str = os.getenv("FIRESTORE_RESULTS_COLLECTION", "resume_results")

# ── SBERT / FAISS ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K: int           = int(os.getenv("TOP_K", "5"))

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_LANGUAGES: list[str] = os.getenv("EASYOCR_LANGUAGES", "en").split(",")
MIN_NATIVE_CHARS: int    = int(os.getenv("MIN_NATIVE_CHARS", "50"))

# ── OCR correction thresholds ─────────────────────────────────────────────────
OCR_EMBED_THRESHOLD: float = float(os.getenv("OCR_EMBED_THRESHOLD", "0.78"))
BLOB_FUZZY_THRESHOLD: int  = int(os.getenv("BLOB_FUZZY_THRESHOLD", "78"))

# ── FastAPI server ────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))