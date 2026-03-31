"""
config.py - Application configuration
Loads all secrets from environment variables / .env file

On Render: Firebase credentials are stored as a JSON string in the
FIREBASE_SERVICE_ACCOUNT_JSON env var (not a file path).
Locally: Use FIREBASE_SERVICE_ACCOUNT_PATH pointing to the JSON file.
"""

import json
import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Firebase
# On Render: set FIREBASE_SERVICE_ACCOUNT_JSON to the full JSON string
# Locally:   set FIREBASE_SERVICE_ACCOUNT_PATH to the file path
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    "firebase_service_account.json"
)
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "your-project-id")


def get_firebase_credentials_path() -> str:
    """
    Returns a file path to the Firebase service account JSON.
    On Render: writes the JSON env var to a temp file and returns that path.
    Locally: returns FIREBASE_SERVICE_ACCOUNT_PATH directly.
    """
    if FIREBASE_SERVICE_ACCOUNT_JSON:
        # Running on Render - write JSON string to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        tmp.write(FIREBASE_SERVICE_ACCOUNT_JSON)
        tmp.flush()
        tmp.close()
        return tmp.name
    return FIREBASE_SERVICE_ACCOUNT_PATH


# HuggingFace
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# LLM - Meta Llama 3 8B via HuggingFace InferenceClient (free tier)
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# spaCy model - use sm on Render free tier (lg exceeds RAM limit)
# Set SPACY_MODEL=en_core_web_lg locally if you want the larger model
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Embedding model
# all-MiniLM-L6-v2: 80MB, fast, fits Render free tier (512MB RAM)
# all-roberta-large-v1: 1.3GB - too large for free tier, use locally if needed
ROBERTA_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS
FAISS_INDEX_PATH = Path("faiss_index/job_embeddings.index")
FAISS_METADATA_PATH = Path("faiss_index/job_metadata.json")

# OCR
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# Embedding Cache
CACHE_DIR = Path("embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

# API Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))  # Render sets PORT
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Firestore Collections
COLLECTION_USERS = "users"
COLLECTION_JOB_DATASET = "job_dataset"
COLLECTION_USER_PROGRESS = "user_progress"