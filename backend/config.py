"""
config.py - Application configuration
"""
import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Firebase
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "../firebase_service_account.json")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "your-project-id")

def get_firebase_credentials_path() -> str:
    if FIREBASE_SERVICE_ACCOUNT_JSON:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(FIREBASE_SERVICE_ACCOUNT_JSON)
        tmp.flush()
        tmp.close()
        return tmp.name
    return FIREBASE_SERVICE_ACCOUNT_PATH

# HuggingFace token — used for BOTH:
#   1. RoBERTa embeddings (sentence-transformers)
#   2. LLM inference via HuggingFace's unified router
# Get yours at: https://huggingface.co/settings/tokens
# Make sure the token has "Make calls to Inference Providers" permission enabled.
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# LLM: Llama 3.1 8B Instruct via HuggingFace's unified inference router
# - No third-party provider account needed (Featherless, etc.)
# - Uses your monthly free HF credits automatically
# - Router picks fastest available backend (Cerebras, Sambanova, etc.)
# - OpenAI-compatible endpoint
LLM_MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
LLM_BASE_URL  = "https://router.huggingface.co/v1"
LLM_PROVIDER  = ""  # No longer used — kept to avoid import errors elsewhere

# Embedding model — small, fits Render free tier 512MB RAM
ROBERTA_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS
FAISS_INDEX_PATH    = Path("faiss_index/job_embeddings.index")
FAISS_METADATA_PATH = Path("faiss_index/job_metadata.json")

# OCR
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# Embedding Cache
CACHE_DIR = Path("embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

# API Server
API_HOST   = os.getenv("API_HOST", "0.0.0.0")
API_PORT   = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# spaCy
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Firestore Collections
COLLECTION_USERS         = "users"
COLLECTION_JOB_DATASET   = "jobs"
COLLECTION_USER_PROGRESS = "user_progress"