"""
config.py - Application configuration
Loads all secrets from environment variables / .env file
"""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Firebase
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    "../firebase_service_account.json"
)
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "your-project-id")

# HuggingFace
# Get your token at: https://huggingface.co/settings/tokens
# Permission needed: "Inference > Make calls to the serverless Inference API"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# LLM for dataset generation and skill resource suggestions.
# Meta-Llama-3-8B-Instruct is available on the HuggingFace free tier
# via huggingface_hub InferenceClient (hf-inference provider).
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# RoBERTa for embeddings (sentence-transformers wrapping)
ROBERTA_MODEL_ID = "sentence-transformers/all-roberta-large-v1"

# FAISS
FAISS_INDEX_PATH = Path("faiss_index/job_embeddings.index")
FAISS_METADATA_PATH = Path("faiss_index/job_metadata.json")

# OCR
# Tesseract binary path (Linux default; adjust for Windows/Mac)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# Embedding Cache
CACHE_DIR = Path("embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

# API Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Firestore Collections
COLLECTION_USERS = "users"
COLLECTION_JOB_DATASET = "job_dataset"
COLLECTION_USER_PROGRESS = "user_progress"
# Note: Resumes are NOT stored - only extracted metadata (skills/certs)