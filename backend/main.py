"""
main.py
───────
FastAPI backend for AI-Based Resume Matching.
Job data is fetched live from Firestore instead of a local JSON file.

Environment variables (loaded from .env):
    FIREBASE_CREDENTIALS_PATH  – path to your serviceAccountKey.json
    FIRESTORE_JOBS_COLLECTION  – Firestore collection name (default: jobs)
    HOST                       – bind host  (default: 0.0.0.0)
    PORT                       – bind port  (default: 8000)
"""

import os
import shutil

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIRESTORE_JOBS_COLLECTION  = os.getenv("FIRESTORE_JOBS_COLLECTION", "jobs")
HOST                       = os.getenv("HOST", "0.0.0.0")
PORT                       = int(os.getenv("PORT", "8000"))

# ── Firebase / Firestore init ────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore as fs

if not firebase_admin._apps:
    if not FIREBASE_CREDENTIALS_PATH:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_PATH is not set. "
            "Add it to your .env file and restart."
        )
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

db = fs.client()

# ── Matcher helpers ──────────────────────────────────────────────────────────
from matcher import (
    extract_text,
    extract_skills_from_corpus,
    calculate_cosine_similarity,
    identify_skill_gap,
    model,
)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="AI Resume Matching API")

# FAISS index – 384 dims for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(384)


# ── Firestore helper ─────────────────────────────────────────────────────────

def fetch_jobs_from_firestore() -> list:
    """Retrieve all job documents from Firestore, ordered by their numeric 'id' field."""
    docs = db.collection(FIRESTORE_JOBS_COLLECTION).stream()
    jobs = [doc.to_dict() for doc in docs]

    if not jobs:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No jobs found in Firestore collection '{FIRESTORE_JOBS_COLLECTION}'. "
                "Run generate_jobs.py first to seed the collection."
            ),
        )

    jobs.sort(key=lambda j: j.get("id", 0))
    return jobs


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/thesis-full-analysis")
async def thesis_full_analysis(
        file: UploadFile = File(...),
):
    # ── Fetch jobs from Firestore ────────────────────────────────────────────
    jobs_corpus = fetch_jobs_from_firestore()

    # ── Save upload to a temp file ───────────────────────────────────────────
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # ── STAGE 1: OCR ────────────────────────────────────────────────────
        raw_text, ocr_quality = extract_text(temp_path)

        # ── STAGE 2: NLP skill extraction from corpus ────────────────────────
        verified_skills = extract_skills_from_corpus(raw_text, jobs_corpus)

        # ── STAGE 3: FAISS vector integrity ─────────────────────────────────
        resume_vec = model.encode(raw_text, convert_to_tensor=True)
        vec_np     = resume_vec.cpu().detach().numpy().reshape(1, -1).astype("float32")

        pre_count = index.ntotal
        index.add(vec_np)

        D, I = index.search(vec_np, 1)
        faiss_integrity = (
            100.0 if I[0][0] == pre_count and D[0][0] < 0.0001 else 0.0
        )

        # ── Matching & career guidance ───────────────────────────────────────
        top_5_matches = calculate_cosine_similarity(raw_text, verified_skills, jobs_corpus)

        best      = top_5_matches[0]
        secondary = top_5_matches[1] if len(top_5_matches) > 1 else best

        return {
            "PIPELINE_ACCURACY_REPORT": {
                "STAGE_1_OCR_QUALITY":     f"{ocr_quality}%",
                "STAGE_2_NLP_ACCURACY":    "Corpus-matched (Firestore skills)",
                "STAGE_3_FAISS_INTEGRITY": f"{faiss_integrity}%",
            },
            "CAREER_GUIDANCE_SYSTEM": {
                "JOB_RECOMMENDATIONS": top_5_matches,
                "BEST_MATCH_SUMMARY": {
                    "job_title":        best["title"],
                    "confidence_score": best["score"],
                    "gap_count":        best["gap_count"],
                    "required_skills":  best["required_skills"],
                    "matched_skills":   best["matched_skills"],
                    "missing_skills":   best["missing_skills"],
                },
                "SKILL_RECOMMENDATIONS": {
                    "target_job":        secondary["title"],
                    "skill(s)_to_learn": secondary["missing_skills"],
                },
            },
            "EXTRACTION_SUMMARY": {
                "file_name":          file.filename,
                "extracted_skills":   verified_skills,
                "total_skills_found": len(verified_skills),
            },
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Dev runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)