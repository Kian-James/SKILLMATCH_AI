"""
app/main.py
===========
FastAPI entry point — HTTP layer only.

All analysis logic lives in pipeline.py. This file handles:
  - File upload / temp file lifecycle
  - FAISS integrity self-check
  - Response assembly
"""

import os
import shutil
import logging

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from dotenv import load_dotenv

# Ensure .env is loaded before config values are read
load_dotenv()

# OpenMP guard — must be before any torch/faiss import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config   import HOST, PORT, HF_MODEL
from pipeline import run_pipeline

log = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Matching API",
    description=(
        "Pipeline: PyMuPDF → EasyOCR → "
        "Llama-3.1-8B (HF Inference API) → SBERT → FAISS → Firestore"
    ),
)

# FAISS integrity index — 384 dims (all-MiniLM-L6-v2)
_integrity_index = faiss.IndexFlatL2(384)


@app.post("/thesis-full-analysis")
async def thesis_full_analysis(file: UploadFile = File(...)) -> dict:
    """Full pipeline analysis for a single uploaded resume PDF."""

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        result = run_pipeline(temp_path, file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[API] Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not result["top_5_matches"]:
        raise HTTPException(status_code=500, detail="Scoring returned no results.")

    # FAISS integrity self-check
    from core.ocr_correction import get_sbert
    _sbert     = get_sbert()
    resume_vec = (
        _sbert.encode(result["resume_text"], convert_to_tensor=False)
        .reshape(1, -1)
        .astype("float32")
    )
    pre_count = _integrity_index.ntotal
    _integrity_index.add(resume_vec)
    D, I = _integrity_index.search(resume_vec, 1)
    faiss_integrity = 100.0 if I[0][0] == pre_count and D[0][0] < 0.0001 else 0.0

    best      = result["best"]
    secondary = result["secondary"]

    return {
        "STAGE_1_TEXT_ACQUISITION": {
            "text_source": result["text_source"],
            "ocr_quality": f"{result['ocr_quality']}%",
            "char_count":  len(result["resume_text"]),
            "raw_text":    result["resume_text"],
        },
        "STAGE_2_SKILL_EXTRACTION": {
            "llm_model":                 HF_MODEL,
            "skills_section_rule_based": result["skills_section_debug"],
            "discovered_skills":         result["discovered_skills"],
        },
        "STAGE_2B_CERT_EXTRACTION": {
            "llm_model":               HF_MODEL,
            "cert_section_rule_based": result["cert_section_debug"],
            "discovered_certificates": result["discovered_certs"],
        },
        "STAGE_3_NLP_VALIDATION": {
            "ground_truth_source": "best_matched_job_required_skills",
            "ground_truth_count":  len(best["required_skills"]),
            "nlp_accuracy":        result["nlp_accuracy"],
            "matched_skills":      result["nlp_verification"]["matched_skills"],
            "unmatched_skills":    result["nlp_verification"]["unmatched_skills"],
            "Top_5_Career_Recommendations_NLP": [
                {
                    "rank":               rank + 1,
                    "job_id":             m["job_id"],
                    "title":              m["title"],
                    "nlp_accuracy":       m["nlp_validation"]["NLP_ACCURACY"],
                    "matched_skills":     m["nlp_validation"]["matched_skills"],
                    "unmatched_skills":   m["nlp_validation"]["unmatched_skills"],
                    "ground_truth_count": len(m["required_skills"]),
                }
                for rank, m in enumerate(result["top_5_matches"])
            ],
        },
        "STAGE_4_FAISS_INTEGRITY": {
            "score":       f"{faiss_integrity}%",
            "index_total": _integrity_index.ntotal,
        },
    }


@app.post("/target-career-analysis")
async def target_career_analysis(
        file:       UploadFile = File(...),
        target_job: str        = Form(...),
) -> dict:
    """
    Targeted resume analysis against a specific job title.

    The caller supplies:
      - file        — the resume PDF (same as /thesis-full-analysis)
      - target_job  — the exact or approximate job title to match against
                      (e.g. "Data Scientist", "Software Engineer")

    The endpoint runs the standard extraction pipeline, then looks up the
    requested job in the Firestore corpus and returns a focused report:
      - which of the job's required skills the candidate already has
      - which skills are still missing
    No top-5 ranking is produced — only the one targeted job is evaluated.
    """

    # ── 1. Save upload to a temp file ────────────────────────────────────────
    temp_path = f"temp_target_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        result = run_pipeline(temp_path, file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[API] Target career pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # ── 2. Find the target job in the corpus ─────────────────────────────────
    # run_pipeline already loaded the corpus; re-use top_5_matches which
    # contains every scored job (pipeline scores all jobs, not just 5).
    # We match by title using case-insensitive substring search so that
    # "data scientist" finds "Senior Data Scientist" etc.
    all_matches: list[dict] = result["top_5_matches"]  # full ranked list

    target_lower = target_job.strip().lower()

    # Prefer an exact title match first, then fall back to substring.
    exact     = [m for m in all_matches if m["title"].lower() == target_lower]
    substring = [m for m in all_matches if target_lower in m["title"].lower()]
    candidates = exact or substring

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No job matching '{target_job}' was found in the corpus. "
                "Check the title and try again."
            ),
        )

    # Pick the highest-scoring candidate among those that match the title.
    target_match: dict = max(candidates, key=lambda m: m["score"])

    # ── 3. NLP validation against the target job ──────────────────────────────
    from core.validation import fuzzy_skill_verify
    from core.utils      import normalize

    nlp = fuzzy_skill_verify(
        discovered_list   = result["discovered_skills"],
        ground_truth_list = target_match["required_skills"],
    )

    # ── 4. Cert gap for the target job ────────────────────────────────────────
    certs_norm    = {normalize(c) for c in result["discovered_certs"]}
    req_certs     = target_match.get("required_certs", [])   # may be empty
    matched_certs = [c for c in req_certs if normalize(c) in certs_norm]
    missing_certs = [c for c in req_certs if normalize(c) not in certs_norm]

    # ── 5. Assemble focused response ──────────────────────────────────────────
    return {
        "target_job": {
            "job_id":   target_match["job_id"],
            "title":    target_match["title"],
            "industry": target_match.get("industry", ""),
        },
        "resume_skills": {
            "discovered_skills":      result["discovered_skills"],
            "discovered_certs":       result["discovered_certs"],
            "total_skills_found":     len(result["discovered_skills"]),
            "total_certs_found":      len(result["discovered_certs"]),
        },
        "skill_gap_analysis": {
            "required_skills":  target_match["required_skills"],
            "matched_skills":   nlp["matched_skills"],
            "missing_skills":   nlp["unmatched_skills"],
            "nlp_accuracy":     nlp["NLP_ACCURACY"],
            "match_ratio":      target_match["match_ratio"],
            "hybrid_score":     target_match["score"],
        },
        "cert_gap_analysis": {
            "required_certs": req_certs,
            "matched_certs":  matched_certs,
            "missing_certs":  missing_certs,
        },
        "text_acquisition": {
            "text_source": result["text_source"],
            "ocr_quality": f"{result['ocr_quality']}%",
            "char_count":  len(result["resume_text"]),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)