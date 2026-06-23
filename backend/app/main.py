"""
app/main.py
===========
FastAPI entry point — HTTP layer only.

All analysis logic lives in pipeline.py. This file handles:
  - File upload / temp file lifecycle
  - FAISS integrity self-check
  - Response assembly

Current pipeline (see pipeline.py for the full breakdown)
-----------------------------------------------------------
Stage 1 — text acquisition: plain parsing (PyMuPDF / EasyOCR). No AI matching.
Stage 2 — skills + certs discovery: candidate phrases are semantically matched
           against the job corpus allowlist (SBERT cosine similarity, >=0.80)
           inside core.skills / core.certs.
Stage 3 — job matching: discovered skills AND certs are compared against
           every job's requirements via SBERT cosine similarity (core.scoring,
           core.validation). No string-fuzzy matching (rapidfuzz) is used
           anywhere in this pipeline anymore — it's all embedding-based.
Stage 4 — FAISS integrity self-check (this file): a sanity check that the
           FAISS index round-trips the resume's own embedding correctly. Not
           part of the analysis itself.
"""

import os
import shutil
import logging

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

# Ensure .env is loaded before config values are read
load_dotenv()

# OpenMP guard — must be before any torch/faiss import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config        import HOST, PORT, HF_MODEL
from pipeline      import run_pipeline

log = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Matching API",
    description=(
        "Pipeline: PyMuPDF / EasyOCR (text) -> Llama-3.1-8B (HF Inference API, "
        "fit-reason generation only) -> SBERT semantic matching (skills + certs "
        "discovery AND job matching) -> FAISS (similarity search) -> Firestore"
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
            "matching":                  "SBERT semantic (cosine >= 0.80)",
            "skills_section_rule_based": result["skills_section_debug"],
            "discovered_skills":         result["discovered_skills"],
        },
        "STAGE_2_CERT_EXTRACTION": {
            "llm_model":               HF_MODEL,
            "matching":                "SBERT semantic (cosine >= 0.80)",
            "cert_section_rule_based": result["cert_section_debug"],
            "discovered_certificates": result["discovered_certs"],
        },
        "STAGE_3_JOB_MATCHING": {
            "matching_method":     "SBERT semantic cosine similarity (skills and certs)",
            "ground_truth_source": "best_matched_job_required_skills_and_certifications",
            "skills": {
                "ground_truth_count": len(best["required_skills"]),
                "nlp_accuracy":       result["nlp_accuracy"],
                "matched_skills":     result["nlp_verification"]["matched_skills"],
                "unmatched_skills":   result["nlp_verification"]["unmatched_skills"],
            },
            "certifications": {
                "ground_truth_count": len(best["required_certifications"]),
                "nlp_accuracy":       result["nlp_accuracy_certs"],
                "matched_certs":      result["matched_certs"],
                "missing_certs":      result["missing_certs"],
            },
            "Top_5_Career_Recommendations_NLP": [
                {
                    "rank":                    rank + 1,
                    "job_id":                  m["job_id"],
                    "title":                   m["title"],
                    "industry":                m.get("industry", ""),
                    "nlp_accuracy":            m["nlp_validation"]["NLP_ACCURACY"],
                    "matched_skills":          m["nlp_validation"]["matched_skills"],
                    "unmatched_skills":        m["nlp_validation"]["unmatched_skills"],
                    "ground_truth_count":      len(m["required_skills"]),
                    "cert_nlp_accuracy":       m["nlp_validation_certs"]["NLP_ACCURACY"],
                    "matched_certs":           m["matched_certs"],
                    "missing_certs":           m["missing_certs"],
                    "cert_ground_truth_count": len(m["required_certifications"]),
                    "fit_reason":              m.get("fit_reason", ""),
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
      - which of the job's required skills AND certifications the candidate
        already has, vs. what's missing
    No top-5 ranking is produced — only the one targeted job is evaluated.

    Skill and cert matching here uses the same SBERT semantic matching
    (core.validation.fuzzy_skill_verify) as the main scoring pipeline —
    this used to be a separate exact-string comparison and has been brought
    in line so a candidate isn't penalised here for wording a skill/cert
    differently than the job posting does.
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


    # ── 2. Find the target job in the full corpus ─────────────────────────────
    # We search the FULL jobs corpus by title — NOT top_5_matches.
    # top_5_matches is score-gated to TOP_K (default 5). A job the user
    # aspires to but currently scores poorly on would never appear there.
    # The point of this endpoint is to show the gap regardless of current
    # score, so we resolve the title directly from the corpus.
    from db               import load_analyzer_data as _load
    from core.validation  import fuzzy_skill_verify

    _, jobs_corpus_full, _ = _load()

    target_lower = target_job.strip().lower()

    # Exact title match first, then substring fallback.
    exact     = [j for j in jobs_corpus_full if j["title"].lower() == target_lower]
    substring = [j for j in jobs_corpus_full if target_lower in j["title"].lower()]
    candidates = exact or substring

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No job matching '{target_job}' was found in the corpus. "
                "Check the title and try again."
            ),
        )

    # Among all title matches, pick the one with the most required skills
    # so the gap analysis uses the richest ground truth available.
    target_job_doc: dict = max(candidates, key=lambda j: len(j.get("required_skills", [])))

    # ── 3. Semantic skill validation against the target job ───────────────────
    req_skills = target_job_doc.get("required_skills", [])
    nlp = fuzzy_skill_verify(
        discovered_list   = result["discovered_skills"],
        ground_truth_list = req_skills,
    )
    match_ratio  = round(len(nlp["matched_skills"]) / len(req_skills), 4) if req_skills else 0.0
    # Hybrid score without SBERT job-similarity re-encoding (we don't re-run
    # the job-corpus cosine pass for an arbitrary target here) — match_ratio
    # only, but match_ratio itself is now semantic, not exact-string.
    hybrid_score = match_ratio

    # ── 4. Semantic cert validation against the target job ────────────────────
    req_certs = target_job_doc.get("required_certifications", [])
    cert_nlp  = fuzzy_skill_verify(
        discovered_list   = result["discovered_certs"],
        ground_truth_list = req_certs,
    )
    matched_certs = cert_nlp["matched_skills"]
    missing_certs = cert_nlp["unmatched_skills"]

    # ── 5. Assemble focused response ──────────────────────────────────────────
    return {
        "target_job": {
            "job_id":   target_job_doc.get("id", ""),
            "title":    target_job_doc.get("title", ""),
            "industry": target_job_doc.get("industry", ""),
        },
        "resume_skills": {
            "discovered_skills":  result["discovered_skills"],
            "discovered_certs":   result["discovered_certs"],
            "total_skills_found": len(result["discovered_skills"]),
            "total_certs_found":  len(result["discovered_certs"]),
        },
        "skill_gap_analysis": {
            "required_skills": req_skills,
            "matched_skills":  nlp["matched_skills"],
            "missing_skills":  nlp["unmatched_skills"],
            "nlp_accuracy":    nlp["NLP_ACCURACY"],
            "match_ratio":     match_ratio,
            "hybrid_score":    hybrid_score,
        },
        "cert_gap_analysis": {
            "required_certs": req_certs,
            "matched_certs":  matched_certs,
            "missing_certs":  missing_certs,
            "nlp_accuracy":   cert_nlp["NLP_ACCURACY"],
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