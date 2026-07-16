"""
app/main.py
===========
FastAPI entry point — HTTP layer only. All analysis logic lives in
pipeline.py; this file handles file upload, the FAISS integrity check,
and response assembly.
"""

import os
import shutil
import logging
import threading

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Must be set before any torch/faiss import
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

# Allow the deployed web frontend (and localhost during development) to call
# this API from the browser. Without this, the browser blocks every response
# with a CORS error even though the request succeeds on the server.
_default_origins = [
    "https://skillmatch-ai-frontend.onrender.com",
    "http://localhost:5000",
    "http://localhost:8080",
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = _default_origins + [o.strip() for o in _env_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 384 dims for all-MiniLM-L6-v2
_integrity_index = faiss.IndexFlatL2(384)

# Now that endpoints run in a threadpool (see `def` not `async def` below),
# two requests CAN run at the same time instead of one fully blocking the
# other. That's good for avoiding timeouts, but on a 512MB instance running
# two full OCR/SBERT/inference pipelines simultaneously risks an out-of-memory
# crash. This semaphore caps how many pipeline runs execute at once — extra
# requests wait briefly (without freezing the server) instead of piling on
# memory pressure. Raise this only if you move to a bigger Render plan.
_pipeline_semaphore = threading.Semaphore(1)


@app.post("/thesis-full-analysis")
def thesis_full_analysis(file: UploadFile = File(...)) -> dict:
    """Full pipeline analysis for a single uploaded resume PDF."""

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        with _pipeline_semaphore:
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

    # Sanity check: does FAISS round-trip this resume's own embedding correctly?
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
def target_career_analysis(
        file:       UploadFile = File(...),
        target_job: str        = Form(...),
) -> dict:
    """
    Targeted resume analysis against a specific job title.

    Runs the standard extraction pipeline, then looks up `target_job` in
    the Firestore corpus and returns a focused gap report (matched vs.
    missing skills/certs) for that one job — no top-5 ranking.

    Uses the same SBERT semantic matching as the main scoring pipeline,
    so wording differences between resume and job posting don't count
    against the candidate.
    """

    temp_path = f"temp_target_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        with _pipeline_semaphore:
            result = run_pipeline(temp_path, file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[API] Target career pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Search the full corpus by title (not top_5_matches, which is score-gated
    # and would hide an aspirational job the candidate scores poorly on).
    from db               import load_analyzer_data as _load
    from core.validation  import semantic_verify

    _, jobs_corpus_full, _ = _load()

    target_lower = target_job.strip().lower()

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

    # Pick the richest match (most required skills) as ground truth
    target_job_doc: dict = max(candidates, key=lambda j: len(j.get("required_skills", [])))

    # Skill gap
    req_skills = target_job_doc.get("required_skills", [])
    nlp = fuzzy_skill_verify(
        discovered_list   = result["discovered_skills"],
        ground_truth_list = req_skills,
    )
    match_ratio  = round(len(nlp["matched_skills"]) / len(req_skills), 4) if req_skills else 0.0
    hybrid_score = match_ratio  # no job-corpus re-encoding pass for arbitrary targets

    # Cert gap
    req_certs = target_job_doc.get("required_certifications", [])
    cert_nlp  = fuzzy_skill_verify(
        discovered_list   = result["discovered_certs"],
        ground_truth_list = req_certs,
    )
    matched_certs = cert_nlp["matched_skills"]
    missing_certs = cert_nlp["unmatched_skills"]

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