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
import uuid
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Must be set before any torch/faiss import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config        import HOST, PORT, HF_MODEL
from pipeline      import run_pipeline, extract_and_discover

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


# ── /refresh-fit-reason request body ────────────────────────────────────────
class FitReasonRefreshRequest(BaseModel):
    """
    Everything generate_fit_reason() needs, EXCEPT a resume file — the
    results screen already has all of this in local state after the user
    promotes/demotes a skill chip, so there's no reason to re-run
    OCR/NLP/FAISS just to reword the insight card.
    """
    title:          str
    industry:       str       = ""
    resume_skills:  list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    # 0-1 scale, same convention as elsewhere in scoring.py. Optional because
    # the results screen doesn't retain the original per-job cosine score —
    # it only affects the rule-based fallback's wording when there are zero
    # matched skills, never the LLM prompt, so 0.0 is a safe default.
    cosine_score:   float     = 0.0


@app.post("/refresh-fit-reason")
def refresh_fit_reason(payload: FitReasonRefreshRequest) -> dict:
    """
    Regenerate the fit_reason narrative from an updated matched/missing
    skill split.

    Without this, fit_reason is written once at scan time and frozen in
    Firestore — if the user later promotes a missing skill to "matched" or
    demotes a hallucinated match, the numeric tiles on the results screen
    recompute live but the AI insight text keeps quoting the original,
    now-stale percentage. This endpoint lets the client ask for a fresh
    fit_reason using the *current* matched/missing lists, so the two stay
    in sync after an edit.
    """
    from core.scoring import generate_fit_reason

    total       = len(payload.matched_skills) + len(payload.missing_skills)
    match_ratio = (len(payload.matched_skills) / total) if total else 0.0

    fit_reason = generate_fit_reason(
        title           = payload.title,
        matched_skills  = payload.matched_skills,
        missing_skills  = payload.missing_skills,
        resume_skills   = payload.resume_skills,
        resume_text     = "",
        job_description = "",
        cosine_score    = payload.cosine_score,
        match_ratio     = match_ratio,
        industry        = payload.industry,
    )

    return {
        "fit_reason":  fit_reason,
        "match_ratio": round(match_ratio, 4),
    }


@app.post("/thesis-full-analysis")
def thesis_full_analysis(file: UploadFile = File(...)) -> dict:
    """Full pipeline analysis for a single uploaded resume PDF."""

    # Unique per request — see the identical fix in target_career_analysis
    # below for why deriving this from file.filename directly is unsafe.
    ext = Path(file.filename).suffix
    temp_path = f"temp_{uuid.uuid4().hex}{ext}"
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

    Runs Stages 1-2 only (text acquisition + skill/cert discovery) — NOT
    the full pipeline — then looks up `target_job` in the Firestore corpus
    and scores that one job directly. This deliberately skips scoring
    against the other ~180 jobs in the corpus: no top-5 ranking, no
    fit_reason generation, no FAISS encoding pass over the full corpus.
    Only the target job gets encoded and scored.

    Uses the same SBERT semantic matching as the main scoring pipeline,
    so wording differences between resume and job posting don't count
    against the candidate.
    """

    # Unique per request (not derived from file.filename) — the old
    # f"temp_target_{file.filename}" pattern meant two uploads with the
    # SAME filename shared one path. Since the pipeline semaphore below
    # serializes requests, a second request with the same filename could
    # end up queued behind a first one; when the first finished and its
    # `finally` block deleted the (shared) temp file, the second request
    # would then try to read a file that no longer existed —
    # "[Errno 2] No such file or directory". A uuid-based path can never
    # collide between requests, so this can't happen anymore. The
    # extension is preserved because Ocr/main.py's _run_ocr() decides
    # PDF-vs-image by checking the file extension.
    ext = Path(file.filename).suffix
    temp_path = f"temp_target_{uuid.uuid4().hex}{ext}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        with _pipeline_semaphore:
            result = extract_and_discover(temp_path, file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[API] Target career pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    from core.scoring import calculate_cosine_similarity

    # jobs_corpus already came back from extract_and_discover's own
    # load_analyzer_data() call above — no second corpus fetch needed.
    jobs_corpus_full = result["jobs_corpus"]

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

    # Same Stage 3 scoring function the full analysis uses (hybrid score,
    # semantic skill/cert partitioning, nlp_validation, fit_reason) — just
    # scoped to a corpus of exactly one job instead of the whole corpus.
    target_match = calculate_cosine_similarity(
        resume_text   = result["resume_text"],
        resume_skills = result["discovered_skills"],
        jobs_corpus   = [target_job_doc],
        top_k         = 1,
        resume_certs  = result["discovered_certs"],
    )[0]

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
        "STAGE_3_TARGET_JOB_MATCHING": {
            "matching_method":     "SBERT semantic cosine similarity (skills and certs)",
            "ground_truth_source": "target_job_required_skills_and_certifications",
            "target_job": {
                "job_id":   target_match["job_id"],
                "title":    target_match["title"],
                "industry": target_match.get("industry", ""),
            },
            "score":              target_match["score"],
            "skills": {
                "ground_truth_count": len(target_match["required_skills"]),
                "nlp_accuracy":       target_match["nlp_validation"]["NLP_ACCURACY"],
                "matched_skills":     target_match["nlp_validation"]["matched_skills"],
                "unmatched_skills":   target_match["nlp_validation"]["unmatched_skills"],
            },
            "certifications": {
                "ground_truth_count": len(target_match["required_certifications"]),
                "nlp_accuracy":       target_match["nlp_validation_certs"]["NLP_ACCURACY"],
                "matched_certs":      target_match["matched_certs"],
                "missing_certs":      target_match["missing_certs"],
            },
            "fit_reason": target_match.get("fit_reason", ""),
        },
        "STAGE_4_FAISS_INTEGRITY": {
            "score":       f"{faiss_integrity}%",
            "index_total": _integrity_index.ntotal,
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)