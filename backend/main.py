"""
main.py
───────
FastAPI backend for AI-Based Resume Matching.

Pipeline (delegated to matcher.py):
  Stage 1   → Text acquisition   (PyMuPDF native text; EasyOCR fallback)
  Stage 2   → Skill extraction   (Llama-3.1-8B via HF Inference API; fuzzy fallback)
  Stage 2B  → Cert extraction    (Llama-3.1-8B via HF Inference API; regex fallback)
  Stage 3   → Hybrid scoring     (SBERT cosine + skill-match ratio)
             → NLP validation     (fuzzy_skill_verify run per-job inside matcher.py)
  Stage 4   → Gap analysis       (missing skills vs. best-matched job)

LLM calls go directly to the HuggingFace Inference API (same pattern as
generate_jobs.py) — no local model is loaded for the LLM stage.

Environment variables (loaded from .env):
    FIREBASE_CREDENTIALS_PATH       — path to serviceAccountKey.json  (required)
    FIRESTORE_JOBS_COLLECTION       — jobs collection name             (default: jobs)
    FIRESTORE_RESULTS_COLLECTION    — results collection name          (default: resume_results)
    HF_TOKEN                        — HuggingFace API token            (required)
    HF_MODEL                        — override LLM model string
                                      (default: meta-llama/Llama-3.1-8B-Instruct:cerebras)
    EASYOCR_LANGUAGES               — comma-separated language codes   (default: en)
    HOST                            — bind host                        (default: 0.0.0.0)
    PORT                            — bind port                        (default: 8000)

Changes (v3):
  NLP validation is now computed inside calculate_cosine_similarity() in
  matcher.py for EVERY matched job, not just the best one.  Each job dict in
  top_5_matches now carries a "nlp_validation" key:
    {
      "NLP_ACCURACY":     "87.50%",
      "matched_skills":   [...],
      "unmatched_skills": [...]
    }
  main.py reads nlp_validation directly from best["nlp_validation"] instead
  of calling fuzzy_skill_verify() separately, eliminating the duplicate call.
  The DEBUG_REPORT now shows per-job NLP accuracy for all top matches.
"""

# ── OpenMP / FAISS segfault guard — must be before any torch/faiss import ─────
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import shutil

import faiss
import fitz as _fitz   # PyMuPDF — used to detect native text availability
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

log = logging.getLogger(__name__)

# ── matcher.py public API ─────────────────────────────────────────────────────
from matcher import (
    extract_text,
    discover_skills_dynamically,
    extract_certificates,
    calculate_cosine_similarity,
    identify_skill_gap,
    normalize,
    model as sbert_model,
    _extract_skills_section_line_based,
    _extract_cert_section,
    HF_MODEL,
)

# ── Firestore client ──────────────────────────────────────────────────────────
from firestore_client import (
    load_analyzer_data,
    cache_job_embeddings,
    save_resume_result,
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Resume Matching API",
    description=(
        "Pipeline: PyMuPDF → EasyOCR → "
        "Llama-3.1-8B (HF Inference API) → SBERT → FAISS → Firestore"
    ),
)

# FAISS integrity index — 384 dims (all-MiniLM-L6-v2)
_integrity_index = faiss.IndexFlatL2(384)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/thesis-full-analysis")
async def thesis_full_analysis(
        file: UploadFile = File(...),
) -> dict:
    """
    Full pipeline analysis for a single uploaded resume PDF.

    Steps
    -----
    1.  Fetch jobs corpus + derive skills/certs allowlists from Firestore.
    2.  Save upload to a temp file.
    3.  Stage 1  : Text acquisition — PyMuPDF native; EasyOCR fallback.
    4.  Stage 2  : Skill extraction via Llama-3.1-8B (HF Inference API).
    5.  Stage 2B : Cert extraction via Llama-3.1-8B (HF Inference API).
    6.  Stage 3  : calculate_cosine_similarity() → hybrid scoring → top-5,
                   each result now includes nlp_validation computed against
                   that job's required_skills.
    7.  NLP accuracy for the pipeline report is read from best["nlp_validation"]
        — no separate fuzzy_skill_verify() call needed.
    8.  FAISS integrity self-check (vector round-trip).
    9.  Cert coverage for the best-matched job.
    10. Persist result to Firestore; cache newly generated job embeddings.
    """

    # ── 1. Load corpus + allowlists ───────────────────────────────────────────
    firestore_skills, jobs_corpus, firestore_certs = load_analyzer_data()

    # ── 2. Save upload to temp file ───────────────────────────────────────────
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # ── STAGE 1: Text acquisition ─────────────────────────────────────────
        resume_text, ocr_quality = extract_text(temp_path)

        _doc    = _fitz.open(temp_path)
        _native = " ".join(p.get_text("text") for p in _doc).strip()
        _doc.close()
        text_source = "native_pdf" if len(_native) >= 50 else "ocr"

        if not resume_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the uploaded file.",
            )

        # ── DEBUG: rule-based section slices ──────────────────────────────────
        skills_section_debug = _extract_skills_section_line_based(resume_text)
        cert_section_debug   = _extract_cert_section(resume_text)

        # ── STAGE 2: Skill extraction (Llama via HF API) ──────────────────────
        discovered_skills = discover_skills_dynamically(
            resume_text  = resume_text,
            jobs_corpus  = jobs_corpus,
            extra_skills = firestore_skills,
        )

        # ── STAGE 2B: Cert extraction (Llama via HF API) ──────────────────────
        discovered_certificates = extract_certificates(
            resume_text = resume_text,
            allowlist   = firestore_certs if firestore_certs else None,
        )

        # ── STAGE 3: Hybrid scoring + per-job NLP validation ──────────────────
        # calculate_cosine_similarity now embeds nlp_validation in every result
        # dict, so we don't need a separate fuzzy_skill_verify() call here.
        top_5_matches = calculate_cosine_similarity(
            resume_text   = resume_text,
            resume_skills = discovered_skills,
            jobs_corpus   = jobs_corpus,
        )

        if not top_5_matches:
            raise HTTPException(
                status_code=500,
                detail="Scoring returned no results. Check jobs corpus.",
            )

        best      = top_5_matches[0]
        secondary = top_5_matches[1] if len(top_5_matches) > 1 else best

        # ── FAISS INTEGRITY: vector round-trip self-check ─────────────────────
        from matcher import _get_sbert
        _sbert     = _get_sbert()
        resume_vec = (
            _sbert.encode(resume_text, convert_to_tensor=False)
            .reshape(1, -1)
            .astype("float32")
        )
        pre_count = _integrity_index.ntotal
        _integrity_index.add(resume_vec)
        D, I = _integrity_index.search(resume_vec, 1)
        faiss_integrity = (
            100.0 if I[0][0] == pre_count and D[0][0] < 0.0001 else 0.0
        )

        # ── NLP accuracy — read directly from the best match's nlp_validation ─
        # No separate fuzzy_skill_verify() call needed; matcher.py already ran
        # it against every job's required_skills during scoring.
        nlp_verification = best["nlp_validation"]
        nlp_accuracy     = nlp_verification["NLP_ACCURACY"]

        # ── Cert coverage for the best-matched job ────────────────────────────
        best_job_doc   = next(
            (j for j in jobs_corpus if j.get("id") == best.get("job_id")), {}
        )
        best_job_certs = best_job_doc.get("required_certifications", [])
        certs_norm     = {normalize(c) for c in discovered_certificates}
        matched_certs  = [c for c in best_job_certs if normalize(c) in certs_norm]
        missing_certs  = [c for c in best_job_certs if normalize(c) not in certs_norm]

        # ── Persist to Firestore ──────────────────────────────────────────────
        analysis_result = {
            "ocr_quality":  ocr_quality,
            "text_source":  text_source,
            "skills":       discovered_skills,
            "certificates": discovered_certificates,
            "validation":   nlp_verification,
            "top_matches":  top_5_matches,
        }
        save_resume_result(file.filename, analysis_result)

        # ── Cache newly computed job embeddings ────────────────────────────────
        cache_job_embeddings(jobs_corpus)

        # ── Build response ────────────────────────────────────────────────────
        return {
            "PIPELINE_ACCURACY_REPORT": {
                "STAGE_1_TEXT_ACQUISITION": {
                    "text_source":   text_source,
                    "char_count":    len(resume_text),
                    "ocr_quality":   f"{ocr_quality}%",
                    "ocr_triggered": text_source == "ocr",
                },
                "STAGE_2_SKILL_EXTRACTION": {
                    "llm_model":          HF_MODEL,
                    "total_skills_found": len(discovered_skills),
                },
                "STAGE_2B_CERT_EXTRACTION": {
                    "llm_model":         HF_MODEL,
                    "total_certs_found": len(discovered_certificates),
                },
                # NLP accuracy shown here is against the best-matched job only,
                # consistent with the previous behaviour — per-job breakdowns
                # are available inside each JOB_RECOMMENDATIONS entry.
                "STAGE_3_NLP_ACCURACY":    nlp_accuracy,
                "STAGE_4_FAISS_INTEGRITY": f"{faiss_integrity}%",
            },
            "CAREER_GUIDANCE_SYSTEM": {
                "JOB_RECOMMENDATIONS": top_5_matches,
                "BEST_MATCH_SUMMARY": {
                    "job_title":              best["title"],
                    "confidence_score":       best["score"],
                    "gap_count":              best["gap_count"],
                    "required_skills":        best["required_skills"],
                    "matched_skills":         best["matched_skills"],
                    "missing_skills":         best["missing_skills"],
                    "matched_certifications": matched_certs,
                    "missing_certifications": missing_certs,
                },
                "SKILL_RECOMMENDATIONS": {
                    "target_job":        secondary["title"],
                    "skill(s)_to_learn": secondary["missing_skills"],
                },
            },
            "EXTRACTION_SUMMARY": {
                "file_name":                file.filename,
                "text_source":              text_source,
                "discovered_skills":        discovered_skills,
                "total_skills_found":       len(discovered_skills),
                "discovered_certificates":  discovered_certificates,
                "total_certificates_found": len(discovered_certificates),
            },
            "DEBUG_REPORT": {
                "STAGE_1_TEXT_ACQUISITION": {
                    "text_source": text_source,
                    "ocr_quality": f"{ocr_quality}%",
                    "char_count":  len(resume_text),
                    "raw_text":    resume_text,
                },
                "STAGE_2_SKILL_EXTRACTION": {
                    "llm_model":                 HF_MODEL,
                    "skills_section_rule_based": skills_section_debug,
                    "discovered_skills":         discovered_skills,
                },
                "STAGE_2B_CERT_EXTRACTION": {
                    "llm_model":               HF_MODEL,
                    "cert_section_rule_based": cert_section_debug,
                    "discovered_certificates": discovered_certificates,
                },
                "STAGE_3_NLP_VALIDATION": {
                    # Best-job breakdown (summary level)
                    "ground_truth_source": "best_matched_job_required_skills",
                    "ground_truth_count":  len(best["required_skills"]),
                    "nlp_accuracy":        nlp_accuracy,
                    "matched_skills":      nlp_verification["matched_skills"],
                    "unmatched_skills":    nlp_verification["unmatched_skills"],
                    # Per-job NLP breakdown for all top matches
                    "per_job_nlp": [
                        {
                            "rank":             rank + 1,
                            "job_id":           match["job_id"],
                            "title":            match["title"],
                            "nlp_accuracy":     match["nlp_validation"]["NLP_ACCURACY"],
                            "matched_skills":   match["nlp_validation"]["matched_skills"],
                            "unmatched_skills": match["nlp_validation"]["unmatched_skills"],
                            "ground_truth_count": len(match["required_skills"]),
                        }
                        for rank, match in enumerate(top_5_matches)
                    ],
                },
                "STAGE_4_FAISS_INTEGRITY": {
                    "score":       f"{faiss_integrity}%",
                    "index_total": _integrity_index.ntotal,
                },
            },
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)