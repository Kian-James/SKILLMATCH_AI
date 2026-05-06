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
from fastapi import FastAPI, File, HTTPException, UploadFile
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
        "PIPELINE_ACCURACY_REPORT": {
            "STAGE_1_TEXT_ACQUISITION": {
                "text_source":   result["text_source"],
                "char_count":    len(result["resume_text"]),
                "ocr_quality":   f"{result['ocr_quality']}%",
                "ocr_triggered": result["text_source"] == "ocr",
            },
            "STAGE_2_SKILL_EXTRACTION": {
                "llm_model":          HF_MODEL,
                "total_skills_found": len(result["discovered_skills"]),
            },
            "STAGE_2B_CERT_EXTRACTION": {
                "llm_model":         HF_MODEL,
                "total_certs_found": len(result["discovered_certs"]),
            },
            "STAGE_3_NLP_ACCURACY":    result["nlp_accuracy"],
            "STAGE_4_FAISS_INTEGRITY": f"{faiss_integrity}%",
        },
        "CAREER_GUIDANCE_SYSTEM": {
            "JOB_RECOMMENDATIONS": result["top_5_matches"],
            "BEST_MATCH_SUMMARY": {
                "job_title":              best["title"],
                "confidence_score":       best["score"],
                "gap_count":              best["gap_count"],
                "required_skills":        best["required_skills"],
                "matched_skills":         best["matched_skills"],
                "missing_skills":         best["missing_skills"],
                "matched_certifications": result["matched_certs"],
                "missing_certifications": result["missing_certs"],
            },
            "SKILL_RECOMMENDATIONS": {
                "target_job":        secondary["title"],
                "skill(s)_to_learn": secondary["missing_skills"],
            },
        },
        "EXTRACTION_SUMMARY": {
            "file_name":                file.filename,
            "text_source":              result["text_source"],
            "discovered_skills":        result["discovered_skills"],
            "total_skills_found":       len(result["discovered_skills"]),
            "discovered_certificates":  result["discovered_certs"],
            "total_certificates_found": len(result["discovered_certs"]),
        },
        "DEBUG_REPORT": {
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
                "per_job_nlp": [
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
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)