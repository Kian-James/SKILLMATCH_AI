"""
pipeline.py
===========
Orchestrator: runs all stages in order and returns the full result dict.

main.py calls run_pipeline() and only handles HTTP concerns (file saving,
response assembly, FAISS integrity check). All analysis logic lives here.
"""

from __future__ import annotations

import logging

from core.extraction  import extract_text
from core.skills      import discover_skills_dynamically, extract_skills_section
from core.certs       import extract_certificates, extract_cert_section
from core.scoring     import calculate_cosine_similarity
from core.utils       import normalize
from db               import load_analyzer_data, cache_job_embeddings, save_resume_result

log = logging.getLogger(__name__)


def run_pipeline(temp_path: str, file_name: str) -> dict:
    """
    Execute the full resume analysis pipeline.

    Parameters
    ----------
    temp_path : path to the saved upload on disk
    file_name : original filename (for Firestore record)

    Returns
    -------
    Full result dict consumed by the FastAPI endpoint.
    """
    # Stage 0 — load corpus
    firestore_skills, jobs_corpus, firestore_certs = load_analyzer_data()

    # Stage 1 — text acquisition
    resume_text, ocr_quality, is_native = extract_text(temp_path)
    text_source = "native_pdf" if is_native else "ocr"

    # Debug: rule-based section slices (captured before LLM runs).
    # Pass firestore_skills as the allowlist so that single-line OCR blobs
    # are segmented by split_ocr_blob and the debug field shows a proper
    # comma-separated skill list instead of one unsplit blob string.
    skills_section_debug = extract_skills_section(
        resume_text,
        allowlist     = firestore_skills if firestore_skills else None,
        is_native_pdf = is_native,
    )
    cert_section_debug = extract_cert_section(resume_text)

    # Stage 2 — skill extraction
    discovered_skills = discover_skills_dynamically(
        resume_text   = resume_text,
        jobs_corpus   = jobs_corpus,
        extra_skills  = firestore_skills,
        is_native_pdf = is_native,
    )

    # Stage 2B — cert extraction
    discovered_certificates = extract_certificates(
        resume_text = resume_text,
        allowlist   = firestore_certs if firestore_certs else None,
    )

    # Stage 3 — hybrid scoring + per-job NLP validation
    top_5_matches = calculate_cosine_similarity(
        resume_text   = resume_text,
        resume_skills = discovered_skills,
        jobs_corpus   = jobs_corpus,
    )

    best      = top_5_matches[0]
    secondary = top_5_matches[1] if len(top_5_matches) > 1 else best

    # Cert coverage for best-matched job
    best_job_doc   = next((j for j in jobs_corpus if j.get("id") == best.get("job_id")), {})
    best_job_certs = best_job_doc.get("required_certifications", [])
    certs_norm     = {normalize(c) for c in discovered_certificates}
    matched_certs  = [c for c in best_job_certs if normalize(c) in certs_norm]
    missing_certs  = [c for c in best_job_certs if normalize(c) not in certs_norm]

    nlp_verification = best["nlp_validation"]
    nlp_accuracy     = nlp_verification["NLP_ACCURACY"]

    # Persist
    analysis_result = {
        "ocr_quality":  ocr_quality,
        "text_source":  text_source,
        "skills":       discovered_skills,
        "certificates": discovered_certificates,
        "validation":   nlp_verification,
        "top_matches":  top_5_matches,
    }
    save_resume_result(file_name, analysis_result)
    cache_job_embeddings(jobs_corpus)

    return {
        "resume_text":           resume_text,
        "ocr_quality":           ocr_quality,
        "text_source":           text_source,
        "discovered_skills":     discovered_skills,
        "discovered_certs":      discovered_certificates,
        "top_5_matches":         top_5_matches,
        "best":                  best,
        "secondary":             secondary,
        "nlp_verification":      nlp_verification,
        "nlp_accuracy":          nlp_accuracy,
        "matched_certs":         matched_certs,
        "missing_certs":         missing_certs,
        "skills_section_debug":  skills_section_debug,
        "cert_section_debug":    cert_section_debug,
    }