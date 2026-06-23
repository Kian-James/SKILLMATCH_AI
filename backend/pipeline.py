"""
pipeline.py
===========
Orchestrator: runs all stages in order and returns the full result dict.

main.py calls run_pipeline() and only handles HTTP concerns (file saving,
response assembly, FAISS integrity check). All analysis logic lives here.

Stages
------
Stage 1 — text acquisition           (core/extraction.py)
                                      Plain parsing — PyMuPDF for native PDFs,
                                      EasyOCR fallback for scanned ones. No AI
                                      matching happens here.

Stage 2 — skills + certs discovery   (core/skills.py, core/certs.py)
                                      Each candidate phrase pulled from the
                                      resume is semantically matched against
                                      the job corpus's skill/cert allowlist
                                      (SBERT cosine similarity, threshold 0.80)
                                      inside discover_items_dynamically(). Only
                                      matches above threshold are accepted.

Stage 3 — job matching                (core/scoring.py, core/validation.py)
                                      Discovered skills AND certs are compared
                                      against every job's required_skills /
                                      required_certifications, also via SBERT
                                      cosine similarity — no string-fuzzy
                                      matching (rapidfuzz) anywhere in this
                                      pipeline anymore. Produces the ranked
                                      top_5_matches plus a parallel semantic
                                      NLP-accuracy diagnostic for both skills
                                      and certs on every job in the list.
"""

from __future__ import annotations

import logging

from core.extraction  import extract_text
from core.skills      import discover_skills_dynamically, extract_skills_section
from core.certs       import extract_certificates, extract_cert_section
from core.scoring     import calculate_cosine_similarity
from db               import load_analyzer_data, cache_job_embeddings, save_resume_result

log = logging.getLogger(__name__)

logging.getLogger("core.skills").setLevel(logging.DEBUG)


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

    # Debug: rule-based section slices (captured before discovery runs).
    skills_section_debug = extract_skills_section(resume_text)
    cert_section_debug   = extract_cert_section(resume_text)

    # Stage 2 — skill discovery (semantic match against allowlist happens
    # inside this call, via discover_items_dynamically)
    discovered_skills = discover_skills_dynamically(
        resume_text   = resume_text,
        jobs_corpus   = jobs_corpus,
        extra_skills  = firestore_skills,
        is_native_pdf = is_native,
    )

    # Stage 2 — cert discovery (same mechanism as skills, separate config)
    discovered_certificates = extract_certificates(
        resume_text   = resume_text,
        jobs_corpus   = jobs_corpus,
        extra_certs   = firestore_certs if firestore_certs else None,
        is_native_pdf = is_native,
    )

    # Stage 3 — hybrid scoring + per-job semantic validation (skills AND certs)
    top_5_matches = calculate_cosine_similarity(
        resume_text   = resume_text,
        resume_skills = discovered_skills,
        jobs_corpus   = jobs_corpus,
        resume_certs  = discovered_certificates,
    )

    best      = top_5_matches[0]
    secondary = top_5_matches[1] if len(top_5_matches) > 1 else best

    # Cert coverage for the best-matched job — sourced directly from
    # calculate_cosine_similarity's per-job semantic partition rather than
    # recomputed here with exact-string matching, so this stays consistent
    # with how skills are matched (same SBERT cosine mechanism, same call).
    matched_certs = best["matched_certs"]
    missing_certs = best["missing_certs"]

    nlp_verification       = best["nlp_validation"]
    nlp_accuracy           = nlp_verification["NLP_ACCURACY"]
    nlp_verification_certs = best["nlp_validation_certs"]
    nlp_accuracy_certs     = nlp_verification_certs["NLP_ACCURACY"]

    # Persist
    analysis_result = {
        "ocr_quality":      ocr_quality,
        "text_source":      text_source,
        "skills":           discovered_skills,
        "certificates":     discovered_certificates,
        "validation":       nlp_verification,
        "validation_certs": nlp_verification_certs,
        "top_matches":      top_5_matches,
    }
    save_resume_result(file_name, analysis_result)
    cache_job_embeddings(jobs_corpus)

    return {
        "resume_text":            resume_text,
        "ocr_quality":            ocr_quality,
        "text_source":            text_source,
        "discovered_skills":      discovered_skills,
        "discovered_certs":       discovered_certificates,
        "top_5_matches":          top_5_matches,
        "best":                   best,
        "secondary":              secondary,
        "nlp_verification":       nlp_verification,
        "nlp_accuracy":           nlp_accuracy,
        "nlp_verification_certs": nlp_verification_certs,
        "nlp_accuracy_certs":     nlp_accuracy_certs,
        "matched_certs":          matched_certs,
        "missing_certs":          missing_certs,
        "skills_section_debug":   skills_section_debug,
        "cert_section_debug":     cert_section_debug,
    }