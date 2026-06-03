"""
scan_pipeline.py
================
Pipeline variant for camera-scanned resumes.

The Flutter camera scanner runs ML Kit OCR on-device and sends the resulting
plain text directly.  There is no file to open and no EasyOCR/PyMuPDF step —
the text is already in memory.

Stage 1 is therefore replaced by a lightweight *text-quality* measurement that:
  • Accepts the raw OCR string as-is.
  • Runs split_ocr_blob on every line that looks like a collapsed blob so that
    the same skill-splitting logic used in the normal pipeline is applied.
  • Reports char count and alnum-ratio as the "ocr_quality" signal.

Stages 2 → 4 are identical to run_pipeline() — we reuse the same core modules
so behaviour is consistent regardless of whether input came from a file or from
the camera scanner.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from core.skills     import discover_skills_dynamically, extract_skills_section
from core.certs      import extract_certificates, extract_cert_section
from core.scoring    import calculate_cosine_similarity
from core.utils      import normalize
from db              import load_analyzer_data, cache_job_embeddings, save_resume_result

log = logging.getLogger(__name__)


# ── Stage 1 replacement ──────────────────────────────────────────────────────

def _alnum_ratio(text: str) -> float:
    """Percentage of alphanumeric characters — proxy for OCR quality."""
    if not text:
        return 0.0
    return round(sum(c.isalnum() for c in text) / len(text) * 100, 2)


def _clean_scan_text(raw: str) -> str:
    """
    Light normalisation pass on ML Kit output:
      1. Collapse runs of blank lines to a single blank line.
      2. Strip leading/trailing whitespace per line.
      3. Remove zero-width / control characters that confuse the tokeniser.
    The text is NOT uppercased or lowercased — downstream modules expect
    mixed-case input (the LLM prompt is case-sensitive for proper nouns).
    """
    # Drop control characters except newline and tab
    cleaned = re.sub(r"[^\S\n\t ]+", " ", raw)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Strip per-line leading/trailing whitespace
    lines = [line.strip() for line in cleaned.splitlines()]

    # Collapse 3+ consecutive blank lines → 1
    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)

    return "\n".join(result).strip()


# ── Public entry points ───────────────────────────────────────────────────────

def run_scan_pipeline_full(raw_text: str, file_name: str) -> dict:
    """
    Full-analysis variant (/scan-thesis-full-analysis).

    Mirrors run_pipeline() but accepts raw scan text instead of a file path.
    Returns the same dict shape so main.py can assemble an identical response.
    """
    # Stage 0 — corpus
    firestore_skills, jobs_corpus, firestore_certs = load_analyzer_data()

    # Stage 1 — text quality (no file I/O)
    resume_text = _clean_scan_text(raw_text)
    ocr_quality = _alnum_ratio(resume_text)
    # Camera scan is never a native PDF — always treat as OCR output
    is_native   = False
    text_source = "camera_scan_mlkit"

    log.info(
        "[Scan Stage 1] Received %d chars, alnum-ratio %.1f%%",
        len(resume_text), ocr_quality,
    )

    # Stage 1½ — rule-based section debug (same as normal pipeline)
    skills_section_debug = extract_skills_section(
        resume_text,
        allowlist     = firestore_skills if firestore_skills else None,
        is_native_pdf = is_native,          # triggers split_ocr_blob path
    )
    cert_section_debug = extract_cert_section(resume_text)

    # Stage 2 — skill extraction (LLM + fuzzy matching against corpus)
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

    # Stage 3 — hybrid SBERT/FAISS scoring + per-job NLP validation
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
    save_resume_result(file_name, {
        "ocr_quality":  ocr_quality,
        "text_source":  text_source,
        "skills":       discovered_skills,
        "certificates": discovered_certificates,
        "validation":   nlp_verification,
        "top_matches":  top_5_matches,
    })
    cache_job_embeddings(jobs_corpus)

    return {
        "resume_text":          resume_text,
        "ocr_quality":          ocr_quality,
        "text_source":          text_source,
        "discovered_skills":    discovered_skills,
        "discovered_certs":     discovered_certificates,
        "top_5_matches":        top_5_matches,
        "best":                 best,
        "secondary":            secondary,
        "nlp_verification":     nlp_verification,
        "nlp_accuracy":         nlp_accuracy,
        "matched_certs":        matched_certs,
        "missing_certs":        missing_certs,
        "skills_section_debug": skills_section_debug,
        "cert_section_debug":   cert_section_debug,
    }


def run_scan_pipeline_targeted(raw_text: str, file_name: str, target_job: str) -> dict:
    """
    Targeted-analysis variant (/scan-target-career-analysis).

    Runs all pipeline stages then resolves the target job directly from the
    FULL corpus by title — not from top_5_matches which is score-gated.
    This ensures aspirational jobs (low current score) are always found.
    """
    from core.validation import fuzzy_skill_verify

    # Run the full pipeline for skill/cert extraction
    result = run_scan_pipeline_full(raw_text, file_name)

    # ── Resolve target job from full corpus ───────────────────────────────────
    # load_analyzer_data is cached in memory after the first call so this is
    # effectively free — no extra Firestore round-trip.
    _, jobs_corpus_full, _ = load_analyzer_data()

    target_lower = target_job.strip().lower()
    exact     = [j for j in jobs_corpus_full if j["title"].lower() == target_lower]
    substring = [j for j in jobs_corpus_full if target_lower in j["title"].lower()]
    candidates = exact or substring

    if not candidates:
        raise ValueError(
            f"No job matching '{target_job}' was found in the corpus. "
            "Check the title and try again."
        )

    # Pick the title match with the most required skills (richest ground truth)
    target_job_doc: dict = max(candidates, key=lambda j: len(j.get("required_skills", [])))

    # ── NLP validation against the target job ─────────────────────────────────
    req_skills = target_job_doc.get("required_skills", [])
    nlp = fuzzy_skill_verify(
        discovered_list   = result["discovered_skills"],
        ground_truth_list = req_skills,
    )

    # ── Match ratio (no SBERT re-encode needed) ────────────────────────────────
    resume_norms = {normalize(s) for s in result["discovered_skills"]}
    matched_now  = [s for s in req_skills if normalize(s) in resume_norms]
    match_ratio  = round(len(matched_now) / len(req_skills), 4) if req_skills else 0.0

    # ── Cert gap ──────────────────────────────────────────────────────────────
    req_certs  = target_job_doc.get("required_certifications", [])
    certs_norm = {normalize(c) for c in result["discovered_certs"]}
    matched_certs = [c for c in req_certs if normalize(c) in certs_norm]
    missing_certs = [c for c in req_certs if normalize(c) not in certs_norm]

    return {
        "resume_text":       result["resume_text"],
        "ocr_quality":       result["ocr_quality"],
        "text_source":       result["text_source"],
        "discovered_skills": result["discovered_skills"],
        "discovered_certs":  result["discovered_certs"],
        "target_job": {
            "job_id":   target_job_doc.get("id", ""),
            "title":    target_job_doc.get("title", ""),
            "industry": target_job_doc.get("industry", ""),
        },
        "skill_gap": {
            "required_skills": req_skills,
            "matched_skills":  nlp["matched_skills"],
            "missing_skills":  nlp["unmatched_skills"],
            "nlp_accuracy":    nlp["NLP_ACCURACY"],
            "match_ratio":     match_ratio,
            "hybrid_score":    match_ratio,  # no SBERT here; ratio-only score
        },
        "cert_gap": {
            "required_certs": req_certs,
            "matched_certs":  matched_certs,
            "missing_certs":  missing_certs,
        },
    }
