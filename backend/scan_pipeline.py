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

    Runs all stages but restricts the final output to the one job that matches
    target_job (exact title first, then substring), mirroring the logic in the
    /target-career-analysis endpoint in main.py.
    """
    # Run the full pipeline — we need all stages anyway
    result = run_scan_pipeline_full(raw_text, file_name)

    # Find the target job among all scored matches
    all_matches: list[dict] = result["top_5_matches"]
    target_lower = target_job.strip().lower()

    exact     = [m for m in all_matches if m["title"].lower() == target_lower]
    substring = [m for m in all_matches if target_lower in m["title"].lower()]
    candidates = exact or substring

    if not candidates:
        raise ValueError(
            f"No job matching '{target_job}' was found in the corpus. "
            "Check the title and try again."
        )

    target_match: dict = max(candidates, key=lambda m: m["score"])

    # Re-run NLP validation specifically against the target job's required skills
    from core.validation import fuzzy_skill_verify

    nlp = fuzzy_skill_verify(
        discovered_list   = result["discovered_skills"],
        ground_truth_list = target_match["required_skills"],
    )

    # Cert gap for the target job
    from db import load_analyzer_data as _lad
    _, jobs_corpus, _ = _lad()
    job_doc    = next((j for j in jobs_corpus if j.get("id") == target_match.get("job_id")), {})
    req_certs  = job_doc.get("required_certifications", [])
    certs_norm = {normalize(c) for c in result["discovered_certs"]}
    matched_certs = [c for c in req_certs if normalize(c) in certs_norm]
    missing_certs = [c for c in req_certs if normalize(c) not in certs_norm]

    return {
        # Scan meta
        "resume_text":   result["resume_text"],
        "ocr_quality":   result["ocr_quality"],
        "text_source":   result["text_source"],
        # Discovered
        "discovered_skills": result["discovered_skills"],
        "discovered_certs":  result["discovered_certs"],
        # Target job
        "target_job": {
            "job_id":   target_match.get("job_id", ""),
            "title":    target_match.get("title",  ""),
            "industry": target_match.get("industry", ""),
        },
        # Gap analysis
        "skill_gap": {
            "required_skills": target_match.get("required_skills", []),
            "matched_skills":  nlp["matched_skills"],
            "missing_skills":  nlp["unmatched_skills"],
            "nlp_accuracy":    nlp["NLP_ACCURACY"],
            "match_ratio":     target_match.get("match_ratio", 0.0),
            "hybrid_score":    target_match.get("score", 0.0),
        },
        "cert_gap": {
            "required_certs": req_certs,
            "matched_certs":  matched_certs,
            "missing_certs":  missing_certs,
        },
    }