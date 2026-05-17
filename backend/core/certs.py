"""
core/certs.py
=============
Stage 2B — Certification extraction.

Changes (v7):
  FIXED    Hallucinated certs no longer pass through.

           Problem: after canonicalize() the code only deduped results — it
           never checked whether the canonicalized cert was actually present in
           the Firestore allowlist. So hallucinated items like "Employee of the
           Quarter" were kept as-is when canonicalize() found no match and
           returned the original string unchanged.

           Fix: mirrors the skills drop-guard introduced in skills.py —
           if normalize(canon) == normalize(original) AND normalize(canon) is
           not a key in allow_norm, the item is not in the allowlist and is
           dropped. Items that canonicalize successfully (i.e. the cert was
           OCR-garbled but matched a real allowlist entry) are kept normally.

Changes (v8):
  CHANGED  Structure now mirrors skills.py exactly.

           - extract_cert_section() returns the raw cert section text,
             equivalent to extract_skills_section().
           - discover_certs_dynamically() builds the corpus allowlist from
             required_certifications, calls the LLM, canonicalises output,
             runs direct_scan_section() to recover anything the LLM missed,
             and falls back to fuzzy_scan() if both produce nothing —
             identical fallback chain to discover_skills_dynamically().
           - Regex keyword fallback removed.
"""

from __future__ import annotations

import logging
import re

from core.utils import (
    normalize,
    llm_extract_list,
    build_allow_norm_sorted,
    canonicalize,
    direct_scan_section,
    fuzzy_scan,
)
from core.ocr_correction import (
    build_ocr_corrector,
    apply_ocr_corrections,
)

log = logging.getLogger(__name__)

_CERT_HEADING_RE = re.compile(
    r"^(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?)",
    re.I | re.M,
    )
_CERT_END_RE = re.compile(
    r"^\s*(skills?|technical\s+skills?|experience|education|employment|"
    r"projects?|awards?|publications?|references?|volunteer|languages?|"
    r"interests?|activities|summary|objective)",
    re.I | re.M,
    )

_CERTS_SYSTEM = (
    "You are a specialized resume parser. Extract COMPLETE professional certification names. "
    "CRITICAL RULE: NEVER split a single certification into multiple array items. "
    "Each certification is ONE complete string "
    "(e.g., 'Microsoft Certified Azure Fundamentals' is ONE item, NOT two). "
    "Each entry must be 10 words or fewer. "
    "Output MUST be a valid JSON array of strings only. "
    "Do not include markdown tags like ```json or any explanation. "
    "Example: [\"AWS Certified Solutions Architect\", \"Google IT Support Professional\"]"
)


def extract_cert_section(text: str) -> str:
    """
    Return the raw text of the certifications section.

    Equivalent to extract_skills_section() in skills.py — locates the heading
    via regex and slices until the next recognised section heading.
    """
    m = _CERT_HEADING_RE.search(text)
    if not m:
        return ""
    rest  = text[m.end():]
    end_m = _CERT_END_RE.search(rest)
    return (rest[: end_m.start()].strip() if end_m else rest.strip())


def discover_certs_dynamically(
        resume_text:  str,
        jobs_corpus:  list[dict],
        extra_certs:  list[str] | None = None,
) -> list[str]:
    """
    Stage 2B — certification extraction via Llama (HF Inference API).

    Mirrors discover_skills_dynamically() in skills.py:

    1. Build corpus allowlist from required_certifications across all jobs.
    2. Build OcrCorrector from that allowlist.
    3. Extract cert section text (returns empty if no section found).
    4. Call LLM to get cert name candidates.
    5. Canonicalise LLM output against allowlist; drop non-allowlist items.
    6. direct_scan_section() to recover anything the LLM missed.
    7. fuzzy_scan() fallback if both produce nothing.

    Returns deduplicated certification name strings.
    """
    # ── 1. Build corpus allowlist ─────────────────────────────────────────────
    allowlist: set[str] = set()
    for job in jobs_corpus:
        allowlist.update(job.get("required_certifications", []))
    if extra_certs:
        allowlist.update(extra_certs)
    allowlist_list = sorted(allowlist)
    allow_norm     = {normalize(a): a for a in allowlist_list}
    sorted_pairs   = build_allow_norm_sorted(allowlist_list)
    corrector      = build_ocr_corrector(allowlist_list)

    # ── 2. Extract cert section ───────────────────────────────────────────────
    # Do NOT fall back to full resume text — the LLM will hallucinate certs
    # from skills/work experience sections (e.g. "AWS" as a fake certification).
    section = extract_cert_section(resume_text)
    if not section:
        log.info("[Certs] No certifications section found — returning empty.")
        return []

    # ── 3. Call LLM ──────────────────────────────────────────────────────────
    user_prompt = (
        f"Extract every professional certification, license, and credential "
        f"from this resume section. "
        f"IMPORTANT: Each certification is ONE complete string — never split "
        f"'Microsoft Certified Azure Fundamentals' into two separate items. "
        f"Each must be 10 words or fewer.\n\n"
        f"Resume section:\n{section[:2500]}\n\n"
        f"Return ONLY a JSON array of certification name strings. No other text."
    )

    raw = llm_extract_list(_CERTS_SYSTEM, user_prompt, "CERTS")

    # ── 4. Canonicalise LLM output ────────────────────────────────────────────
    seen:      set[str]  = set()
    canonical: list[str] = []

    for cert in raw:
        # Enforce 10-word guard before canonicalisation
        cert      = " ".join(cert.split()[:10])
        corrected = apply_ocr_corrections(cert, corrector)
        canon     = canonicalize(
            corrected, allow_norm, threshold=75,
            _sorted_pairs=sorted_pairs, corrector=corrector,
        )
        if normalize(canon) == normalize(corrected) and normalize(canon) not in allow_norm:
            log.debug(f"[Certs/LLM] Dropped non-allowlist item: {cert!r}")
            continue
        cn = normalize(canon)
        if cn and cn not in seen:
            seen.add(cn)
            canonical.append(canon)

    # ── 5. Direct scan to recover anything LLM missed ────────────────────────
    for hit in direct_scan_section(
            section, sorted_pairs, threshold=82, corrector=corrector, allow_norm=allow_norm
    ):
        hn = normalize(hit)
        if hn and hn not in seen:
            seen.add(hn)
            canonical.append(hit)

    # ── 6. Fuzzy fallback if both produced nothing ────────────────────────────
    if not canonical:
        log.info("[Certs] LLM + direct scan produced nothing — fuzzy fallback.")
        canonical = fuzzy_scan(section, allowlist_list)

    log.info(f"[Certs] {len(canonical)} certifications discovered.")
    return canonical