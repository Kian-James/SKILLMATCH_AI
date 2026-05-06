"""
core/certs.py
=============
Stage 2B — Certification extraction.

Locates the CERTIFICATIONS section, calls the LLM, enforces a 10-word-per-item
guard, and canonicalises against the allowlist when one is provided.

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
"""

from __future__ import annotations

import logging
import re

from core.utils import (
    normalize,
    llm_extract_list,
    build_allow_norm_sorted,
    canonicalize,
)
from core.ocr_correction import build_ocr_corrector

log = logging.getLogger(__name__)

_CERT_HEADING_RE = re.compile(
    r"^(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?)",
    re.I | re.M,
    )
_NEXT_SECTION_RE = re.compile(
    r"^\s*(experience|education|employment|projects?|awards?|publications?|"
    r"references?|volunteer|languages?|interests?|activities|summary|objective)",
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
    """Return raw text of the certification section."""
    m = _CERT_HEADING_RE.search(text)
    if not m:
        return ""
    rest  = text[m.end():]
    end_m = _NEXT_SECTION_RE.search(rest)
    return (rest[: end_m.start()].strip() if end_m else rest.strip())


def extract_certificates(
        resume_text: str,
        allowlist:   list[str] | None = None,
) -> list[str]:
    """
    Stage 2B — certification extraction via Llama (HF Inference API).

    When an allowlist is provided (derived from Firestore required_certifications),
    only certs that canonicalise to a known allowlist entry are returned.
    Hallucinated items that do not match anything in the allowlist are dropped.

    Returns deduplicated certification name strings.
    """
    section = extract_cert_section(resume_text)

    # If no cert section exists, return early — do NOT fall back to full resume
    # text. The LLM will hallucinate certs from skills/work experience otherwise
    # (e.g. "AWS" extracted from the SKILLS section as a fake certification).
    if not section:
        log.info("[Certs] No certifications section found — returning empty.")
        return []

    user_prompt = (
        f"Extract every professional certification, license, and credential "
        f"from this resume section. "
        f"IMPORTANT: Each certification is ONE complete string — never split "
        f"'Microsoft Certified Azure Fundamentals' into two separate items. "
        f"Each must be 10 words or fewer.\n\n"
        f"Resume section:\n{section[:2500]}\n\n"
        f"Return ONLY a JSON array of certification name strings. No other text."
    )

    raw     = llm_extract_list(_CERTS_SYSTEM, user_prompt, "CERTS")
    guarded = [" ".join(c.split()[:10]) for c in raw if c.strip()]

    if allowlist:
        allow_norm   = {normalize(a): a for a in allowlist}
        sorted_pairs = build_allow_norm_sorted(allowlist)
        corrector    = build_ocr_corrector(allowlist)
        seen:   set[str]  = set()
        result: list[str] = []

        for cert in guarded:
            canon = canonicalize(
                cert, allow_norm, threshold=75,
                _sorted_pairs=sorted_pairs, corrector=corrector,
            )
            # Drop if canonicalize made no change AND the cert is not in the
            # allowlist — this catches hallucinated items like "Employee of the
            # Quarter" that have no real allowlist match.
            if normalize(canon) == normalize(cert) and normalize(canon) not in allow_norm:
                log.debug(f"[Certs] Dropped non-allowlist item: {cert!r}")
                continue

            n = normalize(canon)
            if n and n not in seen:
                seen.add(n)
                result.append(canon)

        guarded = result
    else:
        # No allowlist — just deduplicate, keep all LLM output
        seen_n:  set[str]  = set()
        deduped: list[str] = []
        for c in guarded:
            n = normalize(c)
            if n not in seen_n:
                seen_n.add(n)
                deduped.append(c)
        guarded = deduped

    if not guarded:
        log.info("[Certs] LLM produced nothing — regex fallback.")
        cert_re = re.compile(
            r"(?i)(certif\w+|licen[cs]\w+|aws|gcp|azure|pmp|cpa|cfa|"
            r"ccna|comptia|cisco|oracle|google|microsoft|istqb|six\s+sigma)\b"
        )
        seen_fb: set[str] = set()
        for line in section.splitlines():
            line  = line.strip()
            words = line.split()
            if cert_re.search(line) and 1 <= len(words) <= 10:
                n = normalize(line)
                # Also apply allowlist check to regex fallback
                if allowlist and n not in {normalize(a) for a in allowlist}:
                    log.debug(f"[Certs/Fallback] Dropped non-allowlist item: {line!r}")
                    continue
                if n not in seen_fb:
                    seen_fb.add(n)
                    guarded.append(line)

    log.info(f"[Certs] {len(guarded)} certifications discovered.")
    return guarded