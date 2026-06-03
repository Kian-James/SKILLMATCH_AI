"""
core/certs.py
=============
Stage 2B — Certification extraction.

Locates the CERTIFICATIONS section, runs a regex-based extractor, and
canonicalises against the allowlist when one is provided.

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

Changes (v8): removed LLM (meta-llama/Llama-3.1-8B-Instruct via HF Inference
           API). The LLM was unreliable and added network latency with no
           benefit over the regex fallback that was already present.

           The regex-based extractor (previously the fallback) is now the
           primary and only extraction path. _CERTS_SYSTEM, user_prompt
           construction, llm_extract_list call, and the 10-word guard on raw
           LLM output are all removed. The 10-word guard is retained on the
           regex path since line length is still a useful noise filter.

           Also removed: llm_extract_list import from core.utils.
"""

from __future__ import annotations

import logging
import re

from core.utils import (
    normalize,
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

_CERT_LINE_RE = re.compile(
    r"(?i)(certif\w+|licen[cs]\w+|aws|gcp|azure|pmp|cpa|cfa|"
    r"ccna|comptia|cisco|oracle|google|microsoft|istqb|six\s+sigma)\b"
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
    Stage 2B — certification extraction via regex line scan.

    Locates the certifications section, scans each line against _CERT_LINE_RE,
    applies the 10-word length guard, then canonicalises against the allowlist
    when one is provided. Non-allowlist items are dropped when an allowlist
    exists to prevent hallucination-style noise.

    Returns deduplicated certification name strings.
    """
    section = extract_cert_section(resume_text)

    if not section:
        log.info("[Certs] No certifications section found — returning empty.")
        return []

    allow_norm   = {normalize(a): a for a in allowlist} if allowlist else {}
    sorted_pairs = build_allow_norm_sorted(allowlist) if allowlist else []
    corrector    = build_ocr_corrector(allowlist) if allowlist else None

    seen:   set[str]  = set()
    result: list[str] = []

    for line in section.splitlines():
        line  = line.strip()
        words = line.split()

        if not line or not _CERT_LINE_RE.search(line):
            continue

        # 10-word guard — anything longer is likely a prose sentence, not a
        # cert name.
        if len(words) > 10:
            log.debug(f"[Certs] Skipping long line ({len(words)} words): {line!r}")
            continue

        if allowlist:
            canon = canonicalize(
                line, allow_norm, threshold=75,
                _sorted_pairs=sorted_pairs, corrector=corrector,
            )
            # Drop if canonicalize made no change AND cert is not in the
            # allowlist — catches noise lines that happen to contain a keyword
            # but don't match any real certification.
            if normalize(canon) == normalize(line) and normalize(canon) not in allow_norm:
                log.debug(f"[Certs] Dropped non-allowlist item: {line!r}")
                continue
        else:
            canon = line

        n = normalize(canon)
        if n and n not in seen:
            seen.add(n)
            result.append(canon)

    log.info(f"[Certs] {len(result)} certifications discovered.")
    return result