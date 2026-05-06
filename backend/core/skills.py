"""
core/skills.py
==============
Stage 2 — Skill extraction.

Changes (v6 backport): added is_native_pdf to fix early-exit guard truncation.
Changes (v7): fixed split_ocr_blob greedy consumption bug.

           Problem: longest-first-hit algorithm committed to win=2 "Zendosk AnyDesk"
           matching a compound allowlist entry (e.g. "Zendesk AnyDesk") at score 93,
           consuming both tokens and silently dropping "AnyDesk" as its own skill.

           Fix: check win=1 first. If score >= _SINGLE_TOKEN_CONFIDENCE (85),
           commit immediately without trying longer windows. Only fall through to
           multi-token search when the single token is ambiguous (score < 85).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from rapidfuzz import fuzz

from config import BLOB_FUZZY_THRESHOLD
from core.utils import (
    normalize,
    llm_extract_list,
    build_allow_norm_sorted,
    canonicalize,
    direct_scan_section,
    fuzzy_scan,
)
from core.ocr_correction import (
    OcrCorrector,
    build_ocr_corrector,
    apply_ocr_corrections,
)

log = logging.getLogger(__name__)

_cached_allowlist: list[str] = []

# If win=1 scores >= this, commit immediately without trying longer windows.
_SINGLE_TOKEN_CONFIDENCE: int = 85

_SKILL_HEADING_RE = re.compile(
    r"^(skills?|technical\s+skills?|core\s+competenc\w+|technologies|"
    r"tools?\s+&?\s*technologies?|expertise|proficienc\w+)",
    re.I | re.M,
    )
_SKILLS_END_RE = re.compile(
    r"^\s*(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?|"
    r"experience|education|employment|projects?|awards?|publications?|"
    r"references?|volunteer|languages?|interests?|activities|summary|objective)",
    re.I | re.M,
    )

_SKILLS_SYSTEM = (
    "You are a precision resume parser. Your task is to identify professional skills. "
    "1. Look for the 'SKILLS' section in the text and extract those items first. "
    "2. Cross-reference with the 'WORK EXPERIENCE' to validate and find additional tools mentioned. "
    "3. IGNORE single-letter artifacts (like 'C', 'R') or two-letter acronyms (like 'AI', 'ML') "
    "UNLESS they are clearly written as distinct skills in the document. "
    "4. Format as a JSON array of strings. Standardize names (e.g., 'Microson' to 'Microsoft'). "
    "Output ONLY the JSON array."
)


def split_ocr_blob(line: str, allowlist: list[str]) -> list[str]:
    """
    Segment a collapsed single-line OCR blob into individual skill phrases.

    Algorithm (v7 — confidence-first):
      For each position i:
      1. Check win=1. If score >= _SINGLE_TOKEN_CONFIDENCE, commit immediately.
         This prevents a confident single token from being swallowed by a longer
         window matching a compound allowlist entry.
      2. Otherwise try all window lengths and commit to the highest scorer.
      3. If nothing clears the threshold, emit the raw token.
    """
    if not allowlist or not line.strip():
        return [line.strip()] if line.strip() else []

    sorted_allow = sorted(allowlist, key=lambda a: len(a.split()), reverse=True)
    norm_allow   = [(normalize(a), a) for a in sorted_allow]
    max_win      = max(len(a.split()) for a in sorted_allow) if sorted_allow else 1

    tokens = line.split()
    n      = len(tokens)
    result: list[str] = []
    i = 0

    while i < n:
        # Step 1: evaluate win=1 first
        w1_norm = normalize(tokens[i])
        w1_thr  = (BLOB_FUZZY_THRESHOLD - 10 if len(w1_norm) <= 5
                   else BLOB_FUZZY_THRESHOLD - 5 if len(w1_norm) <= 9
        else BLOB_FUZZY_THRESHOLD)

        w1_best_score, w1_best_val = 0, ""
        for akey, aval in norm_allow:
            s = max(fuzz.token_sort_ratio(w1_norm, akey), fuzz.ratio(w1_norm, akey))
            if s > w1_best_score:
                w1_best_score, w1_best_val = s, aval

        # High-confidence single token — commit without checking longer windows
        if w1_best_score >= _SINGLE_TOKEN_CONFIDENCE:
            log.debug(f"[BlobSplit] i={i} single-token '{tokens[i]}' -> '{w1_best_val}' ({w1_best_score})")
            result.append(w1_best_val)
            i += 1
            continue

        # Step 2: try all longer windows, keep the highest scorer
        best_score = w1_best_score if w1_best_score >= w1_thr else 0
        best_canon = w1_best_val   if w1_best_score >= w1_thr else None
        best_len   = 1             if w1_best_score >= w1_thr else 0

        for win_len in range(min(max_win, n - i), 1, -1):
            window_norm = normalize(" ".join(tokens[i : i + win_len]))
            if len(window_norm) < 3:
                continue
            wlen = len(window_norm)
            thr  = (BLOB_FUZZY_THRESHOLD - 10 if wlen <= 5
                    else BLOB_FUZZY_THRESHOLD - 5 if wlen <= 9
            else BLOB_FUZZY_THRESHOLD)

            win_best_score, win_best_val = 0, ""
            for akey, aval in norm_allow:
                s = max(fuzz.token_sort_ratio(window_norm, akey), fuzz.ratio(window_norm, akey))
                if s > win_best_score:
                    win_best_score, win_best_val = s, aval

            if win_best_score >= thr and win_best_score > best_score:
                best_score = win_best_score
                best_canon = win_best_val
                best_len   = win_len

        if best_canon:
            log.debug(f"[BlobSplit] i={i} multi-token '{best_canon}' (win={best_len}, score={best_score})")
            result.append(best_canon)
            i += best_len
        else:
            raw_tok = tokens[i].strip(".,;:()[]")
            if raw_tok:
                result.append(raw_tok)
            i += 1

    return result


def extract_skills_section(
        text:          str,
        allowlist:     list[str] | None = None,
        is_native_pdf: bool = False,
) -> str:
    """
    Return the skills section as a comma-separated string.
    Each entry is one complete skill phrase.
    """
    m = _SKILL_HEADING_RE.search(text)
    if not m:
        return ""
    rest  = text[m.end():]
    end_m = _SKILLS_END_RE.search(rest)
    block = rest[: end_m.start()].strip() if end_m else rest.strip()

    raw_lines = block.splitlines()

    if is_native_pdf:
        kept_lines = [line.strip() for line in raw_lines if line.strip()]
    else:
        kept_lines: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            words = stripped.split()
            if (kept_lines
                    and len(words) >= 3
                    and sum(1 for w in words if w[0].isupper()) >= 2):
                break
            kept_lines.append(stripped)

    if not kept_lines:
        return ""

    joined_block = "\n".join(kept_lines)
    _BULLET_RE   = re.compile(r"^\s*[\u2022\-\*\u25e6\u25aa]|\s*\d+\.\s+", re.M)
    entries: list[str] = []

    if "," in joined_block:
        for line in kept_lines:
            for part in re.split(r"\s*,\s*", line):
                if part.strip():
                    entries.append(part.strip())
    elif ";" in joined_block:
        for line in kept_lines:
            for part in re.split(r"\s*;\s*", line):
                if part.strip():
                    entries.append(part.strip())
    elif "|" in joined_block:
        for line in kept_lines:
            for part in line.split("|"):
                if part.strip():
                    entries.append(part.strip())
    elif _BULLET_RE.search(joined_block):
        for line in kept_lines:
            clean = re.sub(r"^\s*[\u2022\-\*\u25e6\u25aa]\s*|\s*\d+\.\s+", "", line).strip()
            if clean:
                entries.append(clean)
    elif len(kept_lines) > 1:
        entries = [line.strip() for line in kept_lines if line.strip()]
    else:
        blob = kept_lines[0] if kept_lines else ""
        if blob:
            effective_allowlist = allowlist if allowlist is not None else _cached_allowlist
            entries = (
                split_ocr_blob(blob, effective_allowlist)
                if effective_allowlist
                else [blob]
            )

    return ", ".join(entries)


def discover_skills_dynamically(
        resume_text:   str,
        jobs_corpus:   list[dict],
        extra_skills:  list[str] | None = None,
        is_native_pdf: bool = False,
) -> list[str]:
    """
    Stage 2 — skill extraction via Llama (HF Inference API).

    1. Build corpus allowlist.
    2. Build OcrCorrector.
    3. Extract skills section (is_native_pdf for v6 guard fix).
    4. Call LLM for extraction.
    5. Canonicalise LLM output against allowlist.
    6. Direct scan to recover anything the LLM missed.
    7. Fuzzy fallback if both produce nothing.
    """
    global _cached_allowlist

    allowlist: set[str] = set()
    for job in jobs_corpus:
        allowlist.update(job.get("required_skills", []))
    if extra_skills:
        allowlist.update(extra_skills)
    allowlist_list = sorted(allowlist)
    allow_norm     = {normalize(a): a for a in allowlist_list}
    sorted_pairs   = build_allow_norm_sorted(allowlist_list)
    corrector      = build_ocr_corrector(allowlist_list)

    _cached_allowlist = allowlist_list

    section = (
            extract_skills_section(
                resume_text,
                allowlist=allowlist_list,
                is_native_pdf=is_native_pdf,
            )
            or resume_text[:3000]
    )

    user_prompt = (
        f"Extract every technical skill, programming language, framework, tool, "
        f"methodology, and soft skill from this resume section.\n\n"
        f"The skills are listed as comma-separated tokens. Each token or short "
        f"phrase between commas is a distinct skill candidate — treat each one "
        f"individually and standardize any misspelled or OCR-garbled names "
        f"(e.g. 'Microson' -> 'Microsoft', 'Azuie' -> 'Azure').\n\n"
        f"Resume section (comma-separated):\n{section[:2500]}\n\n"
        f"Return ONLY a JSON array of skill name strings. No other text."
    )

    raw = llm_extract_list(_SKILLS_SYSTEM, user_prompt, "SKILLS")

    seen:      set[str]  = set()
    canonical: list[str] = []

    for skill in raw:
        corrected = apply_ocr_corrections(skill, corrector)
        canon     = canonicalize(
            corrected, allow_norm, threshold=75,
            _sorted_pairs=sorted_pairs, corrector=corrector,
        )
        if normalize(canon) == normalize(corrected) and normalize(canon) not in allow_norm:
            log.debug(f"[Skills/LLM] Dropped non-allowlist item: {skill!r}")
            continue
        cn = normalize(canon)
        if cn and cn not in seen:
            seen.add(cn)
            canonical.append(canon)

    for hit in direct_scan_section(
            section, sorted_pairs, threshold=82, corrector=corrector, allow_norm=allow_norm
    ):
        hn = normalize(hit)
        if hn and hn not in seen:
            seen.add(hn)
            canonical.append(hit)

    if not canonical:
        log.info("[Skills] LLM + direct scan produced nothing — fuzzy fallback.")
        canonical = fuzzy_scan(section, allowlist_list)

    log.info(f"[Skills] {len(canonical)} skills discovered.")
    return canonical