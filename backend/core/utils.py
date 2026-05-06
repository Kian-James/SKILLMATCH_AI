"""
core/utils.py
=============
Shared helpers: text normalisation, LLM API calls, and allowlist canonicalisation.
Everything here is stateless and has no imports from other app modules.
"""

from __future__ import annotations

import json
import logging
import re
import string
import unicodedata
from typing import Optional

import requests
from rapidfuzz import fuzz

from config import HF_TOKEN, HF_API_URL, HF_MODEL, MAX_RETRIES

log = logging.getLogger(__name__)


# ── Text normalisation ────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Strip punctuation/accents, collapse whitespace, lowercase."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip().lower()


# ── HuggingFace Inference API ─────────────────────────────────────────────────

def call_llama(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """POST to the HF Inference API. Returns raw assistant message string."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"HF API returned {response.status_code}: {response.text[:300]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def clean_json_fence(text: str) -> str:
    """Strip markdown code fences the model may emit."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def llm_extract_list(system: str, user: str, label: str) -> list[str]:
    """Call Llama and parse the returned JSON array. Returns [] on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw   = call_llama(system, user, max_tokens=512)
            raw   = clean_json_fence(raw)
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end <= start:
                raise ValueError("No JSON array found in response.")
            items = json.loads(raw[start:end])
            if isinstance(items, list):
                return [str(i).strip() for i in items if str(i).strip()]
        except Exception as exc:
            log.warning(f"[LLM/{label}] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt == MAX_RETRIES:
                log.warning(f"[LLM/{label}] All retries exhausted — using fallback.")
    return []


# ── Allowlist canonicalisation ────────────────────────────────────────────────

def build_allow_norm_sorted(allowlist_items: list[str]) -> list[tuple[str, str]]:
    """
    Return (normalized_key, canonical_value) pairs sorted by descending word
    count so that longer allowlist entries are evaluated first.
    """
    pairs = [(normalize(a), a) for a in allowlist_items]
    pairs.sort(key=lambda p: len(p[0].split()), reverse=True)
    return pairs


def canonicalize(
        raw_item:      str,
        allow_norm:    dict[str, str],
        threshold:     int = 75,
        _sorted_pairs: list[tuple[str, str]] | None = None,
        corrector=None,  # OcrCorrector | None — avoid circular import
) -> str:
    """
    Map a raw extracted string to its canonical allowlist form.

    Priority:
      1. Embedding-based OCR correction.
      2. Exact normalized match.
      3. Longest-match prefix containment.
      4. max(token_sort_ratio, ratio) fuzzy match.
      5. Return raw item unchanged when below threshold (fails-open).
    """
    from core.ocr_correction import apply_ocr_corrections
    corrected = apply_ocr_corrections(raw_item, corrector)
    sn = normalize(corrected)

    if sn in allow_norm:
        return allow_norm[sn]

    pairs = _sorted_pairs if _sorted_pairs is not None else list(allow_norm.items())

    best_score: int = 0
    best_val:   str = raw_item
    sn_word_count   = len(sn.split())

    for akey, aval in pairs:
        akey_word_count = len(akey.split())
        prefix_eligible = (
                len(sn) >= 4
                and akey.startswith(sn)
                and sn not in allow_norm
                and akey_word_count <= sn_word_count * 2
        )
        score = 90 if prefix_eligible else max(
            fuzz.token_sort_ratio(sn, akey),
            fuzz.ratio(sn, akey),
        )
        if score > best_score:
            best_score = score
            best_val   = aval
            if score == 90:
                break

    return best_val if best_score >= threshold else raw_item


def direct_scan_section(
        section_text: str,
        sorted_pairs: list[tuple[str, str]],
        threshold:    int = 82,
        corrector=None,
        allow_norm:   dict[str, str] | None = None,
) -> list[str]:
    """
    Atomic entry scan: split section on commas (or lines), match each entry
    against the allowlist as a whole unit — no sub-phrase windowing.
    """
    from core.ocr_correction import apply_ocr_corrections

    found: list[str] = []
    seen:  set[str]  = set()
    _allow_norm = allow_norm or {}

    raw_entries = (
        [e.strip() for e in section_text.split(",") if e.strip()]
        if "," in section_text
        else [l.strip() for l in section_text.splitlines() if l.strip()]
    )

    for entry in raw_entries:
        corrected  = apply_ocr_corrections(entry, corrector)
        entry_norm = normalize(corrected)

        if not entry_norm or len(entry_norm) < 3:
            continue

        if entry_norm in _allow_norm:
            canon = _allow_norm[entry_norm]
            cn = normalize(canon)
            if cn not in seen:
                seen.add(cn)
                found.append(canon)
            continue

        best_score: int = 0
        best_val:   str = ""
        for akey, aval in sorted_pairs:
            score = max(fuzz.token_sort_ratio(entry_norm, akey), fuzz.ratio(entry_norm, akey))
            if score > best_score:
                best_score = score
                best_val   = aval

        if best_score >= threshold and best_val:
            cn = normalize(best_val)
            if cn not in seen:
                seen.add(cn)
                found.append(best_val)

    return found


def fuzzy_scan(text: str, allowlist: list[str], threshold: int = 88) -> list[str]:
    """Fallback: find allowlist items that appear in text via fuzzy partial match."""
    lower = text.lower()
    return [a for a in allowlist if fuzz.partial_ratio(a.lower(), lower) >= threshold]