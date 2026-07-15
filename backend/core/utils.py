"""
core/utils.py
=============
Shared helpers: text normalisation, LLM API calls, and allowlist canonicalisation.
Everything here is stateless and has no imports from other app modules.

Canonicalisation notes
------------------------
normalize() strips punctuation, so distinct skills can collide once
normalized — e.g. "C#" and "C++" both normalize to "c". A plain
{normalized_key: canonical_value} dict lookup would then silently return
whichever entry was inserted last for any input that normalizes to "c",
swapping one skill for the other. Both canonicalize() and
direct_scan_section() guard against this: after a normalized-key lookup,
they verify the returned canonical value round-trips back to the raw input
(case-insensitive comparison). On a mismatch, they fall through to a direct
case-insensitive match against the raw allowlist strings instead, which
correctly resolves punctuation-significant skills like "C#", "C++", "F#",
and ".NET".

Both functions also score each allowlist key against a dot-replaced-with-space
variant (in addition to the original), which lets OCR-spaced "."-delimited
skills like "Node js" match an allowlist entry written as "Node.js" — the
canonical value returned is always the original allowlist form.
"""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from typing import Optional

from rapidfuzz import fuzz

log = logging.getLogger(__name__)


# ── Text normalisation ────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Strip punctuation/accents, collapse whitespace, lowercase."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip().lower()


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
      2. Exact normalized match, with a collision guard (see module docstring).
      3. Direct case-insensitive match on the raw string. Catches
         punctuation-significant skills like "C#", "C++", "F#", ".NET"
         whose normalized forms collide and would otherwise be swapped by fuzzy.
      4. Longest-match prefix containment.
      5. max(token_sort_ratio, ratio) fuzzy match.
      6. Return raw item unchanged when below threshold (fails-open).
    """
    from core.ocr_correction import apply_ocr_corrections
    corrected = apply_ocr_corrections(raw_item, corrector)
    sn = normalize(corrected)

    pairs = _sorted_pairs if _sorted_pairs is not None else list(allow_norm.items())

    # Fast path 1: exact normalized match, with a collision guard — verify
    # the returned canonical round-trips back to the raw corrected string
    # (case-insensitive). If normalize() collapsed two different skills to
    # the same key (e.g. "C#" and "C++" both -> "c"), allow_norm[sn] returns
    # whichever was inserted last, which may be wrong. On mismatch, skip
    # this path and fall through to Fast Path 2 below.
    corrected_lower = corrected.lower().strip()
    if sn in allow_norm:
        canon = allow_norm[sn]
        if canon.lower().strip() == corrected_lower:
            return canon
        else:
            log.debug(
                f"[Canonicalize] Fast Path 1 collision: allow_norm[{sn!r}]={canon!r} "
                f"!= corrected={corrected!r}, falling through to Fast Path 2"
            )

    # Fast path 2: direct case-insensitive match on the raw corrected string.
    # Must run before fuzzy scoring to prevent substitution of wrong-but-similar
    # entries whose normalized forms collide (e.g. "C#" and "C++" both -> "c").
    for akey, aval in pairs:
        if aval.lower().strip() == corrected_lower:
            log.debug(f"[Canonicalize] raw match '{corrected}' -> '{aval}'")
            return aval

    # Fuzzy path
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
        if prefix_eligible:
            score = 90
        else:
            # Also score against a dot-replaced-with-space variant of the
            # allowlist key, to catch OCR-spaced "."-delimited skills like
            # "Node js" -> "Node.js".
            akey_alt = normalize(aval.replace(".", " "))
            score = max(
                fuzz.token_sort_ratio(sn, akey),
                fuzz.token_sort_ratio(sn, akey_alt),
                fuzz.ratio(sn, akey),
                fuzz.ratio(sn, akey_alt),
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

    Uses the same collision guard and dot-variant fuzzy matching described
    in the module docstring, so punctuation-significant skills ("C#", "C++",
    "F#") and OCR-spaced dotted skills ("Node js" vs "Node.js") both resolve
    to the correct allowlist entry.
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
        corrected   = apply_ocr_corrections(entry, corrector)
        entry_norm  = normalize(corrected)
        entry_dedup = corrected.lower().strip()

        log.warning(
            f"[DScan] entry={entry!r} "
            f"corrected={corrected!r} "
            f"entry_norm={entry_norm!r} "
            f"entry_dedup={entry_dedup!r}"
        )

        if not entry_dedup or len(entry_dedup) < 2:
            log.warning(f"[DScan]  -> skipped (too short)")
            continue

        # Fast path 1: exact normalized key lookup, with a collision guard —
        # verify the returned canonical actually matches the raw entry
        # case-insensitively. If normalize() collapsed two different skills
        # to the same key (e.g. "C#" and "C++" both -> "c"), the dict returns
        # whichever was inserted last, which may be wrong. On a mismatch,
        # skip this path and let Fast Path 2 find the correct raw match.
        if entry_norm in _allow_norm:
            canon = _allow_norm[entry_norm]
            if canon.lower().strip() == entry_dedup:
                dedup_key = canon.lower().strip()
                log.warning(f"[DScan]  -> Fast Path 1 match: {canon!r}")
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    found.append(canon)
                continue
            else:
                log.warning(
                    f"[DScan]  -> Fast Path 1 collision: allow_norm[{entry_norm!r}]="
                    f"{canon!r} != entry_dedup={entry_dedup!r}, falling through to Fast Path 2"
                )

        # Fast path 2: direct case-insensitive match on the raw string.
        raw_matched = False
        for akey, aval in sorted_pairs:
            if aval.lower().strip() == entry_dedup:
                dedup_key = aval.lower().strip()
                log.warning(f"[DScan]  -> Fast Path 2 match: {aval!r}")
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    found.append(aval)
                raw_matched = True
                break
        if raw_matched:
            continue

        # Fuzzy path
        best_score: int = 0
        best_val:   str = ""
        best_akey:  str = ""
        best_aalt:  str = ""
        for akey, aval in sorted_pairs:
            akey_alt = normalize(aval.replace(".", " "))
            score = max(
                fuzz.token_sort_ratio(entry_norm, akey),
                fuzz.token_sort_ratio(entry_norm, akey_alt),
                fuzz.ratio(entry_norm, akey),
                fuzz.ratio(entry_norm, akey_alt),
            )
            if score > best_score:
                best_score = score
                best_val   = aval
                best_akey  = akey
                best_aalt  = akey_alt

        log.warning(
            f"[DScan]  -> fuzzy best: {best_val!r} "
            f"score={best_score} threshold={threshold} "
            f"akey={best_akey!r} akey_alt={best_aalt!r} "
            f"passed={best_score >= threshold}"
        )

        if best_score >= threshold and best_val:
            dedup_key = best_val.lower().strip()
            if dedup_key not in seen:
                seen.add(dedup_key)
                found.append(best_val)

    return found


def fuzzy_scan(text: str, allowlist: list[str], threshold: int = 88) -> list[str]:
    """Fallback: find allowlist items that appear in text via fuzzy partial match."""
    lower = text.lower()
    return [a for a in allowlist if fuzz.partial_ratio(a.lower(), lower) >= threshold]