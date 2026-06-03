"""
core/utils.py
=============
Shared helpers: text normalisation, LLM API calls, and allowlist canonicalisation.
Everything here is stateless and has no imports from other app modules.

Changes (v22): fixed direct_scan_section silently dropping punctuation-significant
           skills like "C#", "C++", "F#" via the entry_norm length guard.

           Root cause: normalize("C#") -> "c" (length 1), normalize("C++") -> "c"
           (length 1). The guard `len(entry_norm) < 3` fired for both, skipping
           them entirely before fuzzy matching even had a chance to run.
           The internal dedup `seen` set also used normalize(canon) as its key,
           causing the same collision problem fixed in skills.py v21.

           Fix:
             - Length guard now checks len(entry_dedup) where entry_dedup is
               corrected.lower().strip(). "c#" has length 2, "c++" has length 3
               — both clear the revised threshold (< 2 for the raw form).
             - Internal seen set switched to canon.lower().strip() / best_val.lower().strip()
               so "c#" and "c++" each occupy a unique dedup slot.
             - normalize(entry_norm) is still used for the actual fuzzy scoring
               since the allowlist keys are also normalized — they match correctly
               even though normalize is lossy (both sides lose the same chars).

Changes (v23): fixed canonicalize() substituting a wrong-but-similar allowlist
           entry for punctuation-significant skills whose normalized forms collide.

           Root cause: normalize("C#") -> "c" and normalize("C++") -> "c" are
           identical. canonicalize() fell through to fuzzy scoring and picked
           whichever of "C#" / "C++" appeared first in sorted_pairs (determined
           by Firestore sort order), silently replacing one with the other.

           Fix: added a direct case-insensitive match on the raw corrected string
           against allowlist canonical values BEFORE fuzzy scoring. This runs
           after the normalized exact-match fast path but before any fuzzy logic,
           so "C#" matches the "C#" allowlist entry exactly and is never swapped
           for "C++". Only falls through to fuzzy when no exact raw match exists.

Changes (v24): fixed both canonicalize() and direct_scan_section() missing
           OCR-spaced "."-delimited skills like "Node js", "Express js",
           "Vue js" when the allowlist entry uses a dot ("Node.js",
           "Express.js", "Vue.js").

           Root cause: normalize("Node js") -> "node js" and
           normalize("Node.js") -> "nodejs" (dot stripped by punctuation
           removal). These produce different strings so fuzz.ratio /
           token_sort_ratio scores fall below their respective thresholds
           (~80 vs 82 for direct_scan, ~73 vs 75 for canonicalize),
           causing the hit to be silently dropped in BOTH the LLM path
           (which calls canonicalize()) and the direct_scan_section path.

           Fix: in the fuzzy path of both functions, each allowlist key is
           also evaluated with its dots replaced by spaces (akey_alt). The
           scorer takes the maximum across all four comparisons:

               akey_alt = normalize(aval.replace(".", " "))
               score = max(
                   fuzz.token_sort_ratio(query, akey),
                   fuzz.token_sort_ratio(query, akey_alt),
                   fuzz.ratio(query, akey),
                   fuzz.ratio(query, akey_alt),
               )

           This is purely additive — the canonical value returned is always
           the original allowlist form (e.g. "Node.js"), not the dot-free
           variant. All existing fast paths (exact normalized, raw
           case-insensitive) are unaffected in both functions.

Changes (v26): fixed canonicalize() Fast Path 1 returning the wrong canonical
           for punctuation-significant skills whose normalized forms collide in
           allow_norm — the identical collision that was fixed in
           direct_scan_section in v25, but canonicalize() was missed.

           Root cause: normalize("C#") -> "c" and normalize("C++") -> "c".
           allow_norm["c"] returns whichever was inserted last. When the LLM
           is unavailable and direct_scan_section finds "C#" and passes it to
           canonicalize(), Fast Path 1 hits allow_norm["c"] = "C++" and returns
           "C++", which deduplicates against the wrong skill and "C#" is lost.

           Fix: same round-trip guard as v25 — after the dict lookup, verify
           allow_norm[sn].lower().strip() == corrected.lower().strip(). On
           mismatch (collision) skip Fast Path 1 and fall through to Fast Path 2
           which iterates sorted_pairs and finds the correct raw match.

           Before: canonicalize("C#") -> allow_norm["c"] = "C++" -> "C++"  ❌
           After:  canonicalize("C#") -> allow_norm["c"] = "C++"
                                      -> "c++" != "c#" -> skip Fast Path 1
                                      -> Fast Path 2: aval="C#" == "c#" -> "C#" ✓

Changes (v25): fixed direct_scan_section Fast Path 1 (exact normalized key
           lookup) returning the wrong canonical value for punctuation-significant
           skills whose normalized forms collide in allow_norm.

           Root cause: normalize("C#") -> "c" and normalize("C++") -> "c".
           allow_norm is a plain dict built as {normalize(a): a for a in
           allowlist_items}. When both "C#" and "C++" are in the allowlist,
           whichever was inserted last wins the key "c". So when entry="C#"
           arrives, entry_norm="c", Fast Path 1 hits allow_norm["c"] and
           returns "C++" (the last-inserted winner), silently swapping the
           skill identity.

           Fix: Fast Path 1 now validates that the canonical value returned
           by the dict lookup actually round-trips to the same raw entry via
           a case-insensitive comparison (aval.lower().strip() == entry_dedup).
           If it does not match (collision case), Fast Path 1 is skipped and
           execution falls through to Fast Path 2, which iterates sorted_pairs
           and does a direct case-insensitive match on aval — correctly finding
           "C#" in the allowlist and returning it without any substitution.

           Before: entry="C#" -> entry_norm="c" -> allow_norm["c"]="C++" -> WRONG
           After:  entry="C#" -> entry_norm="c" -> allow_norm["c"]="C++"
                              -> "c++" != "c#" -> skip Fast Path 1
                              -> Fast Path 2: aval="C#", "c#"=="c#" -> "C#" ✓
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
      2. Exact normalized match with collision guard (v26).
      3. Direct case-insensitive match on the raw string (v23).
         Catches punctuation-significant skills like "C#", "C++", "F#", ".NET"
         whose normalized forms collide and would otherwise be swapped by fuzzy.
      4. Longest-match prefix containment.
      5. max(token_sort_ratio, ratio) fuzzy match.
      6. Return raw item unchanged when below threshold (fails-open).
    """
    from core.ocr_correction import apply_ocr_corrections
    corrected = apply_ocr_corrections(raw_item, corrector)
    sn = normalize(corrected)

    pairs = _sorted_pairs if _sorted_pairs is not None else list(allow_norm.items())

    # Fast path 1: exact normalized match.
    # v26 collision guard: verify the returned canonical round-trips back to
    # the raw corrected string via case-insensitive comparison. If normalize()
    # collapsed two different skills to the same key (e.g. "C#" and "C++" both
    # -> "c"), allow_norm["c"] returns whichever was inserted last — which may
    # be wrong. On mismatch, skip Fast Path 1 and fall through to Fast Path 2
    # which finds the correct raw match via direct case-insensitive iteration.
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

    # Fast path 2 (v23): direct case-insensitive match on the raw corrected string.
    # Must run BEFORE fuzzy scoring to prevent substitution of wrong-but-similar
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
            # v24: also score against a dot-replaced-with-space variant of the
            # allowlist key to catch OCR-spaced "."-delimited skills like
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

    v22: fixed silent drop of punctuation-significant skills ("C#", "C++", "F#").

    v24: fixed missing hits for OCR-spaced "."-delimited skills ("Node js",
    "Express js"). In the fuzzy path each allowlist key is now scored against
    both its original form and a dot-replaced-with-space variant (akey_alt),
    taking the max across all four comparisons.

    v25: fixed Fast Path 1 returning the wrong canonical for skills whose
    normalized forms collide in allow_norm (e.g. "C#" and "C++" both normalize
    to "c"). The dict lookup allow_norm["c"] returns whichever was inserted
    last, silently swapping "C#" for "C++". Fix: after the dict lookup, verify
    the returned canonical round-trips back to entry_dedup via case-insensitive
    comparison. On mismatch (collision) skip Fast Path 1 and fall through to
    Fast Path 2, which iterates sorted_pairs and finds the exact raw match.
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
            f"[DEBUG/DScan] entry={entry!r} "
            f"corrected={corrected!r} "
            f"entry_norm={entry_norm!r} "
            f"entry_dedup={entry_dedup!r}"
        )

        if not entry_dedup or len(entry_dedup) < 2:
            log.warning(f"[DEBUG/DScan]  -> SKIPPED (too short)")
            continue

        # Fast path 1: exact normalized key lookup.
        # v25 collision guard: verify the returned canonical actually matches
        # the raw entry case-insensitively. If normalize() collapsed two
        # different skills to the same key (e.g. "C#" and "C++" both -> "c"),
        # the dict returns whichever was inserted last — which may be wrong.
        # On a mismatch, skip Fast Path 1 and let Fast Path 2 find the correct
        # raw match via direct case-insensitive comparison on sorted_pairs.
        if entry_norm in _allow_norm:
            canon = _allow_norm[entry_norm]
            if canon.lower().strip() == entry_dedup:
                # Genuine exact match — commit immediately.
                dedup_key = canon.lower().strip()
                log.warning(f"[DEBUG/DScan]  -> FAST PATH 1 match: {canon!r}")
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    found.append(canon)
                continue
            else:
                # Collision — normalized key matched but canonical differs from
                # the raw entry. Fall through to Fast Path 2 for a correct lookup.
                log.warning(
                    f"[DEBUG/DScan]  -> FAST PATH 1 collision: allow_norm[{entry_norm!r}]="
                    f"{canon!r} != entry_dedup={entry_dedup!r}, falling through to Fast Path 2"
                )

        # Fast path 2 (v23): direct case-insensitive match on the raw string.
        raw_matched = False
        for akey, aval in sorted_pairs:
            if aval.lower().strip() == entry_dedup:
                dedup_key = aval.lower().strip()
                log.warning(f"[DEBUG/DScan]  -> FAST PATH 2 match: {aval!r}")
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    found.append(aval)
                raw_matched = True
                break
        if raw_matched:
            continue

        # Fuzzy path (v24)
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
            f"[DEBUG/DScan]  -> FUZZY best: {best_val!r} "
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