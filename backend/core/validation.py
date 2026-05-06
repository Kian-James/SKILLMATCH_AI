"""
core/validation.py
==================
Stage 3 (NLP side) — fuzzy skill verification.

Computes NLP accuracy using greedy sliding-window token matching with
phrase-level consumption to prevent double-counting.
"""

from __future__ import annotations

import logging

from rapidfuzz import fuzz

from core.utils import normalize

log = logging.getLogger(__name__)


def _sliding_window_match(
        truth_norm:       str,
        disc_norms:       set,
        consumed_phrases: set,
        threshold:        int = 80,
) -> tuple[bool, set]:
    available = disc_norms - consumed_phrases

    if truth_norm in available:
        consumed_phrases.add(truth_norm)
        return True, consumed_phrases

    truth_tokens = truth_norm.split()
    prev_best: float = 0.0

    for end in range(1, len(truth_tokens) + 1):
        window = " ".join(truth_tokens[:end])

        if window in available:
            consumed_phrases.add(window)
            return True, consumed_phrases

        best_score: float = 0.0
        best_dn:    str   = ""
        for dn in available:
            score = max(fuzz.token_sort_ratio(window, dn), fuzz.ratio(window, dn))
            if score > best_score:
                best_score = score
                best_dn    = dn

        if best_score >= threshold and best_dn:
            consumed_phrases.add(best_dn)
            return True, consumed_phrases

        if end > 1 and best_score < prev_best:
            break

        prev_best = best_score

    return False, consumed_phrases


def fuzzy_skill_verify(
        discovered_list:   list[str],
        ground_truth_list: list[str],
        threshold:         int = 80,
        resume_text:       str = "",
) -> dict:
    """
    Compute NLP accuracy using greedy sliding-window phrase matching.

    Returns
    -------
    {"NLP_ACCURACY": "87.50%", "matched_skills": [...], "unmatched_skills": [...]}
    """
    if not ground_truth_list:
        return {"NLP_ACCURACY": "N/A", "matched_skills": [], "unmatched_skills": []}

    disc_norms:       set  = {normalize(s) for s in discovered_list}
    consumed_phrases: set  = set()
    matched:          list = []
    unmatched:        list = []

    for truth in ground_truth_list:
        tn = normalize(truth)
        hit, consumed_phrases = _sliding_window_match(tn, disc_norms, consumed_phrases, threshold)
        (matched if hit else unmatched).append(truth)

    pct = len(matched) / len(ground_truth_list) * 100
    return {
        "NLP_ACCURACY":     f"{pct:.2f}%",
        "matched_skills":   matched,
        "unmatched_skills": unmatched,
    }