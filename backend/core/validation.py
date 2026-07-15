"""
core/validation.py
==================
Stage 3 (NLP side) — semantic skill/cert verification.

Computes NLP accuracy using greedy sliding-window phrase matching, scored
via SBERT cosine similarity. This replaces the previous rapidfuzz
token_sort_ratio/ratio scorer with embedding-based comparison, so that
discovered items that are worded differently from the ground-truth string
but mean the same thing (e.g. "Amazon Web Services" vs "AWS") are recognised
on meaning rather than surface character overlap. Phrase-level consumption
still prevents double-counting, same as before.

The function is agnostic to what kind of list it's given — it is used for
both discovered skills and discovered certifications against their
respective ground-truth lists.
"""

from __future__ import annotations

import logging

import numpy as np

from core.ocr_correction import get_sbert
from core.utils import normalize

log = logging.getLogger(__name__)

# Cosine similarity floor for a semantic match (0-1 scale). Matches the
# convention used elsewhere in the pipeline for SBERT-based matching
# (see extractor_base._SBERT_COSINE_THRESHOLD).
_SEMANTIC_THRESHOLD: float = 0.80


def _encode(texts: list[str]) -> np.ndarray:
    """Batch-encode normalized embeddings via the shared SBERT model."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    sbert = get_sbert()
    return sbert.encode(texts, normalize_embeddings=True).astype(np.float32)


def _sliding_window_match(
        truth_norm:       str,
        disc_vec_map:     dict[str, np.ndarray],
        consumed_phrases: set,
        threshold:        float = _SEMANTIC_THRESHOLD,
) -> tuple[bool, set]:
    """
    Try to match truth_norm against the not-yet-consumed discovered phrases,
    growing the truth phrase token-by-token and scoring each growing window
    against every available discovered phrase via SBERT cosine similarity
    (embeddings are pre-computed once per discovered phrase and reused
    across every ground-truth item via disc_vec_map — only the truth window
    itself needs encoding here).
    """
    available = {k: v for k, v in disc_vec_map.items() if k not in consumed_phrases}

    if truth_norm in available:
        consumed_phrases.add(truth_norm)
        return True, consumed_phrases

    if not available:
        return False, consumed_phrases

    available_names = list(available.keys())
    available_vecs  = np.vstack(list(available.values()))

    truth_tokens = truth_norm.split()
    prev_best: float = 0.0

    for end in range(1, len(truth_tokens) + 1):
        window = " ".join(truth_tokens[:end])

        if window in available:
            consumed_phrases.add(window)
            return True, consumed_phrases

        window_vec = _encode([window])[0]
        sims       = available_vecs @ window_vec
        best_idx   = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_dn    = available_names[best_idx]

        if best_score >= threshold:
            consumed_phrases.add(best_dn)
            return True, consumed_phrases

        if end > 1 and best_score < prev_best:
            break

        prev_best = best_score

    return False, consumed_phrases


def semantic_verify(
        discovered_list:   list[str],
        ground_truth_list: list[str],
        threshold:         float = _SEMANTIC_THRESHOLD,
        resume_text:       str = "",
) -> dict:
    """
    Compute NLP accuracy using greedy sliding-window phrase matching, scored
    by SBERT cosine similarity rather than string-level fuzzy ratio.

    NOTE: kept the name `fuzzy_skill_verify` (rather than e.g.
    `semantic_verify`) purely for call-site compatibility — scoring.py
    imports it under this name. The scoring mechanism itself is now
    semantic, not fuzzy-string-based. `threshold` is on the 0-1 cosine
    scale now (was 0-100 on the rapidfuzz scale).

    Works for both discovered skills and discovered certifications — the
    function doesn't care what kind of items are in the lists.

    Returns
    -------
    {"NLP_ACCURACY": "87.50%", "matched_skills": [...], "unmatched_skills": [...]}
    """
    if not ground_truth_list:
        return {"NLP_ACCURACY": "N/A", "matched_skills": [], "unmatched_skills": []}

    disc_list: list[str] = sorted({normalize(s) for s in discovered_list})
    disc_vecs            = _encode(disc_list)
    disc_vec_map: dict[str, np.ndarray] = dict(zip(disc_list, disc_vecs))

    consumed_phrases: set  = set()
    matched:          list = []
    unmatched:        list = []

    for truth in ground_truth_list:
        tn = normalize(truth)
        hit, consumed_phrases = _sliding_window_match(tn, disc_vec_map, consumed_phrases, threshold)
        (matched if hit else unmatched).append(truth)

    pct = len(matched) / len(ground_truth_list) * 100
    return {
        "NLP_ACCURACY":     f"{pct:.2f}%",
        "matched_skills":   matched,
        "unmatched_skills": unmatched,
    }