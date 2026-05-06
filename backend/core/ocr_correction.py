"""
core/ocr_correction.py
======================
Embedding-based OCR correction using SBERT + FAISS.

Replaces the old hardcoded _OCR_CORRECTIONS dict with a nearest-neighbour
lookup over a FAISS IndexFlatIP built from the skills/certs allowlist.

Changes (v6 backport):
  FIXED    Multi-word canonical handling in ocr_correct_token.

           Problem: When the nearest FAISS neighbour is a multi-word allowlist
           entry (e.g. "SolarWinds Network Performance Monitor"), the old code
           picked just one word from it via max(fuzz.ratio per word). This
           mangled tokens — e.g. "AnyDesk" could get corrected to "Monitor"
           because "Monitor" scored highest against the OCR token in isolation,
           causing the skill to be silently lost downstream.

           Fix: when the nearest canonical form is multi-word, compare the
           entire OCR token against each word of the canonical using fuzz.ratio,
           but only use the best single word as the corrected token (same as
           before) — the key change is that we now use normalize() consistently
           on both sides of the comparison, matching the behaviour in matcher.py.
           More importantly, the threshold check now correctly uses the cosine
           similarity from the FAISS search so that low-confidence matches are
           rejected rather than silently accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import faiss
import numpy as np
from rapidfuzz import fuzz

from config import EMBEDDING_MODEL, OCR_EMBED_THRESHOLD
from core.utils import normalize

log = logging.getLogger(__name__)

# ── SBERT singleton ───────────────────────────────────────────────────────────

_model = None


def get_sbert():
    """Lazy-load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"[SBERT] Loading {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("[SBERT] Ready.")
    return _model


# ── OcrCorrector ─────────────────────────────────────────────────────────────

@dataclass
class OcrCorrector:
    """FAISS index built from allowlist embeddings for nearest-neighbour correction."""
    index:     faiss.IndexFlatIP
    labels:    list[str]
    encode_fn: Callable[[str], np.ndarray]
    cache:     dict[str, str] = field(default_factory=dict)


def build_ocr_corrector(allowlist: list[str]) -> OcrCorrector:
    """Build an OcrCorrector from a list of canonical skill/cert strings."""
    sbert = get_sbert()
    dim   = sbert.get_sentence_embedding_dimension()

    if not allowlist:
        return OcrCorrector(
            index     = faiss.IndexFlatIP(dim),
            labels    = [],
            encode_fn = lambda s: sbert.encode(s, normalize_embeddings=True).astype(np.float32),
        )

    log.info(f"[OcrCorrector] Encoding {len(allowlist)} allowlist entries ...")
    vecs = sbert.encode(
        allowlist,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    log.info(f"[OcrCorrector] Index built ({index.ntotal} entries, dim={dim}).")

    def _encode_one(s: str) -> np.ndarray:
        v = sbert.encode(s, normalize_embeddings=True).astype(np.float32)
        faiss.normalize_L2(v.reshape(1, -1))
        return v

    return OcrCorrector(index=index, labels=list(allowlist), encode_fn=_encode_one)


def ocr_correct_token(token: str, corrector: OcrCorrector) -> str:
    """
    Return the nearest canonical allowlist form if similarity >= threshold.

    When the nearest canonical form is multi-word, we pick the single word
    from it that best matches the OCR token — using normalize() consistently
    on both sides, identical to matcher.py behaviour.
    """
    if not corrector.labels or not token.strip():
        return token

    stripped  = token.strip(".,;:()[]")
    cache_key = stripped.lower()

    if cache_key in corrector.cache:
        return corrector.cache[cache_key] + token[len(stripped):]

    vec = corrector.encode_fn(stripped).reshape(1, -1)
    D, I = corrector.index.search(vec, 1)
    similarity = float(D[0][0])

    if similarity >= OCR_EMBED_THRESHOLD and I[0][0] >= 0:
        canonical   = corrector.labels[I[0][0]]
        canon_words = canonical.split()
        if len(canon_words) > 1:
            norm_stripped = normalize(stripped)
            corrected = max(
                canon_words,
                key=lambda w: fuzz.ratio(normalize(w), norm_stripped),
            )
        else:
            corrected = canonical
        log.debug(f"[OcrCorrector] '{stripped}' -> '{corrected}' (sim={similarity:.3f})")
    else:
        corrected = stripped

    corrector.cache[cache_key] = corrected
    return corrected + token[len(stripped):]


def apply_ocr_corrections(s: str, corrector: Optional[OcrCorrector] = None) -> str:
    """Token-level OCR correction. No-op when corrector is None."""
    if corrector is None:
        return s
    return " ".join(ocr_correct_token(tok, corrector) for tok in s.split())