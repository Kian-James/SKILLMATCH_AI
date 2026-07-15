"""
app/dependencies.py
===================
FastAPI dependency-injected singletons.

Warms up the SBERT model and jobs corpus at startup, and exposes both
as injectable dependencies.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends

from core.ocr_correction import get_sbert
from db import load_analyzer_data

log = logging.getLogger(__name__)


# ── Corpus + allowlists ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_analyzer_data() -> tuple[list, list, list]:
    """Load and cache (firestore_skills, jobs_corpus, firestore_certs)."""
    log.info("[dependencies] Loading analyzer data from Firestore ...")
    data = load_analyzer_data()
    log.info("[dependencies] Analyzer data loaded and cached.")
    return data


def invalidate_analyzer_data() -> None:
    """Clear the cached corpus."""
    get_analyzer_data.cache_clear()
    log.info("[dependencies] Analyzer data cache cleared.")


# ── SBERT model ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_sbert_model():
    """Warm up and cache the SentenceTransformer model."""
    log.info("[dependencies] Warming up SBERT model ...")
    sbert = get_sbert()
    log.info("[dependencies] SBERT model ready.")
    return sbert


# ── Startup hook ──────────────────────────────────────────────────────────────

def warm_up() -> None:
    """Pre-load everything before the first request arrives (call from lifespan)."""
    get_sbert_model()
    get_analyzer_data()
    log.info("[dependencies] Warm-up complete.")