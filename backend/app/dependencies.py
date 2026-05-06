"""
app/dependencies.py
===================
FastAPI dependency-injected singletons.

Warms up the SBERT model and loads the jobs corpus once at startup so the
first request doesn't pay the cold-start penalty. Both are exposed as
FastAPI dependencies so they can be injected into routes or overridden in
tests.
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
    """
    Load and cache (firestore_skills, jobs_corpus, firestore_certs).

    lru_cache ensures this only hits Firestore once per process lifetime.
    Call invalidate_analyzer_data() in tests or after re-seeding the corpus.
    """
    log.info("[dependencies] Loading analyzer data from Firestore ...")
    data = load_analyzer_data()
    log.info("[dependencies] Analyzer data loaded and cached.")
    return data


def invalidate_analyzer_data() -> None:
    """Clear the cached corpus — useful in tests or after re-seeding jobs."""
    get_analyzer_data.cache_clear()
    log.info("[dependencies] Analyzer data cache cleared.")


# ── SBERT model ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_sbert_model():
    """
    Warm up and cache the SentenceTransformer model.

    Delegates to core.ocr_correction.get_sbert() which already manages its
    own singleton — this wrapper just makes it injectable as a FastAPI dep.
    """
    log.info("[dependencies] Warming up SBERT model ...")
    sbert = get_sbert()
    log.info("[dependencies] SBERT model ready.")
    return sbert


# ── Startup hook ──────────────────────────────────────────────────────────────

def warm_up() -> None:
    """
    Call this from the FastAPI lifespan handler to pre-load everything
    before the first request arrives.

    Example usage in app/main.py:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            warm_up()
            yield

        app = FastAPI(lifespan=lifespan, ...)
    """
    get_sbert_model()
    get_analyzer_data()
    log.info("[dependencies] Warm-up complete.")