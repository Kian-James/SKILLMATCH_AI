"""
utils/cache_manager.py
Lightweight disk-based cache manager for embeddings and API responses.
Wraps diskcache with a simple get/set/invalidate interface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache
from loguru import logger

from config import CACHE_DIR

# Main embedding cache (shared with embedding_engine.py)
_embedding_cache = diskcache.Cache(str(CACHE_DIR / "embeddings"))

# Secondary cache for LLM responses (skill resources, etc.)
_llm_cache = diskcache.Cache(str(CACHE_DIR / "llm_responses"), size_limit=50_000_000)  # 50MB


def make_key(*parts: str) -> str:
    """Create a stable cache key from multiple string parts."""
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()


# ── Embedding Cache ────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float] | None:
    key = make_key("emb", text)
    try:
        return _embedding_cache.get(key)
    except Exception as e:
        logger.warning(f"Embedding cache read error: {e}")
        return None


def set_embedding(text: str, vector: list[float]) -> None:
    key = make_key("emb", text)
    try:
        _embedding_cache.set(key, vector)
    except Exception as e:
        logger.warning(f"Embedding cache write error: {e}")


# ── LLM Response Cache ────────────────────────────────────────────────────────

def get_llm_response(prompt_key: str) -> Any | None:
    """
    Retrieve a cached LLM response.
    Useful for skill resources — same skill shouldn't call the LLM twice.
    """
    try:
        return _llm_cache.get(prompt_key)
    except Exception as e:
        logger.warning(f"LLM cache read error: {e}")
        return None


def set_llm_response(prompt_key: str, response: Any, ttl: int = 86400) -> None:
    """
    Cache an LLM response with a TTL (default 24 hours).
    """
    try:
        _llm_cache.set(prompt_key, response, expire=ttl)
    except Exception as e:
        logger.warning(f"LLM cache write error: {e}")


def invalidate_llm_cache() -> None:
    """Clear all cached LLM responses (e.g. after model change)."""
    _llm_cache.clear()
    logger.info("LLM response cache cleared.")


def cache_stats() -> dict:
    """Return cache size info for monitoring."""
    return {
        "embedding_count": len(_embedding_cache),
        "llm_response_count": len(_llm_cache),
        "embedding_size_mb": round(_embedding_cache.volume() / 1_000_000, 2),
        "llm_size_mb": round(_llm_cache.volume() / 1_000_000, 2),
    }