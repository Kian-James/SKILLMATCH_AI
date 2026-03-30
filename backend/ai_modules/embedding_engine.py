"""
ai_modules/embedding_engine.py
Generates dense vector embeddings using sentence-transformers (RoBERTa).
Includes disk-based caching to avoid redundant inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Union

import numpy as np
import diskcache
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import ROBERTA_MODEL_ID, CACHE_DIR

# ─── Disk Cache ───────────────────────────────────────────────────────────────
_cache = diskcache.Cache(str(CACHE_DIR / "embeddings"))

# ─── Model Singleton ──────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {ROBERTA_MODEL_ID}")
        _model = SentenceTransformer(ROBERTA_MODEL_ID)
        logger.info("Embedding model loaded.")
    return _model


def _cache_key(text: str) -> str:
    """SHA-256 hash of text as cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingEngine:
    """
    Wraps sentence-transformers RoBERTa to produce L2-normalized embeddings.
    Cache-first: if a text's embedding is cached, skip inference.
    """

    def __init__(self):
        self.model = _get_model()

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        Returns normalized 1-D float32 numpy array of shape (embedding_dim,).
        """
        key = _cache_key(text)
        if key in _cache:
            return np.array(_cache[key], dtype=np.float32)

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # L2 normalization — cosine sim = dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        _cache[key] = embedding.tolist()  # store as list for JSON serialization
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        Returns 2-D array of shape (n_texts, embedding_dim).
        Checks cache first; only runs inference on cache misses.
        """
        results = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Partition into cached vs not cached
        for i, text in enumerate(texts):
            key = _cache_key(text)
            if key in _cache:
                results.append((i, np.array(_cache[key], dtype=np.float32)))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append((i, None))  # placeholder

        # Batch-encode uncached
        if uncached_texts:
            logger.debug(f"Embedding {len(uncached_texts)} uncached texts...")
            new_embeddings = self.model.encode(
                uncached_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=len(uncached_texts) > 10,
            )
            for j, (orig_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                emb = new_embeddings[j].astype(np.float32)
                _cache[_cache_key(text)] = emb.tolist()
                results[orig_idx] = (orig_idx, emb)

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in results])

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized vectors.
        Since we normalize on embed(), this is just the dot product.
        """
        return float(np.dot(vec_a, vec_b))

    def embed_skill_list(self, skills: list[str]) -> np.ndarray:
        """
        Embed a list of skills as a single concatenated query vector.
        The mean of individual skill embeddings represents overall skill profile.
        """
        if not skills:
            # Return zero vector of correct dimension
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(dim, dtype=np.float32)

        embeddings = self.embed_batch(skills)
        mean_embedding = embeddings.mean(axis=0)
        # Re-normalize the mean vector
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        return mean_embedding.astype(np.float32)
