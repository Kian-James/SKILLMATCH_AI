"""
ai_modules/faiss_index.py
Builds and queries a FAISS index over job embeddings from Firestore.
Index is persisted to disk and rebuilt when the job dataset changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from ai_modules.embedding_engine import EmbeddingEngine
from services.firebase_service import fetch_all_jobs


class FAISSIndex:
    """
    Maintains a FAISS flat inner-product index over job skill profiles.
    Since embeddings are L2-normalized, inner product = cosine similarity.
    """

    def __init__(self):
        self.engine = EmbeddingEngine()
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []  # parallel list of job info per vector
        self._embedding_dim: int | None = None

    async def load_or_build(self) -> None:
        """
        Load persisted index from disk if available; otherwise build from Firestore.
        """
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
            logger.info("Loading FAISS index from disk...")
            self._load_from_disk()
        else:
            logger.info("No FAISS index on disk — building from Firestore...")
            await self.rebuild()

    def _load_from_disk(self) -> None:
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "r") as f:
            self.metadata = json.load(f)
        self._embedding_dim = self.index.d
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, dim={self._embedding_dim}")

    def _save_to_disk(self) -> None:
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)
        logger.info("FAISS index saved to disk.")

    async def rebuild(self) -> None:
        """
        Fetch all jobs from Firestore, embed each job's skill profile,
        and add to the FAISS index.
        """
        jobs = fetch_all_jobs()
        if not jobs:
            logger.warning("No jobs in Firestore — FAISS index is empty.")
            dim = self.engine.model.get_sentence_embedding_dimension()
            self._embedding_dim = dim
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []
            return

        logger.info(f"Building FAISS index for {len(jobs)} jobs...")

        # Each job's embedding = mean of its skill embeddings
        # We combine job_title + skills into one representative text for richer context
        texts = []
        for job in jobs:
            combined = job["job_title"] + " " + " ".join(job.get("skills", []))
            texts.append(combined)

        embeddings = self.engine.embed_batch(texts)  # shape: (n_jobs, dim)
        self._embedding_dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self._embedding_dim)
        self.index.add(embeddings)

        self.metadata = [
            {
                "job_title": job["job_title"],
                "skills": job.get("skills", []),
                "certifications": job.get("certifications", []),
                "doc_id": job.get("id", ""),
            }
            for job in jobs
        ]

        self._save_to_disk()
        logger.info(f"FAISS index built: {self.index.ntotal} vectors.")

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5,
    ) -> list[dict]:
        """
        Find the top-K most similar jobs for a given query embedding.
        Returns a list of dicts with job info + similarity score.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no results.")
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            job_info = self.metadata[idx].copy()
            # Store raw cosine similarity for internal ranking only
            # The displayed match_score is computed from actual skill overlap
            # in the resume_service after FAISS retrieval
            job_info["_cosine_sim"] = round(max(float(score), 0.0), 4)
            job_info["match_score"] = round(max(float(score), 0.0) * 100, 1)
            results.append(job_info)

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results