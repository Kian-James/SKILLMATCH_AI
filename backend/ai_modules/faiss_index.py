"""
ai_modules/faiss_index.py
Builds and queries a FAISS index over job embeddings from Firestore.
Index is persisted to disk and rebuilt when the job dataset changes.

PERFORMANCE REDESIGN:
  FAISS is used ONLY for fast candidate retrieval (semantic search), NOT scoring.
  The final match score is computed entirely from skill overlap in resume_service.py.

  Key optimization — pre-computed skill embeddings:
    At build time, each job's required skill embeddings are computed once and stored
    in metadata. During resume analysis, resume_service retrieves these pre-computed
    vectors directly — eliminating all per-request embedding calls for job skills.

  Before: 5 jobs × ~12 skills each = up to 65 embed() calls → ~130s on Render.
  After:  pure numpy dot products against pre-computed vectors → <0.1s scoring.
          Only user skills are embedded at request time (once, then disk-cached).
"""

from __future__ import annotations

import json

import faiss
import numpy as np
from loguru import logger

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from ai_modules.embedding_engine import EmbeddingEngine
from services.firebase_service import fetch_all_jobs


class FAISSIndex:
    """
    FAISS flat inner-product index over job skill profiles.
    FAISS role: fast top-K candidate retrieval ONLY.
    Scoring role: delegated to resume_service._compute_unified_score()
                  which uses pre-computed skill embeddings from metadata.
    """

    def __init__(self):
        self.engine = EmbeddingEngine()
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self._embedding_dim: int | None = None

    async def load_or_build(self) -> None:
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

        # Restore pre-computed skill embeddings: JSON lists → numpy arrays
        for job in self.metadata:
            raw = job.get("skill_embeddings", [])
            job["skill_embeddings"] = [
                np.array(e, dtype=np.float32) for e in raw
            ] if raw else []

        total_embs = sum(len(j["skill_embeddings"]) for j in self.metadata)
        logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors, "
            f"dim={self._embedding_dim}, "
            f"{total_embs} pre-computed skill embeddings ready"
        )

    def _save_to_disk(self) -> None:
        # numpy arrays → lists for JSON
        serializable = []
        for job in self.metadata:
            entry = {k: v for k, v in job.items() if k != "skill_embeddings"}
            entry["skill_embeddings"] = [
                e.tolist() for e in job.get("skill_embeddings", [])
            ]
            serializable.append(entry)

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "w") as f:
            json.dump(serializable, f)
        logger.info("FAISS index + skill embeddings saved to disk.")

    async def rebuild(self) -> None:
        """
        Fetch all jobs from Firestore, build FAISS index, and
        pre-compute per-skill embeddings stored in metadata.
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

        # ── FAISS vectors: job_title + all skills combined ────────────────────
        job_texts = [
            job["job_title"] + " " + " ".join(job.get("skills", []))
            for job in jobs
        ]
        job_embeddings = self.engine.embed_batch(job_texts)
        self._embedding_dim = job_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self._embedding_dim)
        self.index.add(job_embeddings)

        # ── Pre-compute ALL individual skill embeddings in one batch ──────────
        # Flatten all skills from all jobs into one list, embed once,
        # then slice back per-job. This is maximally efficient — one GPU/CPU
        # batch call instead of one-per-skill-per-job.
        all_skills: list[str] = []
        job_skill_slices: list[tuple[int, int]] = []  # (start, end) per job

        for job in jobs:
            job_skills = job.get("skills", [])
            start = len(all_skills)
            all_skills.extend(job_skills)
            job_skill_slices.append((start, start + len(job_skills)))

        if all_skills:
            logger.info(
                f"Pre-computing {len(all_skills)} skill embeddings "
                f"across {len(jobs)} jobs (single batch)..."
            )
            all_skill_embs = self.engine.embed_batch(all_skills)
        else:
            all_skill_embs = np.empty((0, self._embedding_dim), dtype=np.float32)

        # ── Build metadata with sliced per-job skill embeddings ───────────────
        self.metadata = []
        for i, job in enumerate(jobs):
            start, end = job_skill_slices[i]
            skill_embs = list(all_skill_embs[start:end]) if end > start else []
            self.metadata.append({
                "job_title": job["job_title"],
                "skills": job.get("skills", []),
                "certifications": job.get("certifications", []),
                "doc_id": job.get("id", ""),
                "skill_embeddings": skill_embs,  # list[np.ndarray], one per skill
            })

        self._save_to_disk()
        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, "
            f"{len(all_skills)} skill embeddings pre-computed and stored."
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Top-K candidate retrieval by FAISS cosine similarity.
        Returns job metadata (including pre-computed skill_embeddings).
        faiss_score is a retrieval signal only — NOT the displayed match %.
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
            raw_sim = round(max(float(score), 0.0), 4)
            job_info["faiss_score"] = raw_sim
            job_info["match_score"] = raw_sim  # overwritten by scorer
            results.append(job_info)

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results