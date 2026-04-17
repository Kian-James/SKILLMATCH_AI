"""
ai_modules/faiss_index.py

Builds and queries TWO FAISS indexes:

  1. job_index  (IndexFlatIP over job profile embeddings)
     — Top-K job retrieval by semantic similarity (current behaviour, unchanged).
     — Each vector = embed(job_title + all_skills).

  2. skill_index (IndexFlatIP over individual skill embeddings from all jobs)
     — Skill-level fuzzy matching: find the single closest job-skill to any
       user-skill without looping. Used by resume_service Gate 3 instead of
       the raw dot-product loop.
     — metadata maps vector_id → {job_idx, skill_name, skill_emb}

FAISS role upgrade:
  Before: FAISS used only for job-level candidate retrieval.
  After:
    - Job index   → still retrieves top-K candidate jobs (retrieval signal).
    - Skill index → precise per-skill fuzzy matching during scoring (replaces
                    the user_skill_embs @ job_skill_emb vectorized loop).
    - Duplicate detection: cosine matrix over job embeddings at rebuild time
      flags near-duplicate jobs (sim > 0.92) and keeps only the richer entry.
    - IDF weights: skill document frequency computed at build time and stored
      in metadata so scoring can weight rare skills more heavily.
    - MMR (Maximal Marginal Relevance): search() fetches 2×top_k candidates
      and re-ranks for diversity so top-5 results aren't all the same job title.

Performance:
  skill_index queries are O(n_total_skills) FAISS flat search — still <1ms.
  All per-job skill embeddings and IDF weights are precomputed at build time.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import faiss
import numpy as np
from loguru import logger

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from ai_modules.embedding_engine import EmbeddingEngine
from services.firebase_service import fetch_all_jobs

# Paths for the new skill-level index
from pathlib import Path
FAISS_SKILL_INDEX_PATH = Path("faiss_index/skill_embeddings.index")
FAISS_SKILL_META_PATH  = Path("faiss_index/skill_metadata.json")

# MMR diversity weight (0 = pure relevance, 1 = pure diversity)
MMR_LAMBDA: float = 0.5

# Duplicate detection threshold: jobs with cosine sim above this are duplicates
DUPLICATE_SIM_THRESHOLD: float = 0.92


class FAISSIndex:
    """
    Dual FAISS index: job-level retrieval + skill-level fuzzy matching.
    """

    def __init__(self):
        self.engine = EmbeddingEngine()

        # Job index
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self._embedding_dim: int | None = None

        # Skill index
        self.skill_index: faiss.IndexFlatIP | None = None
        self.skill_meta: list[dict] = []   # [{job_idx, skill_name}, ...]

        # IDF weights: canonical_skill_name → idf_score
        self.idf_weights: dict[str, float] = {}

    # ── Load / Build ──────────────────────────────────────────────────────────

    async def load_or_build(self) -> None:
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        if (FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists()
                and FAISS_SKILL_INDEX_PATH.exists() and FAISS_SKILL_META_PATH.exists()):
            logger.info("Loading FAISS indexes from disk...")
            self._load_from_disk()
        else:
            logger.info("No FAISS indexes on disk — building from Firestore...")
            await self.rebuild()

    def _load_from_disk(self) -> None:
        # Job index
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "r") as f:
            self.metadata = json.load(f)
        self._embedding_dim = self.index.d

        for job in self.metadata:
            raw = job.get("skill_embeddings", [])
            job["skill_embeddings"] = [
                np.array(e, dtype=np.float32) for e in raw
            ] if raw else []

        self.idf_weights = self.metadata[0].get("_idf_weights", {}) if self.metadata else {}
        for job in self.metadata:
            job.pop("_idf_weights", None)

        # Skill index
        self.skill_index = faiss.read_index(str(FAISS_SKILL_INDEX_PATH))
        with open(FAISS_SKILL_META_PATH, "r") as f:
            self.skill_meta = json.load(f)

        total_skill_vecs = self.skill_index.ntotal
        logger.info(
            f"FAISS loaded: {self.index.ntotal} jobs, "
            f"{total_skill_vecs} skill vectors, "
            f"{len(self.idf_weights)} IDF weights."
        )

    def _save_to_disk(self) -> None:
        # Serialize job metadata (numpy arrays → lists)
        serializable = []
        for job in self.metadata:
            entry = {k: v for k, v in job.items() if k != "skill_embeddings"}
            entry["skill_embeddings"] = [
                e.tolist() for e in job.get("skill_embeddings", [])
            ]
            serializable.append(entry)

        # Attach IDF weights to first entry as a side-channel (avoids separate file)
        if serializable:
            serializable[0]["_idf_weights"] = self.idf_weights

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "w") as f:
            json.dump(serializable, f)

        faiss.write_index(self.skill_index, str(FAISS_SKILL_INDEX_PATH))
        with open(FAISS_SKILL_META_PATH, "w") as f:
            json.dump(self.skill_meta, f)

        logger.info("FAISS indexes + skill index + IDF weights saved to disk.")

    # ── Rebuild ───────────────────────────────────────────────────────────────

    async def rebuild(self) -> None:
        jobs = fetch_all_jobs()
        if not jobs:
            logger.warning("No jobs in Firestore — FAISS indexes are empty.")
            dim = self.engine.model.get_sentence_embedding_dimension()
            self._embedding_dim = dim
            self.index = faiss.IndexFlatIP(dim)
            self.skill_index = faiss.IndexFlatIP(dim)
            self.metadata = []
            self.skill_meta = []
            self.idf_weights = {}
            return

        logger.info(f"Building FAISS indexes for {len(jobs)} jobs...")

        # ── Step 1: Compute IDF weights ───────────────────────────────────────
        N = len(jobs)
        skill_doc_freq: Counter = Counter()
        for job in jobs:
            unique_skills = set(s.lower() for s in job.get("skills", []))
            skill_doc_freq.update(unique_skills)

        self.idf_weights = {
            skill: math.log(N / freq)
            for skill, freq in skill_doc_freq.items()
            if freq > 0
        }
        logger.info(f"IDF weights computed for {len(self.idf_weights)} unique skills.")

        # ── Step 2: Job-level embeddings ──────────────────────────────────────
        job_texts = [
            job["job_title"] + " " + " ".join(job.get("skills", []))
            for job in jobs
        ]
        job_embeddings = self.engine.embed_batch(job_texts)
        self._embedding_dim = job_embeddings.shape[1]

        # ── Step 3: Duplicate detection ───────────────────────────────────────
        # Compute pairwise cosine matrix (embeddings are already L2-normalized)
        sim_matrix = job_embeddings @ job_embeddings.T  # (N, N)
        duplicate_of: dict[int, int] = {}  # idx → kept_idx

        for i in range(N):
            if i in duplicate_of:
                continue
            for j in range(i + 1, N):
                if j in duplicate_of:
                    continue
                if sim_matrix[i, j] >= DUPLICATE_SIM_THRESHOLD:
                    # Keep the job with more skills; drop the other
                    len_i = len(jobs[i].get("skills", []))
                    len_j = len(jobs[j].get("skills", []))
                    dropped, kept = (j, i) if len_i >= len_j else (i, j)
                    duplicate_of[dropped] = kept
                    logger.info(
                        f"Duplicate detected: '{jobs[dropped]['job_title']}' ≈ "
                        f"'{jobs[kept]['job_title']}' (sim={sim_matrix[i,j]:.3f}) — "
                        f"dropping '{jobs[dropped]['job_title']}'"
                    )

        kept_indices = [i for i in range(N) if i not in duplicate_of]
        if len(kept_indices) < N:
            logger.info(
                f"Removed {N - len(kept_indices)} duplicate jobs. "
                f"{len(kept_indices)} remain."
            )
        kept_jobs = [jobs[i] for i in kept_indices]
        kept_embeddings = job_embeddings[kept_indices]

        # ── Step 4: Build job FAISS index ─────────────────────────────────────
        self.index = faiss.IndexFlatIP(self._embedding_dim)
        self.index.add(kept_embeddings)

        # ── Step 5: Pre-compute per-skill embeddings (single batch) ───────────
        all_skills: list[str] = []
        job_skill_slices: list[tuple[int, int]] = []

        for job in kept_jobs:
            job_skills = job.get("skills", [])
            start = len(all_skills)
            all_skills.extend(job_skills)
            job_skill_slices.append((start, start + len(job_skills)))

        if all_skills:
            logger.info(
                f"Pre-computing {len(all_skills)} skill embeddings across "
                f"{len(kept_jobs)} jobs (single batch)..."
            )
            all_skill_embs = self.engine.embed_batch(all_skills)
        else:
            all_skill_embs = np.empty((0, self._embedding_dim), dtype=np.float32)

        # ── Step 6: Build skill FAISS index ───────────────────────────────────
        self.skill_index = faiss.IndexFlatIP(self._embedding_dim)
        self.skill_meta = []

        if len(all_skill_embs) > 0:
            self.skill_index.add(all_skill_embs)
            for job_idx, (job, (start, end)) in enumerate(
                zip(kept_jobs, job_skill_slices)
            ):
                for k, skill_name in enumerate(job.get("skills", [])):
                    self.skill_meta.append({
                        "job_idx": job_idx,
                        "skill_name": skill_name,
                    })

        # ── Step 7: Build job metadata ─────────────────────────────────────────
        self.metadata = []
        for job_idx, (job, (start, end)) in enumerate(
            zip(kept_jobs, job_skill_slices)
        ):
            skill_embs = list(all_skill_embs[start:end]) if end > start else []

            # Contextual embeddings: embed "job_title: skill" for each skill
            ctx_texts = [
                f"{job['job_title']}: {s}" for s in job.get("skills", [])
            ]
            if ctx_texts:
                ctx_embs = list(self.engine.embed_batch(ctx_texts))
            else:
                ctx_embs = []

            self.metadata.append({
                "job_title": job["job_title"],
                "skills": job.get("skills", []),
                "certifications": job.get("certifications", []),
                "doc_id": job.get("id", ""),
                "skill_embeddings": skill_embs,      # bare skill embeddings
                "skill_ctx_embeddings": ctx_embs,    # contextual embeddings
            })

        self._save_to_disk()
        logger.info(
            f"FAISS rebuild complete: {self.index.ntotal} jobs, "
            f"{self.skill_index.ntotal} skill vectors."
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Top-K job retrieval with MMR re-ranking for result diversity.
        Fetches 2*top_k candidates then applies Maximal Marginal Relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS job index is empty — no results.")
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        fetch_k = min(top_k * 2, self.index.ntotal)
        scores, indices = self.index.search(query, fetch_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            job_info = self.metadata[idx].copy()
            job_info["faiss_score"] = round(max(float(score), 0.0), 4)
            job_info["match_score"] = job_info["faiss_score"]
            candidates.append((idx, job_info))

        if len(candidates) <= top_k:
            return [info for _, info in candidates]

        # MMR re-ranking: balance relevance vs diversity
        selected_indices: list[int] = []
        selected_infos: list[dict] = []
        candidate_embs = np.array([
            kept_embeddings_row
            for kept_embeddings_row in [
                self.index.reconstruct(int(idx)) for idx, _ in candidates
            ]
        ])

        while len(selected_indices) < top_k and candidates:
            if not selected_indices:
                # First pick: highest FAISS score
                best_pos = int(np.argmax([info["faiss_score"] for _, info in candidates]))
            else:
                # MMR score = relevance - λ * max_sim_to_already_selected
                selected_embs = candidate_embs[selected_indices]
                mmr_scores = []
                for pos, (_, info) in enumerate(candidates):
                    rel = info["faiss_score"]
                    emb = candidate_embs[pos]
                    max_sim = float(np.max(selected_embs @ emb))
                    mmr_scores.append(rel - MMR_LAMBDA * max_sim)
                best_pos = int(np.argmax(mmr_scores))

            selected_indices.append(best_pos)
            selected_infos.append(candidates[best_pos][1])
            candidates.pop(best_pos)
            candidate_embs = np.delete(candidate_embs, best_pos, axis=0)

        return selected_infos

    # ── Skill-level fuzzy search ───────────────────────────────────────────────

    def find_closest_skill(
        self, user_skill_emb: np.ndarray, threshold: float = 0.68
    ) -> tuple[str | None, float]:
        """
        Find the single closest skill in the skill index to a user skill embedding.
        Returns (skill_name, similarity) or (None, 0.0) if below threshold.
        Used by resume_service as Gate 3 (replaces vectorized dot-product loop).
        """
        if self.skill_index is None or self.skill_index.ntotal == 0:
            return None, 0.0

        query = user_skill_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self.skill_index.search(query, 1)
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        if best_idx < 0 or best_score < threshold:
            return None, best_score

        return self.skill_meta[best_idx]["skill_name"], best_score
