"""
ai_modules/faiss_index.py

Dual FAISS index: job-level retrieval + skill-level fuzzy matching.

MEMORY FIXES vs previous version:
  1. skill_embeddings REMOVED from job_metadata dict and JSON entirely.
     - Was: 36MB JSON → ~200MB+ RAM after deserialization into list[np.ndarray].
     - Now: all skill lookups go through the FAISS skill_index instead.
     - job_metadata.json is now ~2MB (titles, skill names, certs, doc_ids only).

  2. skill_ctx_embeddings (contextual) stored as a single compact float16
     numpy matrix per-process instead of per-job list[np.ndarray].
     - float16 halves storage vs float32 with negligible precision loss for
       cosine similarity (embeddings are already normalized to unit length).
     - Stored in a separate .npy file, not in JSON.

  3. IDF weights stored in a separate small JSON file (not piggy-backed on metadata).

  4. MMR re-ranking uses index.reconstruct() only for selected candidates
     (not all candidates), so no full matrix reconstruction.

  5. SkillClusterer removed from FAISSIndex — it belongs in resume_service
     only and should be a module-level singleton, not re-instantiated per request.

Memory budget (500 jobs, 12 skills each, dim=384):
  PyTorch + sentence-transformers model : ~300 MB
  spaCy en_core_web_sm                  : ~15 MB
  Firebase SDK + FastAPI                : ~60 MB
  FAISS job index (500 vectors)         : ~1 MB
  FAISS skill index (6000 vectors)      : ~10 MB
  job_metadata.json (no embeddings)     : ~2 MB
  ctx_embeddings float16 matrix         : ~5 MB
  IDF weights dict                      : <1 MB
  SkillNormalizer canonical embs        : ~1 MB (shared model, lazy)
  ──────────────────────────────────────────────
  Total                                 : ~395 MB  ← safely under 512 MB
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from ai_modules.embedding_engine import EmbeddingEngine
from services.firebase_service import fetch_all_jobs

FAISS_SKILL_INDEX_PATH = Path("faiss_index/skill_embeddings.index")
FAISS_SKILL_META_PATH  = Path("faiss_index/skill_metadata.json")
FAISS_CTX_EMBS_PATH    = Path("faiss_index/ctx_embeddings.npy")   # float16
FAISS_IDF_PATH         = Path("faiss_index/idf_weights.json")

MMR_LAMBDA              = 0.5
DUPLICATE_SIM_THRESHOLD = 0.92


class FAISSIndex:

    def __init__(self):
        self.engine = EmbeddingEngine()

        # Job index
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []          # no embeddings inside — lightweight
        self._embedding_dim: int | None = None

        # Skill index
        self.skill_index: faiss.IndexFlatIP | None = None
        self.skill_meta: list[dict] = []        # [{job_idx, skill_name}]

        # Contextual embeddings matrix: (n_total_skills, dim) float16
        # Stored separately from skill_meta to avoid Python object overhead
        self._ctx_embs: np.ndarray | None = None  # shape (n_skills, dim)

        # IDF weights
        self.idf_weights: dict[str, float] = {}

    # ── Load / Build ──────────────────────────────────────────────────────────

    async def load_or_build(self) -> None:
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        indexes_exist = (
                FAISS_INDEX_PATH.exists()
                and FAISS_METADATA_PATH.exists()
                and FAISS_SKILL_INDEX_PATH.exists()
                and FAISS_SKILL_META_PATH.exists()
        )
        if indexes_exist:
            logger.info("Loading FAISS indexes from disk...")
            self._load_from_disk()
        else:
            logger.info("No FAISS indexes on disk — building from Firestore...")
            await self.rebuild()

    def _load_from_disk(self) -> None:
        # Job index + lightweight metadata (no embeddings)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "r") as f:
            self.metadata = json.load(f)
        self._embedding_dim = self.index.d

        # Skill index
        self.skill_index = faiss.read_index(str(FAISS_SKILL_INDEX_PATH))
        with open(FAISS_SKILL_META_PATH, "r") as f:
            self.skill_meta = json.load(f)

        # Contextual embeddings matrix (float16 .npy)
        if FAISS_CTX_EMBS_PATH.exists():
            self._ctx_embs = np.load(str(FAISS_CTX_EMBS_PATH))  # float16
            logger.debug(f"Ctx embeddings loaded: shape={self._ctx_embs.shape}")

        # IDF weights
        if FAISS_IDF_PATH.exists():
            with open(FAISS_IDF_PATH, "r") as f:
                self.idf_weights = json.load(f)

        logger.info(
            f"FAISS loaded: {self.index.ntotal} jobs, "
            f"{self.skill_index.ntotal} skill vectors, "
            f"{len(self.idf_weights)} IDF weights."
        )

    def _save_to_disk(self) -> None:
        # Job index + metadata (no embeddings in JSON)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

        # Skill index
        faiss.write_index(self.skill_index, str(FAISS_SKILL_INDEX_PATH))
        with open(FAISS_SKILL_META_PATH, "w") as f:
            json.dump(self.skill_meta, f)

        # Contextual embeddings as float16 numpy binary
        if self._ctx_embs is not None:
            np.save(str(FAISS_CTX_EMBS_PATH), self._ctx_embs.astype(np.float16))

        # IDF weights
        with open(FAISS_IDF_PATH, "w") as f:
            json.dump(self.idf_weights, f)

        logger.info("All FAISS indexes + IDF weights saved to disk.")

    # ── Rebuild ───────────────────────────────────────────────────────────────

    async def rebuild(self) -> None:
        import gc
        jobs = fetch_all_jobs()
        if not jobs:
            logger.warning("No jobs in Firestore — FAISS indexes are empty.")
            dim = self.engine.model.get_sentence_embedding_dimension()
            self._embedding_dim = dim
            self.index       = faiss.IndexFlatIP(dim)
            self.skill_index = faiss.IndexFlatIP(dim)
            self.metadata    = []
            self.skill_meta  = []
            self.idf_weights = {}
            return

        N = len(jobs)
        logger.info(f"Memory-safe rebuild: {N} jobs in batches of 50...")

        # Step 1: IDF weights (pure counting — no embeddings)
        freq: Counter = Counter()
        for job in jobs:
            freq.update(set(s.lower() for s in job.get("skills", [])))
        self.idf_weights = {s: math.log(N / f) for s, f in freq.items() if f > 0}
        logger.info(f"IDF weights: {len(self.idf_weights)} unique skills.")

        # Step 2: Embed job texts in batches of 50 → add to FAISS → free immediately
        BATCH = 50
        dim = self.engine.model.get_sentence_embedding_dimension()
        self._embedding_dim = dim
        self.index    = faiss.IndexFlatIP(dim)
        self.metadata = []

        for start in range(0, N, BATCH):
            chunk = jobs[start : start + BATCH]
            texts = [
                job["job_title"] + " " + " ".join(job.get("skills", []))
                for job in chunk
            ]
            embs = self.engine.embed_batch(texts)
            self.index.add(embs)
            for job in chunk:
                self.metadata.append({
                    "job_title":      job["job_title"],
                    "skills":         job.get("skills", []),
                    "certifications": job.get("certifications", []),
                    "doc_id":         job.get("id", ""),
                })
            del embs, texts
            gc.collect()
            logger.info(f"  Job batch done: {min(start + BATCH, N)}/{N}")

        # Step 3: Embed bare skill texts per job → add to skill FAISS → free immediately
        # Collect ctx texts separately for Step 4
        self.skill_index = faiss.IndexFlatIP(dim)
        self.skill_meta  = []
        all_ctx_texts: list[str] = []

        for job_idx, job in enumerate(jobs):
            skills = job.get("skills", [])
            if not skills:
                continue
            bare_embs = self.engine.embed_batch(skills)
            self.skill_index.add(bare_embs)
            for skill_name in skills:
                self.skill_meta.append({"job_idx": job_idx, "skill_name": skill_name})
            all_ctx_texts.extend(f"{job['job_title']}: {s}" for s in skills)
            del bare_embs
            if (job_idx + 1) % BATCH == 0:
                gc.collect()
                logger.info(f"  Skill batch done: {job_idx + 1}/{N}")

        gc.collect()
        logger.info(f"Skill index: {self.skill_index.ntotal} vectors.")

        # Step 4: Embed contextual texts in batches of 500 → store as float16 matrix
        ctx_chunks: list[np.ndarray] = []
        CTX_BATCH = 500
        for start in range(0, len(all_ctx_texts), CTX_BATCH):
            chunk_embs = self.engine.embed_batch(
                all_ctx_texts[start : start + CTX_BATCH]
            ).astype(np.float16)
            ctx_chunks.append(chunk_embs)
            del chunk_embs
            gc.collect()

        self._ctx_embs = np.concatenate(ctx_chunks) if ctx_chunks else None
        del ctx_chunks, all_ctx_texts
        gc.collect()

        self._save_to_disk()
        gc.collect()

        logger.info(
            f"FAISS rebuild complete: {self.index.ntotal} jobs, "
            f"{self.skill_index.ntotal} skill vectors."
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Top-K job retrieval with MMR diversity re-ranking."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS job index is empty.")
            return []

        query   = query_embedding.reshape(1, -1).astype(np.float32)
        fetch_k = min(top_k * 2, self.index.ntotal)
        scores, indices = self.index.search(query, fetch_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            info = self.metadata[idx].copy()
            info["faiss_score"] = round(max(float(score), 0.0), 4)
            info["match_score"] = info["faiss_score"]
            candidates.append((int(idx), info))

        if len(candidates) <= top_k:
            return [info for _, info in candidates]

        # MMR: reconstruct embeddings only for the candidate set (not all jobs)
        cand_embs = np.stack([
            self.index.reconstruct(idx) for idx, _ in candidates
        ])  # (fetch_k, dim)

        selected_positions: list[int] = []
        selected_embs: list[np.ndarray] = []
        result: list[dict] = []

        while len(result) < top_k and candidates:
            if not selected_embs:
                best_pos = int(np.argmax([info["faiss_score"] for _, info in candidates]))
            else:
                sel_matrix = np.stack(selected_embs)  # (n_sel, dim)
                mmr_scores = []
                for pos, (_, info) in enumerate(candidates):
                    rel      = info["faiss_score"]
                    max_sim  = float(np.max(sel_matrix @ cand_embs[pos]))
                    mmr_scores.append(rel - MMR_LAMBDA * max_sim)
                best_pos = int(np.argmax(mmr_scores))

            selected_embs.append(cand_embs[best_pos])
            result.append(candidates[best_pos][1])
            candidates.pop(best_pos)
            cand_embs = np.delete(cand_embs, best_pos, axis=0)

        return result

    # ── Skill-level fuzzy search ───────────────────────────────────────────────

    def find_closest_skill(
            self, user_skill_emb: np.ndarray, threshold: float = 0.68
    ) -> tuple[str | None, float]:
        """
        Find the closest skill in the skill_index to a user skill embedding.
        Used by resume_service as Gate 3a (replaces per-job dot-product loop).
        """
        if self.skill_index is None or self.skill_index.ntotal == 0:
            return None, 0.0

        query = user_skill_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self.skill_index.search(query, 1)
        best_score = float(scores[0][0])
        best_idx   = int(indices[0][0])

        if best_idx < 0 or best_score < threshold:
            return None, best_score

        return self.skill_meta[best_idx]["skill_name"], best_score

    # ── Contextual embedding lookup ───────────────────────────────────────────

    def get_ctx_embedding(self, skill_global_idx: int) -> np.ndarray | None:
        """
        Return the contextual embedding for a skill at its global skill index position.
        Returns float32 (upcast from stored float16).
        """
        if (self._ctx_embs is None
                or skill_global_idx >= len(self._ctx_embs)):
            return None
        return self._ctx_embs[skill_global_idx].astype(np.float32)

    def get_skill_global_offset(self, job_idx: int) -> int:
        """Return the first global skill index for a given job index."""
        offset = 0
        for meta in self.skill_meta:
            if meta["job_idx"] == job_idx:
                return offset
            offset += 1
        return -1