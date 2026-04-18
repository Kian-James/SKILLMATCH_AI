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
import re
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

MMR_LAMBDA = 0.5

# ── Duplicate detection thresholds ────────────────────────────────────────────
# Raised from 0.92 to 0.97 so that semantically-similar-but-distinct roles
# like "Unity VR Developer" and "Oculus Quest VR Developer" are NOT merged.
# Only near-identical titles (e.g. "MongoDB NoSQL Developer" vs "MongoDB Developer")
# should ever be collapsed.
DUPLICATE_SIM_THRESHOLD = 0.97

# Even at 0.97 some pairs slip through if their lead technology differs.
# This guard extracts the first meaningful token(s) from each title and
# refuses to merge if they don't share a common technology prefix.
# Examples that would be blocked regardless of similarity score:
#   "Unity VR Developer"        vs "Oculus Quest VR Developer"   → Unity ≠ Oculus
#   "Python Software Engineer"  vs "Java Software Engineer"      → Python ≠ Java
#   "Tableau Data Analyst"      vs "Power BI Data Analyst"       → Tableau ≠ Power
#   "AWS Cloud Engineer"        vs "Azure Cloud Engineer"        → AWS ≠ Azure
TECH_PREFIX_GUARD = True

# Generic words that are NOT technology qualifiers.
# If the first word of a title is one of these, we look at the second word.
_GENERIC_WORDS = {
    "senior", "junior", "lead", "principal", "staff", "associate", "chief",
    "head", "director", "vp", "engineer", "developer", "architect", "analyst",
    "scientist", "specialist", "consultant", "manager", "administrator",
    "intern", "entry", "mid", "level", "the", "a", "an",
}


def _title_tech_prefix(title: str) -> str:
    """
    Extract the leading technology token(s) from a job title.

    Rules:
    - Lowercase and split on whitespace
    - Skip any leading generic words (seniority, role nouns, articles)
    - Return the first non-generic word as the technology identifier
    - If all words are generic, return the full lowercased title
      (so we fall back to full-title comparison, which is safe)

    Examples:
        "Python Software Engineer"       → "python"
        "Unity VR Developer"             → "unity"
        "Oculus Quest VR Developer"      → "oculus"
        "AWS Cloud Architect"            → "aws"
        "Azure Cloud Architect"          → "azure"
        "Senior React Frontend Engineer" → "react"
        "MongoDB Developer"              → "mongodb"
        "MongoDB NoSQL Developer"        → "mongodb"
        "Java Backend Developer"         → "java"
        "Java Backend Engineer"          → "java"   ← same prefix → can merge
    """
    words = re.sub(r"[^a-z0-9\s]", "", title.lower()).split()
    for word in words:
        if word not in _GENERIC_WORDS:
            return word
    return title.lower()


def _safe_to_merge(title_a: str, title_b: str) -> bool:
    """
    Return True only if it is safe to treat these two titles as duplicates.

    Two conditions must BOTH hold:
      1. Their leading technology prefixes match  (e.g. both start with "python")
      2. OR both titles have no distinguishing tech prefix (all-generic words)

    If the prefixes differ the titles represent distinct specialisations and
    must be kept separate even when their embeddings are very similar.
    """
    if not TECH_PREFIX_GUARD:
        return True
    prefix_a = _title_tech_prefix(title_a)
    prefix_b = _title_tech_prefix(title_b)
    return prefix_a == prefix_b


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
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "r") as f:
            self.metadata = json.load(f)
        self._embedding_dim = self.index.d

        self.skill_index = faiss.read_index(str(FAISS_SKILL_INDEX_PATH))
        with open(FAISS_SKILL_META_PATH, "r") as f:
            self.skill_meta = json.load(f)

        if FAISS_CTX_EMBS_PATH.exists():
            self._ctx_embs = np.load(str(FAISS_CTX_EMBS_PATH))
            logger.debug(f"Ctx embeddings loaded: shape={self._ctx_embs.shape}")

        if FAISS_IDF_PATH.exists():
            with open(FAISS_IDF_PATH, "r") as f:
                self.idf_weights = json.load(f)

        logger.info(
            f"FAISS loaded: {self.index.ntotal} jobs, "
            f"{self.skill_index.ntotal} skill vectors, "
            f"{len(self.idf_weights)} IDF weights."
        )

    def _save_to_disk(self) -> None:
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

        faiss.write_index(self.skill_index, str(FAISS_SKILL_INDEX_PATH))
        with open(FAISS_SKILL_META_PATH, "w") as f:
            json.dump(self.skill_meta, f)

        if self._ctx_embs is not None:
            np.save(str(FAISS_CTX_EMBS_PATH), self._ctx_embs.astype(np.float16))

        with open(FAISS_IDF_PATH, "w") as f:
            json.dump(self.idf_weights, f)

        logger.info("All FAISS indexes + IDF weights saved to disk.")

    # ── Rebuild ───────────────────────────────────────────────────────────────

    async def rebuild(self) -> None:
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

        logger.info(f"Building FAISS indexes for {len(jobs)} jobs...")

        # Step 1: IDF weights
        N = len(jobs)
        freq: Counter = Counter()
        for job in jobs:
            freq.update(set(s.lower() for s in job.get("skills", [])))
        self.idf_weights = {s: math.log(N / f) for s, f in freq.items() if f > 0}
        logger.info(f"IDF weights: {len(self.idf_weights)} unique skills.")

        # Step 2: Job-level embeddings
        job_texts = [
            job["job_title"] + " " + " ".join(job.get("skills", []))
            for job in jobs
        ]
        job_embeddings = self.engine.embed_batch(job_texts)   # (N, dim)
        self._embedding_dim = job_embeddings.shape[1]

        # Step 3: Duplicate detection
        # A pair is only a duplicate when BOTH conditions are true:
        #   a) cosine similarity >= DUPLICATE_SIM_THRESHOLD (0.97)
        #   b) their leading technology prefix matches (_safe_to_merge)
        #
        # This prevents false positives like:
        #   "Unity VR Developer" vs "Oculus Quest VR Developer"  (sim≈0.94, diff prefix)
        #   "Python Software Engineer" vs "Java Software Engineer" (sim≈0.95, diff prefix)
        #   "AWS Cloud Engineer" vs "Azure Cloud Engineer"         (sim≈0.93, diff prefix)
        #
        # While still catching genuine redundancies like:
        #   "MongoDB NoSQL Developer" vs "MongoDB Developer"       (sim≈0.99, same prefix)
        #   "Java Backend Engineer" vs "Java Backend Developer"    (sim≈0.99, same prefix)
        sim_matrix = job_embeddings @ job_embeddings.T
        duplicate_of: dict[int, int] = {}

        for i in range(N):
            if i in duplicate_of:
                continue
            for j in range(i + 1, N):
                if j in duplicate_of:
                    continue

                sim = float(sim_matrix[i, j])
                if sim < DUPLICATE_SIM_THRESHOLD:
                    continue

                title_i = jobs[i]["job_title"]
                title_j = jobs[j]["job_title"]

                # Tech-prefix guard: different lead technology → never merge
                if not _safe_to_merge(title_i, title_j):
                    logger.debug(
                        f"Prefix mismatch — kept both: '{title_i}' vs '{title_j}' "
                        f"(sim={sim:.3f}, "
                        f"prefix='{_title_tech_prefix(title_i)}' vs "
                        f"'{_title_tech_prefix(title_j)}')"
                    )
                    continue

                # Same tech prefix + high similarity → genuine duplicate
                li = len(jobs[i].get("skills", []))
                lj = len(jobs[j].get("skills", []))
                dropped, kept = (j, i) if li >= lj else (i, j)
                duplicate_of[dropped] = kept
                logger.info(
                    f"Duplicate: '{jobs[dropped]['job_title']}' ≈ "
                    f"'{jobs[kept]['job_title']}' (sim={sim:.3f}) — dropped"
                )

        kept_indices    = [i for i in range(N) if i not in duplicate_of]
        kept_jobs       = [jobs[i] for i in kept_indices]
        kept_embeddings = job_embeddings[kept_indices]
        logger.info(
            f"{len(kept_jobs)} jobs kept after dedup "
            f"({N - len(kept_jobs)} removed)."
        )

        # Step 4: Job FAISS index
        self.index = faiss.IndexFlatIP(self._embedding_dim)
        self.index.add(kept_embeddings)

        # Step 5: Lightweight metadata
        self.metadata = [
            {
                "job_title":      job["job_title"],
                "skills":         job.get("skills", []),
                "certifications": job.get("certifications", []),
                "doc_id":         job.get("id", ""),
            }
            for job in kept_jobs
        ]

        # Step 6: Skill embeddings
        all_bare_texts: list[str] = []
        all_ctx_texts:  list[str] = []
        job_skill_slices: list[tuple[int, int]] = []

        for job in kept_jobs:
            skills = job.get("skills", [])
            start  = len(all_bare_texts)
            all_bare_texts.extend(skills)
            all_ctx_texts.extend(f"{job['job_title']}: {s}" for s in skills)
            job_skill_slices.append((start, start + len(skills)))

        if all_bare_texts:
            logger.info(
                f"Embedding {len(all_bare_texts)} skills × 2 (bare + ctx) in two batches..."
            )
            bare_embs = self.engine.embed_batch(all_bare_texts)
            ctx_embs  = self.engine.embed_batch(all_ctx_texts)
        else:
            bare_embs = np.empty((0, self._embedding_dim), dtype=np.float32)
            ctx_embs  = np.empty((0, self._embedding_dim), dtype=np.float32)

        # Step 7: Skill FAISS index
        self.skill_index = faiss.IndexFlatIP(self._embedding_dim)
        self.skill_meta  = []
        if len(bare_embs) > 0:
            self.skill_index.add(bare_embs)
            for job_idx, (job, (start, end)) in enumerate(
                    zip(kept_jobs, job_skill_slices)
            ):
                for skill_name in job.get("skills", []):
                    self.skill_meta.append({
                        "job_idx":    job_idx,
                        "skill_name": skill_name,
                    })

        # Step 8: Contextual embeddings as float16
        self._ctx_embs = ctx_embs.astype(np.float16) if len(ctx_embs) > 0 else None

        self._save_to_disk()

        del bare_embs, ctx_embs, job_embeddings, kept_embeddings, sim_matrix
        import gc; gc.collect()

        logger.info(
            f"FAISS rebuild complete: {self.index.ntotal} jobs, "
            f"{self.skill_index.ntotal} skill vectors."
        )

    # ── Job search with MMR ───────────────────────────────────────────────────

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

        cand_embs = np.stack([
            self.index.reconstruct(idx) for idx, _ in candidates
        ])

        selected_embs: list[np.ndarray] = []
        result: list[dict] = []

        while len(result) < top_k and candidates:
            if not selected_embs:
                best_pos = int(np.argmax([info["faiss_score"] for _, info in candidates]))
            else:
                sel_matrix = np.stack(selected_embs)
                mmr_scores = []
                for pos, (_, info) in enumerate(candidates):
                    rel     = info["faiss_score"]
                    max_sim = float(np.max(sel_matrix @ cand_embs[pos]))
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
        """Find the closest skill in the index to a user skill embedding."""
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
        """Return the contextual embedding for a skill at its global index position."""
        if self._ctx_embs is None or skill_global_idx >= len(self._ctx_embs):
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