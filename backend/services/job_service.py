"""
services/job_service.py
Skill gap detection: compares user's extracted skills vs job requirements.
Computes completion score and identifies missing skills.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from services.firebase_service import fetch_all_jobs
from ai_modules.embedding_engine import EmbeddingEngine
from utils.skill_normalizer import SkillNormalizer


class JobService:
    """Handles skill gap analysis and job requirement lookups."""

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.normalizer = SkillNormalizer()

    def find_job_by_title(self, target_title: str) -> dict | None:
        """
        Find the best matching job in Firestore for a given title string.
        Uses embedding similarity to handle fuzzy title matching.
        """
        jobs = fetch_all_jobs()
        if not jobs:
            return None

        target_lower = target_title.lower().strip()

        # Exact match first (case-insensitive)
        for job in jobs:
            if job["job_title"].lower() == target_lower:
                return job

        # Fuzzy match via embedding similarity
        target_emb = self.embedder.embed(target_title)
        job_title_texts = [j["job_title"] for j in jobs]
        title_embeddings = self.embedder.embed_batch(job_title_texts)

        sims = [float(np.dot(target_emb, t)) for t in title_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= 0.75:  # Threshold for fuzzy match
            matched = jobs[best_idx]
            logger.info(
                f"Fuzzy matched '{target_title}' → '{matched['job_title']}' (sim={best_sim:.2f})"
            )
            return matched

        return None

    def compute_skill_gap(
        self,
        user_skills: list[str],
        target_job_title: str,
    ) -> dict:
        """
        Compare user's skills against job requirements.
        Returns completion score, matched skills, missing skills.
        """
        job = self.find_job_by_title(target_job_title)
        if job is None:
            return {
                "error": f"Job '{target_job_title}' not found in dataset.",
                "found": False,
            }

        required_skills: list[str] = job.get("skills", [])
        required_certs: list[str] = job.get("certifications", [])

        # Normalize both lists
        user_normalized = set(self.normalizer.normalize(s) for s in user_skills)
        user_normalized.discard(None)

        required_normalized = [
            (self.normalizer.normalize(s) or s) for s in required_skills
        ]

        # Exact skill matching
        matched: list[str] = []
        missing: list[str] = []
        for raw_req, norm_req in zip(required_skills, required_normalized):
            if norm_req in user_normalized:
                matched.append(raw_req)
            else:
                # Semantic fuzzy match: check if user has a semantically similar skill
                if self._semantic_match(norm_req, list(user_normalized)):
                    matched.append(raw_req)
                else:
                    missing.append(raw_req)

        total_required = len(required_skills)
        completion_score = (
            round(len(matched) / total_required * 100, 1) if total_required > 0 else 0.0
        )

        return {
            "found": True,
            "target_job": job["job_title"],
            "required_skills": required_skills,
            "required_certifications": required_certs,
            "matched_skills": matched,
            "missing_skills": missing,
            "completion_score": completion_score,
            "match_count": len(matched),
            "total_required": total_required,
        }

    def _semantic_match(self, target_skill: str, user_skills: list[str], threshold: float = 0.82) -> bool:
        """
        Returns True if any user skill is semantically close enough to the target.
        Uses embedding cosine similarity as the measure.
        """
        if not user_skills or not target_skill:
            return False
        try:
            target_emb = self.embedder.embed(target_skill)
            user_embeddings = self.embedder.embed_batch(list(user_skills))
            sims = [float(np.dot(target_emb, u)) for u in user_embeddings]
            return max(sims) >= threshold
        except Exception:
            return False
