"""
services/job_service.py
Skill gap detection with improved matching:
- Lowers semantic threshold to catch "Flutter" matching "Flutter and Dart"
- Adds keyword overlap matching as a fallback before semantic matching
- Accounts for certifications in completion score
- Normalizes both user and required skills before comparison
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from services.firebase_service import fetch_all_jobs
from ai_modules.embedding_engine import EmbeddingEngine
from utils.skill_normalizer import SkillNormalizer


class JobService:

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.normalizer = SkillNormalizer()

    def find_job_by_title(self, target_title: str) -> dict | None:
        jobs = fetch_all_jobs()
        if not jobs:
            return None

        target_lower = target_title.lower().strip()

        # 1. Exact match
        for job in jobs:
            if job["job_title"].lower() == target_lower:
                return job

        # 2. Embedding similarity
        target_emb = self.embedder.embed(target_title)
        job_title_texts = [j["job_title"] for j in jobs]
        title_embeddings = self.embedder.embed_batch(job_title_texts)
        sims = [float(np.dot(target_emb, t)) for t in title_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= 0.72:
            matched = jobs[best_idx]
            logger.info(f"Fuzzy matched '{target_title}' → '{matched['job_title']}' (sim={best_sim:.2f})")
            return matched

        return None

    def compute_skill_gap(
            self,
            user_skills: list[str],
            target_job_title: str,
            user_certifications: list[str] | None = None,
    ) -> dict:
        job = self.find_job_by_title(target_job_title)
        if job is None:
            return {"error": f"Job '{target_job_title}' not found in dataset.", "found": False}

        required_skills: list[str] = job.get("skills", [])
        required_certs: list[str] = job.get("certifications", [])
        user_certs = user_certifications or []

        # Build normalised user skill set — includes expanded keywords
        user_normalized: set[str] = set()
        user_keywords: set[str] = set()   # individual words from all user skills
        for s in user_skills:
            norm = self.normalizer.normalize(s)
            if norm:
                user_normalized.add(norm.lower())
            # Add every word from the skill as a keyword
            for word in s.lower().split():
                if len(word) > 2:
                    user_keywords.add(word)

        # Also expand cert names into keywords for matching
        for cert in user_certs:
            for word in cert.lower().split():
                if len(word) > 2:
                    user_keywords.add(word)

        matched: list[str] = []
        missing: list[str] = []

        for req_skill in required_skills:
            norm_req = (self.normalizer.normalize(req_skill) or req_skill).lower()

            # 1. Exact normalized match
            if norm_req in user_normalized:
                matched.append(req_skill)
                continue

            # 2. Keyword overlap — does the required skill's keywords appear in user skills?
            req_words = set(w for w in req_skill.lower().split() if len(w) > 2)
            overlap = req_words & user_keywords
            # Match if majority of meaningful words overlap
            if req_words and len(overlap) / len(req_words) >= 0.6:
                matched.append(req_skill)
                continue

            # 3. Semantic embedding match — lower threshold to 0.75
            if self._semantic_match(req_skill, list(user_skills), threshold=0.75):
                matched.append(req_skill)
                continue

            missing.append(req_skill)

        # Cert bonus — check if user certs semantically match required certs
        cert_matched = 0
        if required_certs and user_certs:
            for req_cert in required_certs:
                if self._semantic_match(req_cert, user_certs, threshold=0.78):
                    cert_matched += 1

        total_required = len(required_skills)
        skill_score = len(matched) / total_required if total_required > 0 else 0.0

        # Blend cert bonus into score (certs worth up to 15% bonus)
        cert_bonus = 0.0
        if required_certs:
            cert_bonus = (cert_matched / len(required_certs)) * 0.15

        completion_score = round(min((skill_score + cert_bonus) * 100, 100.0), 1)

        logger.info(
            f"Skill gap for '{job['job_title']}': "
            f"{len(matched)}/{total_required} skills matched, "
            f"{cert_matched}/{len(required_certs)} certs matched, "
            f"score={completion_score}%"
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

    def _semantic_match(
            self,
            target_skill: str,
            user_skills: list[str],
            threshold: float = 0.75,
    ) -> bool:
        """
        Returns True if any user skill is semantically close enough to the target.
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