"""
services/job_service.py
Skill gap detection with unified, calibrated scoring.

ROOT CAUSE FIX:
  Before: FAISS score and skill match score were computed independently.
  The displayed % came from FAISS (resume embedding similarity) while the
  skill chips showed 0 matched — making the score look wrong (60% / 0 skills).

  Now: Score = skill_match * SKILL_WEIGHT + faiss_score * FAISS_WEIGHT
  If 0 skills matched, score is capped at ZERO_SKILL_SCORE_CAP (default 0%).
  This makes the displayed % always consistent with the skill chips shown.

Matching pipeline per required skill:
  1. Exact normalized match
  2. Keyword overlap (configurable threshold)
  3. Semantic embedding (configurable threshold)

All thresholds live in matching_config.py.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from services.firebase_service import fetch_all_jobs
from ai_modules.embedding_engine import EmbeddingEngine
from utils.skill_normalizer import SkillNormalizer
try:
    from matching_config import (
        KEYWORD_OVERLAP_THRESHOLD,
        SEMANTIC_SKILL_THRESHOLD,
        SEMANTIC_CERT_THRESHOLD,
        JOB_TITLE_MATCH_THRESHOLD,
        SKILL_WEIGHT,
        FAISS_WEIGHT,
        CERT_BONUS_MAX,
        ZERO_SKILL_SCORE_CAP,
        FAISS_SKILL_SCALE_THRESHOLD,
        VERBOSE_MATCH_LOGGING,
    )
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matching_config import (
        KEYWORD_OVERLAP_THRESHOLD,
        SEMANTIC_SKILL_THRESHOLD,
        SEMANTIC_CERT_THRESHOLD,
        JOB_TITLE_MATCH_THRESHOLD,
        SKILL_WEIGHT,
        FAISS_WEIGHT,
        CERT_BONUS_MAX,
        ZERO_SKILL_SCORE_CAP,
        FAISS_SKILL_SCALE_THRESHOLD,
        VERBOSE_MATCH_LOGGING,
    )


class JobService:

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.normalizer = SkillNormalizer()

    # ── Job lookup ──────────────────────────────────────────────────────────────

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
        title_embeddings = self.embedder.embed_batch([j["job_title"] for j in jobs])
        sims = [float(np.dot(target_emb, t)) for t in title_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= JOB_TITLE_MATCH_THRESHOLD:
            matched = jobs[best_idx]
            logger.info(
                f"Fuzzy title match: '{target_title}' → '{matched['job_title']}' "
                f"(sim={best_sim:.3f}, threshold={JOB_TITLE_MATCH_THRESHOLD})"
            )
            return matched

        logger.warning(
            f"No job match found for '{target_title}' "
            f"(best sim={best_sim:.3f} < {JOB_TITLE_MATCH_THRESHOLD})"
        )
        return None

    # ── Skill gap ───────────────────────────────────────────────────────────────

    def compute_skill_gap(
            self,
            user_skills: list[str],
            target_job_title: str,
            user_certifications: list[str] | None = None,
            faiss_score: float | None = None,  # pass in if available (from FAISS search)
    ) -> dict:
        """
        Compute skill gap and unified match score.

        faiss_score: cosine similarity from FAISS (0-1). If provided, it is
                     blended into the final score. If None, score is purely
                     based on skill matching.
        """
        job = self.find_job_by_title(target_job_title)
        if job is None:
            return {"error": f"Job '{target_job_title}' not found.", "found": False}

        required_skills: list[str] = job.get("skills", [])
        required_certs: list[str] = job.get("certifications", [])
        user_certs = user_certifications or []

        # Build user skill lookup structures
        user_normalized: set[str] = set()
        user_keywords: set[str] = set()

        for s in user_skills:
            norm = self.normalizer.normalize(s)
            if norm:
                user_normalized.add(norm.lower())
            for word in s.lower().split():
                if len(word) > 2:
                    user_keywords.add(word)

        # Add cert keywords too (Cisco ITF+ cert helps with networking roles etc.)
        for cert in user_certs:
            for word in cert.lower().split():
                if len(word) > 3:
                    user_keywords.add(word)

        # ── Match each required skill ─────────────────────────────────────────
        matched: list[str] = []
        missing: list[str] = []
        match_log: list[str] = []

        for req_skill in required_skills:
            norm_req = (self.normalizer.normalize(req_skill) or req_skill).lower()
            matched_by: str | None = None

            # 1. Exact normalized match
            if norm_req in user_normalized:
                matched_by = "exact"

            # 2. Keyword overlap
            if not matched_by:
                req_words = {w for w in req_skill.lower().split() if len(w) > 2}
                if req_words:
                    overlap_ratio = len(req_words & user_keywords) / len(req_words)
                    if overlap_ratio >= KEYWORD_OVERLAP_THRESHOLD:
                        matched_by = f"keyword_overlap({overlap_ratio:.0%})"

            # 3. Semantic embedding
            if not matched_by:
                if self._semantic_match(req_skill, list(user_skills), SEMANTIC_SKILL_THRESHOLD):
                    matched_by = f"semantic(≥{SEMANTIC_SKILL_THRESHOLD})"

            if matched_by:
                matched.append(req_skill)
                match_log.append(f"  ✓ {req_skill!r} [{matched_by}]")
            else:
                missing.append(req_skill)
                match_log.append(f"  ✗ {req_skill!r} [no match]")

        # ── Certification bonus ───────────────────────────────────────────────
        cert_matched = 0
        if required_certs and user_certs:
            for req_cert in required_certs:
                if self._semantic_match(req_cert, user_certs, SEMANTIC_CERT_THRESHOLD):
                    cert_matched += 1

        # ── Unified score calculation ─────────────────────────────────────────
        total_required = len(required_skills)
        skill_score = len(matched) / total_required if total_required > 0 else 0.0

        # Cert bonus (proportional, capped at CERT_BONUS_MAX)
        cert_bonus = 0.0
        if required_certs:
            cert_bonus = (cert_matched / len(required_certs)) * CERT_BONUS_MAX

        # Blend skill + FAISS scores.
        # FAISS contribution is scaled by how many skills matched so it cannot
        # inflate the score when only 1-2 skills are present (mirrors resume_service fix).
        if faiss_score is not None:
            faiss_multiplier = (
                min(1.0, skill_score / FAISS_SKILL_SCALE_THRESHOLD)
                if FAISS_SKILL_SCALE_THRESHOLD > 0 else 1.0
            )
            blended = skill_score * SKILL_WEIGHT + faiss_score * FAISS_WEIGHT * faiss_multiplier
        else:
            blended = skill_score  # skill-only when no FAISS score is available

        # Cap at ZERO_SKILL_SCORE_CAP when no skills matched
        if len(matched) == 0:
            blended = min(blended, ZERO_SKILL_SCORE_CAP)

        final_score = round(min((blended + cert_bonus) * 100, 100.0), 1)

        # ── Logging ───────────────────────────────────────────────────────────
        if VERBOSE_MATCH_LOGGING:
            faiss_mult_val = (
                min(1.0, skill_score / FAISS_SKILL_SCALE_THRESHOLD)
                if faiss_score is not None and FAISS_SKILL_SCALE_THRESHOLD > 0 else 1.0
            )
            logger.info(
                f"\n{'='*50}\n"
                f"SKILL GAP: {job['job_title']!r}\n"
                f"  User skills    : {user_skills}\n"
                f"  Required skills: {required_skills}\n"
                f"  Skill matches  :\n" + "\n".join(match_log) + "\n"
                                                                 f"  Skill score    : {len(matched)}/{total_required} = {skill_score:.1%}\n"
                                                                 f"  FAISS score    : {faiss_score:.3f if faiss_score is not None else 'N/A'}\n"
                                                                 f"  FAISS mult     : {faiss_mult_val:.2f} (scale threshold={FAISS_SKILL_SCALE_THRESHOLD})\n"
                                                                 f"  Cert bonus     : +{cert_bonus:.1%} ({cert_matched}/{len(required_certs)} certs)\n"
                                                                 f"  Zero-skill cap : {'applied' if len(matched)==0 else 'not needed'}\n"
                                                                 f"  FINAL SCORE    : {final_score}%\n"
                                                                 f"  Config weights : skill={SKILL_WEIGHT} faiss={FAISS_WEIGHT}\n"
                                                                 f"{'='*50}"
            )
        else:
            logger.info(
                f"Skill gap '{job['job_title']}': "
                f"{len(matched)}/{total_required} skills, "
                f"certs={cert_matched}/{len(required_certs)}, "
                f"score={final_score}%"
            )

        return {
            "found": True,
            "target_job": job["job_title"],
            "required_skills": required_skills,
            "required_certifications": required_certs,
            "matched_skills": matched,
            "missing_skills": missing,
            "completion_score": final_score,
            "match_count": len(matched),
            "total_required": total_required,
        }

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _semantic_match(
            self,
            target: str,
            candidates: list[str],
            threshold: float,
    ) -> bool:
        if not candidates or not target:
            return False
        try:
            target_emb = self.embedder.embed(target)
            cand_embeddings = self.embedder.embed_batch(list(candidates))
            sims = [float(np.dot(target_emb, c)) for c in cand_embeddings]
            best = max(sims)
            return best >= threshold
        except Exception as e:
            logger.warning(f"Semantic match error: {e}")
            return False