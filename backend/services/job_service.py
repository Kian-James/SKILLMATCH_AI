"""
services/job_service.py
Skill gap detection with IDF-weighted, confidence-adjusted scoring.

Matching pipeline per required skill (same gates as resume_service):
  1. Exact normalized match
  2. Keyword overlap (configurable threshold)
  3. Semantic embedding (configurable threshold)

IDF weighting and FAISS confidence multiplier applied consistently
with resume_service so skill-gap scores are comparable to match scores.
"""

from __future__ import annotations

import math
import time

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
        CERT_BONUS_MAX,
        ZERO_SKILL_SCORE_CAP,
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
        CERT_BONUS_MAX,
        ZERO_SKILL_SCORE_CAP,
        VERBOSE_MATCH_LOGGING,
    )

_SIGMOID_CENTER = 0.5
_SIGMOID_SCALE  = 8.0
_CONF_MIN       = 0.7


class JobService:

    _jobs_cache: list[dict] = []
    _idf_cache:  dict[str, float] = {}
    _cache_ts:   float = 0.0
    _CACHE_TTL:  float = 600.0

    def __init__(self):
        self.embedder   = EmbeddingEngine()
        self.normalizer = SkillNormalizer()

    def _get_jobs(self) -> list[dict]:
        now = time.time()
        if not JobService._jobs_cache or (now - JobService._cache_ts) > JobService._CACHE_TTL:
            JobService._jobs_cache = fetch_all_jobs()
            JobService._cache_ts   = now

            # Recompute IDF weights from fresh job list
            N = len(JobService._jobs_cache)
            if N > 0:
                from collections import Counter
                freq: Counter = Counter()
                for job in JobService._jobs_cache:
                    freq.update(set(s.lower() for s in job.get("skills", [])))
                JobService._idf_cache = {
                    skill: math.log(N / f) for skill, f in freq.items() if f > 0
                }
            logger.info(
                f"Job cache refreshed: {len(JobService._jobs_cache)} jobs, "
                f"{len(JobService._idf_cache)} IDF weights."
            )
        return JobService._jobs_cache

    # ── Job lookup ────────────────────────────────────────────────────────────

    def find_job_by_title(self, target_title: str) -> dict | None:
        jobs = self._get_jobs()
        if not jobs:
            return None

        target_lower = target_title.lower().strip()

        # 1. Exact
        for job in jobs:
            if job["job_title"].lower() == target_lower:
                return job

        # 2. Substring
        for job in jobs:
            if (target_lower in job["job_title"].lower()
                    or job["job_title"].lower() in target_lower):
                logger.info(f"Substring match: '{target_title}' → '{job['job_title']}'")
                return job

        # 3. Embedding similarity
        target_emb    = self.embedder.embed(target_title)
        title_embs    = self.embedder.embed_batch([j["job_title"] for j in jobs])
        sims          = [float(np.dot(target_emb, t)) for t in title_embs]
        best_idx      = int(np.argmax(sims))
        best_sim      = sims[best_idx]

        if best_sim >= JOB_TITLE_MATCH_THRESHOLD:
            matched = jobs[best_idx]
            logger.info(
                f"Fuzzy title match: '{target_title}' → '{matched['job_title']}' "
                f"(sim={best_sim:.3f})"
            )
            return matched

        logger.warning(
            f"No job match for '{target_title}' (best sim={best_sim:.3f} "
            f"< {JOB_TITLE_MATCH_THRESHOLD})"
        )
        return None

    # ── Skill gap ─────────────────────────────────────────────────────────────

    def compute_skill_gap(
        self,
        user_skills:         list[str],
        target_job_title:    str,
        user_certifications: list[str] | None = None,
        faiss_score:         float | None = None,
    ) -> dict:
        """
        IDF-weighted skill gap with optional FAISS confidence multiplier.
        """
        self._get_jobs()   # ensure IDF cache is warm
        job = self.find_job_by_title(target_job_title)
        if job is None:
            return {"error": f"Job '{target_job_title}' not found.", "found": False}

        required_skills: list[str] = job.get("skills", [])
        required_certs:  list[str] = job.get("certifications", [])
        user_certs = user_certifications or []

        # Build user lookup structures
        user_normalized: set[str] = set()
        user_keywords:   set[str] = set()

        for s in user_skills:
            norm = self.normalizer.normalize(s)
            if norm:
                user_normalized.add(norm.lower())
            for word in s.lower().split():
                if len(word) > 2:
                    user_keywords.add(word)
        for cert in user_certs:
            for word in cert.lower().split():
                if len(word) > 3:
                    user_keywords.add(word)

        # ── Match each required skill ─────────────────────────────────────────
        matched: list[str] = []
        missing: list[str] = []
        match_log: list[str] = []
        matched_weights: list[float] = []
        total_weight = 0.0

        idf = JobService._idf_cache

        for req_skill in required_skills:
            norm_req   = (self.normalizer.normalize(req_skill) or req_skill).lower()
            matched_by: str | None = None

            # Gate 1: Exact
            if norm_req in user_normalized:
                matched_by = "exact"

            # Gate 2: Keyword overlap
            if not matched_by:
                req_words = {w for w in req_skill.lower().split() if len(w) > 2}
                if req_words:
                    ratio = len(req_words & user_keywords) / len(req_words)
                    if ratio >= KEYWORD_OVERLAP_THRESHOLD:
                        matched_by = f"keyword_overlap({ratio:.0%})"

            # Gate 3: Semantic embedding
            if not matched_by:
                if self._semantic_match(req_skill, list(user_skills),
                                        SEMANTIC_SKILL_THRESHOLD):
                    matched_by = f"semantic(>={SEMANTIC_SKILL_THRESHOLD})"

            skill_weight = idf.get(req_skill.lower(), 1.0)
            total_weight += skill_weight

            if matched_by:
                matched.append(req_skill)
                matched_weights.append(skill_weight)
                match_log.append(f"  ✓ {req_skill!r} [{matched_by}] (idf={skill_weight:.2f})")
            else:
                missing.append(req_skill)
                match_log.append(f"  ✗ {req_skill!r} [no match] (idf={skill_weight:.2f})")

        # IDF-weighted skill score
        total_required = len(required_skills)
        skill_score = (
            sum(matched_weights) / total_weight if total_weight > 0 else 0.0
        )

        # Cert bonus
        cert_matched = 0
        if required_certs and user_certs:
            for req_cert in required_certs:
                if self._semantic_match(req_cert, user_certs, SEMANTIC_CERT_THRESHOLD):
                    cert_matched += 1
        cert_bonus = 0.0
        if required_certs:
            cert_bonus = (cert_matched / len(required_certs)) * CERT_BONUS_MAX

        # FAISS confidence multiplier (if provided)
        if faiss_score is not None:
            sigmoid = 1.0 / (1.0 + math.exp(
                -_SIGMOID_SCALE * (faiss_score - _SIGMOID_CENTER)
            ))
            confidence = _CONF_MIN + (1.0 - _CONF_MIN) * sigmoid
            blended = skill_score * confidence
        else:
            blended = skill_score

        # Cap when no skills matched
        if len(matched) == 0:
            blended    = min(blended, ZERO_SKILL_SCORE_CAP)
            cert_bonus = 0.0

        final_score = round(min((blended + cert_bonus) * 100, 100.0), 1)

        if VERBOSE_MATCH_LOGGING:
            logger.info(
                f"\n{'='*52}\n"
                f"SKILL GAP: {job['job_title']!r}\n"
                f"  User skills ({len(user_skills)}): {user_skills}\n"
                f"  Required ({total_required}):\n" + "\n".join(match_log) + "\n"
                f"  ─────────────────────────────────────\n"
                f"  IDF-weighted score : {skill_score:.1%}\n"
                f"  FAISS confidence   : "
                f"{faiss_score:.3f if faiss_score is not None else 'N/A'}\n"
                f"  Cert bonus         : +{cert_bonus:.1%}\n"
                f"  FINAL SCORE        : {final_score}%\n"
                f"{'='*52}"
            )

        return {
            "found":                True,
            "target_job":           job["job_title"],
            "required_skills":      required_skills,
            "required_certifications": required_certs,
            "matched_skills":       matched,
            "missing_skills":       missing,
            "completion_score":     final_score,
            "match_count":          len(matched),
            "total_required":       total_required,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _semantic_match(
        self, target: str, candidates: list[str], threshold: float
    ) -> bool:
        if not candidates or not target:
            return False
        try:
            target_emb  = self.embedder.embed(target)
            cand_embs   = self.embedder.embed_batch(list(candidates))
            sims        = [float(np.dot(target_emb, c)) for c in cand_embs]
            return max(sims) >= threshold
        except Exception as e:
            logger.warning(f"Semantic match error: {e}")
            return False
