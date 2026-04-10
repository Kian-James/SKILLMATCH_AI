"""
services/resume_service.py
Full resume analysis pipeline: OCR → NLP → Embeddings → FAISS → Unified Score

FIXES (v3):
  FIX 1 — FAISS_MIN_SCORE_THRESHOLD (now 0.35) filters out low-similarity jobs
           before scoring, preventing stale FAISS metadata from showing inflated
           scores like 16% when 0 skills matched (Full Stack Software Engineer).

  FIX 2 — FAISS contribution is now scaled by skill_score via FAISS_SKILL_SCALE_THRESHOLD.
           When only 1-2 skills match, FAISS no longer adds 8-18 percentage points.
           Formula: faiss_contribution = faiss_score * FAISS_WEIGHT
                                       * min(1.0, skill_score / FAISS_SKILL_SCALE_THRESHOLD)

  FIX 3 — _compute_unified_score now returns matched_skills: list[str] (the job's
           required skill labels that were matched). This is returned in the API
           response so the frontend can show green chips consistent with the score.
           Previously only matched_count (int) was returned, causing the UI to show
           "0 matched" while the score was > 0 (e.g. Mobile App Developer 26% / 0 chips).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ai_modules.ocr_extractor import extract_text
from ai_modules.nlp_processor import NLPProcessor
from ai_modules.embedding_engine import EmbeddingEngine
from ai_modules.faiss_index import FAISSIndex
from utils.skill_normalizer import SkillNormalizer
try:
    from matching_config import (
        SKILL_WEIGHT,
        FAISS_WEIGHT,
        CERT_BONUS_MAX,
        KEYWORD_OVERLAP_THRESHOLD,
        SEMANTIC_SKILL_THRESHOLD,
        SEMANTIC_CERT_THRESHOLD,
        ZERO_SKILL_SCORE_CAP,
        FAISS_MIN_SCORE_THRESHOLD,
        FAISS_SKILL_SCALE_THRESHOLD,
        VERBOSE_MATCH_LOGGING,
    )
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matching_config import (
        SKILL_WEIGHT,
        FAISS_WEIGHT,
        CERT_BONUS_MAX,
        KEYWORD_OVERLAP_THRESHOLD,
        SEMANTIC_SKILL_THRESHOLD,
        SEMANTIC_CERT_THRESHOLD,
        ZERO_SKILL_SCORE_CAP,
        FAISS_MIN_SCORE_THRESHOLD,
        FAISS_SKILL_SCALE_THRESHOLD,
        VERBOSE_MATCH_LOGGING,
    )


class ResumeService:

    def __init__(self, faiss_index: FAISSIndex):
        self.faiss = faiss_index
        self.nlp = NLPProcessor()
        self.embedder = EmbeddingEngine()
        self.normalizer = SkillNormalizer()

    def analyze(self, file_bytes: bytes, content_type: str) -> dict:
        """
        Full pipeline:
        1. OCR  — extract raw text
        2. NLP  — extract skills + certifications
        3. Embed — generate skill profile embedding
        4. FAISS — find top 5 similar jobs
        5. Score — compute unified match % per job (skill + FAISS blend)
        """
        # Step 1: OCR
        logger.info("Step 1: OCR")
        raw_text = extract_text(file_bytes, content_type)
        if not raw_text or len(raw_text.strip()) < 20:
            raise ValueError("Could not extract meaningful text from the uploaded file.")

        # Step 2: NLP
        logger.info("Step 2: NLP")
        extracted = self.nlp.process(raw_text)
        skills: list[str] = extracted["skills"]
        certifications: list[str] = extracted["certifications"]

        if VERBOSE_MATCH_LOGGING:
            logger.info(f"  Extracted skills       : {skills}")
            logger.info(f"  Extracted certs        : {certifications}")

        # Step 3: Embed skill profile
        logger.info("Step 3: Embedding")
        skill_embedding = self.embedder.embed_skill_list(skills)
        combined_text = " ".join(skills + certifications)
        combined_embedding = (
            self.embedder.embed(combined_text) if combined_text.strip()
            else skill_embedding
        )
        query_embedding = ((skill_embedding + combined_embedding) / 2.0).astype("float32")
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding /= norm

        # Step 4: FAISS — returns raw cosine scores [0, 1]
        logger.info("Step 4: FAISS search")
        faiss_results = self.faiss.search(query_embedding, top_k=5)

        # Step 5: Unified scoring per job
        logger.info("Step 5: Unified scoring")
        scored_jobs = []
        for job in faiss_results:
            raw_faiss = job.get("faiss_score", job.get("match_score", 0.0))

            # FIX 1: Skip jobs whose FAISS cosine similarity is below threshold.
            # This removes stale-metadata jobs that would otherwise bypass the
            # ZERO_SKILL_SCORE_CAP and appear with inflated FAISS-only scores.
            if raw_faiss < FAISS_MIN_SCORE_THRESHOLD:
                logger.debug(
                    f"Skipping {job['job_title']!r}: faiss={raw_faiss:.3f} < {FAISS_MIN_SCORE_THRESHOLD}"
                )
                continue

            unified = self._compute_unified_score(
                user_skills=skills,
                user_certs=certifications,
                job=job,
                faiss_score=raw_faiss,
            )
            job_entry = {
                "job_title": job["job_title"],
                "match_score": unified["final_score_pct"],
                "faiss_score_raw": raw_faiss,
                "skill_score_raw": unified["skill_score"],
                "matched_skill_count": unified["matched_count"],
                # FIX 3: return the actual matched skill labels for frontend green chips
                "matched_skills": unified["matched_skills"],
                "skills": job.get("skills", []),
                "certifications": job.get("certifications", []),
            }
            scored_jobs.append(job_entry)
            if VERBOSE_MATCH_LOGGING:
                logger.info(
                    f"  {job['job_title']!r}: "
                    f"skill={unified['skill_score']:.1%} ({unified['matched_count']}/{len(job.get('skills', []))}) "
                    f"faiss={raw_faiss:.3f} "
                    f"cap={'applied' if unified['cap_applied'] else 'no'} "
                    f"→ {unified['final_score_pct']}%"
                )

        # Sort by unified score descending
        scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)
        best_score = scored_jobs[0]["match_score"] if scored_jobs else 0.0

        return {
            "skills": skills,
            "certifications": certifications,
            "word_count": extracted["word_count"],
            "top_job_matches": scored_jobs,
            "best_match_score": best_score,
            "query_embedding": query_embedding.tolist(),
        }

    def _compute_unified_score(
            self,
            user_skills: list[str],
            user_certs: list[str],
            job: dict,
            faiss_score: float,
    ) -> dict:
        """
        Compute the unified match score for one job.

        Uses KEYWORD_OVERLAP_THRESHOLD and SEMANTIC_SKILL_THRESHOLD from
        matching_config.py (previously these were hardcoded, causing the
        config values to be silently ignored).

        Returns:
          skill_score     — 0-1 fraction of required skills matched
          cert_bonus      — 0-1 bonus from cert matching
          matched_count   — integer count of matched skills
          matched_skills  — list of required skill labels that were matched (for UI chips)
          cap_applied     — bool, whether ZERO_SKILL_SCORE_CAP fired
          final_score_pct — 0-100 displayed percentage
        """
        required_skills: list[str] = job.get("skills", [])
        required_certs: list[str] = job.get("certifications", [])

        if not required_skills:
            return {
                "skill_score": 0.0, "cert_bonus": 0.0,
                "matched_count": 0, "matched_skills": [],
                "cap_applied": False, "final_score_pct": 0.0,
            }

        # Build user skill lookup structures
        user_keywords: set[str] = set()
        user_normalized: set[str] = set()
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

        # Match each required skill using config thresholds
        matched_count = 0
        matched_skills: list[str] = []   # FIX 3: track which job skills matched
        for req_skill in required_skills:
            norm_req = (self.normalizer.normalize(req_skill) or req_skill).lower()
            matched_by = None

            # 1. Exact normalized match
            if norm_req in user_normalized:
                matched_by = "exact"

            # 2. Keyword overlap — use KEYWORD_OVERLAP_THRESHOLD from config
            if not matched_by:
                req_words = {w for w in req_skill.lower().split() if len(w) > 2}
                if req_words and len(req_words & user_keywords) / len(req_words) >= KEYWORD_OVERLAP_THRESHOLD:
                    matched_by = "keyword"

            # 3. Semantic — use SEMANTIC_SKILL_THRESHOLD from config
            if not matched_by:
                try:
                    req_emb = self.embedder.embed(req_skill)
                    if user_skills:
                        user_embs = self.embedder.embed_batch(user_skills)
                        sims = [float(np.dot(req_emb, u)) for u in user_embs]
                        if max(sims) >= SEMANTIC_SKILL_THRESHOLD:
                            matched_by = "semantic"
                except Exception:
                    pass

            if matched_by:
                matched_count += 1
                matched_skills.append(req_skill)  # FIX 3: add the job's label

        skill_score = matched_count / len(required_skills)

        # Cert bonus
        cert_bonus = 0.0
        if required_certs and user_certs:
            cert_matched = 0
            for req_cert in required_certs:
                try:
                    req_emb = self.embedder.embed(req_cert)
                    user_cert_embs = self.embedder.embed_batch(user_certs)
                    sims = [float(np.dot(req_emb, u)) for u in user_cert_embs]
                    if max(sims) >= SEMANTIC_CERT_THRESHOLD:
                        cert_matched += 1
                except Exception:
                    pass
            cert_bonus = (cert_matched / len(required_certs)) * CERT_BONUS_MAX

        # FIX 2: Scale FAISS contribution by how many skills matched.
        # When skill_score < FAISS_SKILL_SCALE_THRESHOLD, FAISS is scaled down
        # so it cannot add 8-18% when only 1-2 skills match.
        # At 0 skills → multiplier=0 → FAISS adds nothing (consistent with cap).
        faiss_multiplier = min(1.0, skill_score / FAISS_SKILL_SCALE_THRESHOLD) if FAISS_SKILL_SCALE_THRESHOLD > 0 else 1.0
        blended = skill_score * SKILL_WEIGHT + faiss_score * FAISS_WEIGHT * faiss_multiplier

        # Cap everything to zero when no skills matched at all.
        # cert_bonus is also zeroed here — certifications should only boost a
        # score that already has skill matches, never create a score from nothing.
        # Previously cert_bonus was added AFTER this block, which allowed it to
        # bypass the cap and produce e.g. 8% despite 0 matched skills.
        cap_applied = False
        if matched_count == 0:
            blended = min(blended, ZERO_SKILL_SCORE_CAP)
            cert_bonus = 0.0   # ← THE FIX: zero cert bonus when no skills match
            cap_applied = True

        final_pct = round(min((blended + cert_bonus) * 100, 100.0), 1)

        return {
            "skill_score": skill_score,
            "cert_bonus": cert_bonus,
            "matched_count": matched_count,
            "matched_skills": matched_skills,
            "cap_applied": cap_applied,
            "final_score_pct": final_pct,
        }