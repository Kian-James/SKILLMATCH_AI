"""
services/resume_service.py
Full resume analysis pipeline: OCR → NLP → Embeddings → FAISS → Unified Score

The unified match score for each job is computed by blending:
  - Skill-by-skill match score  (SKILL_WEIGHT)
  - FAISS embedding similarity  (FAISS_WEIGHT)

This ensures the displayed % is always consistent with the skill chips shown.
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
        SEMANTIC_CERT_THRESHOLD,
        ZERO_SKILL_SCORE_CAP,
        VERBOSE_MATCH_LOGGING,
    )
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matching_config import (
        SKILL_WEIGHT,
        FAISS_WEIGHT,
        CERT_BONUS_MAX,
        SEMANTIC_CERT_THRESHOLD,
        ZERO_SKILL_SCORE_CAP,
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
            unified = self._compute_unified_score(
                user_skills=skills,
                user_certs=certifications,
                job=job,
                faiss_score=job.get("faiss_score", job.get("match_score", 0.0)),
            )
            job_entry = {
                "job_title": job["job_title"],
                "match_score": unified["final_score_pct"],   # the one % shown in UI
                "faiss_score_raw": job["faiss_score"],        # for debugging
                "skill_score_raw": unified["skill_score"],    # for debugging
                "skills": job.get("skills", []),
                "certifications": job.get("certifications", []),
            }
            scored_jobs.append(job_entry)
            if VERBOSE_MATCH_LOGGING:
                logger.info(
                    f"  {job['job_title']!r}: "
                    f"skill={unified['skill_score']:.1%} "
                    f"faiss={job.get('faiss_score', 0):.3f} "
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
        Returns skill_score (0-1), cert_bonus (0-1), final_score_pct (0-100).
        """
        required_skills: list[str] = job.get("skills", [])
        required_certs: list[str] = job.get("certifications", [])

        if not required_skills:
            return {"skill_score": 0.0, "cert_bonus": 0.0, "final_score_pct": 0.0}

        # Build user skill keyword set
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

        # Count matched skills
        matched = 0
        for req_skill in required_skills:
            norm_req = (self.normalizer.normalize(req_skill) or req_skill).lower()

            # Exact
            if norm_req in user_normalized:
                matched += 1
                continue

            # Keyword overlap
            req_words = {w for w in req_skill.lower().split() if len(w) > 2}
            if req_words and len(req_words & user_keywords) / len(req_words) >= 0.6:
                matched += 1
                continue

            # Semantic
            try:
                req_emb = self.embedder.embed(req_skill)
                if user_skills:
                    user_embs = self.embedder.embed_batch(user_skills)
                    sims = [float(np.dot(req_emb, u)) for u in user_embs]
                    if max(sims) >= 0.75:
                        matched += 1
                        continue
            except Exception:
                pass

        skill_score = matched / len(required_skills)

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

        # Blend
        blended = skill_score * SKILL_WEIGHT + faiss_score * FAISS_WEIGHT

        # Cap at zero if no skills matched
        if matched == 0:
            blended = min(blended, ZERO_SKILL_SCORE_CAP)

        final_pct = round(min((blended + cert_bonus) * 100, 100.0), 1)

        return {
            "skill_score": skill_score,
            "cert_bonus": cert_bonus,
            "final_score_pct": final_pct,
        }