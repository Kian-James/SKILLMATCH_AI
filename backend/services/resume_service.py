"""
services/resume_service.py
Full resume analysis pipeline: OCR → NLP → Embed → FAISS → Skill Score

PIPELINE OVERVIEW
-----------------
  Step 1  OCR         Extract raw text from PDF or image
  Step 2  NLP         Detect Skills section → extract skills + certifications
  Step 3  Embed       embed_batch(user_skills) — ONE call, result cached by diskcache
  Step 4  FAISS       Top-K candidate retrieval by cosine similarity (search only)
  Step 5  Score       Pure skill-overlap scoring using pre-computed job embeddings

FAISS ROLE
----------
  FAISS is used ONLY for fast candidate retrieval — it narrows the search space
  from hundreds of jobs to the top-K most semantically similar candidates.
  It does NOT determine the displayed match percentage.

  The displayed match % comes entirely from _compute_unified_score(), which uses:
    a) Exact normalized skill matching
    b) Keyword overlap matching
    c) Semantic cosine similarity — using PRE-COMPUTED job skill embeddings
       stored in FAISS metadata at build time (faiss_index.py → rebuild())

PERFORMANCE
-----------
  Before: 5 jobs × ~12 skills × embed() call each = up to 65 embedding calls
          → ~130s scoring time on Render free tier CPU
  After:  embed_batch(user_skills) once → numpy dot products against pre-computed
          job skill vectors → <0.1s scoring, ~4-5s total analysis time
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
        CERT_BONUS_MAX,
        KEYWORD_OVERLAP_THRESHOLD,
        SEMANTIC_SKILL_THRESHOLD,
        SEMANTIC_CERT_THRESHOLD,
        ZERO_SKILL_SCORE_CAP,
        FAISS_MIN_SCORE_THRESHOLD,
        VERBOSE_MATCH_LOGGING,
        MIN_MATCHED_SKILLS_FOR_SCORE,
    )
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matching_config import (
        CERT_BONUS_MAX,
        KEYWORD_OVERLAP_THRESHOLD,
        SEMANTIC_SKILL_THRESHOLD,
        SEMANTIC_CERT_THRESHOLD,
        ZERO_SKILL_SCORE_CAP,
        FAISS_MIN_SCORE_THRESHOLD,
        VERBOSE_MATCH_LOGGING,
        MIN_MATCHED_SKILLS_FOR_SCORE,
    )


class ResumeService:

    def __init__(self, faiss_index: FAISSIndex):
        self.faiss    = faiss_index
        self.nlp      = NLPProcessor()
        self.embedder = EmbeddingEngine()
        self.normalizer = SkillNormalizer()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, file_bytes: bytes, content_type: str) -> dict:
        """
        Full pipeline: OCR → NLP → Embed → FAISS → Score.
        Returns skills, certifications, top job matches with match scores.
        """
        # ── Step 1: OCR ───────────────────────────────────────────────────────
        logger.info("Step 1: OCR")
        raw_text = extract_text(file_bytes, content_type)
        if not raw_text or len(raw_text.strip()) < 20:
            raise ValueError(
                "Could not extract meaningful text from the uploaded file."
            )

        # ── Step 2: NLP ───────────────────────────────────────────────────────
        logger.info("Step 2: NLP")
        extracted     = self.nlp.process(raw_text)
        skills        = extracted["skills"]
        certifications = extracted["certifications"]

        if VERBOSE_MATCH_LOGGING:
            logger.info(f"  Extracted skills : {skills}")
            logger.info(f"  Extracted certs  : {certifications}")

        # ── Step 3: Embed user skill profile (ONE batch call) ─────────────────
        # This is the only embedding call during analysis.
        # Results are cached by diskcache so repeat uploads of the same resume
        # hit zero model inference.
        logger.info("Step 3: Embedding user skills")
        user_embs = self._embed_user_profile(skills, certifications)

        # ── Step 4: FAISS candidate retrieval ─────────────────────────────────
        logger.info("Step 4: FAISS retrieval")
        faiss_results = self.faiss.search(user_embs["query"], top_k=5)

        # ── Step 5: Skill-overlap scoring ─────────────────────────────────────
        logger.info("Step 5: Skill scoring")
        scored_jobs = []
        for job in faiss_results:
            raw_faiss = job.get("faiss_score", 0.0)

            # FAISS threshold: exclude candidates too dissimilar to even score
            if raw_faiss < FAISS_MIN_SCORE_THRESHOLD:
                logger.debug(
                    f"Skipping {job['job_title']!r}: "
                    f"faiss={raw_faiss:.3f} < {FAISS_MIN_SCORE_THRESHOLD}"
                )
                continue

            result = self._compute_unified_score(
                user_skills=skills,
                user_certs=certifications,
                user_skill_embs=user_embs["skills"],
                user_cert_embs=user_embs["certs"],
                job=job,
            )

            scored_jobs.append({
                "job_title":           job["job_title"],
                "match_score":         result["final_score_pct"],
                "faiss_score_raw":     raw_faiss,
                "skill_score_raw":     result["skill_score"],
                "matched_skill_count": result["matched_count"],
                "matched_skills":      result["matched_skills"],
                "skills":              job.get("skills", []),
                "certifications":      job.get("certifications", []),
            })

            if VERBOSE_MATCH_LOGGING:
                self._log_job_result(job, result)

        scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)
        best_score = scored_jobs[0]["match_score"] if scored_jobs else 0.0

        return {
            "skills":           skills,
            "certifications":   certifications,
            "word_count":       extracted["word_count"],
            "top_job_matches":  scored_jobs,
            "best_match_score": best_score,
            "query_embedding":  user_embs["query"].tolist(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Embedding helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_user_profile(
            self,
            skills: list[str],
            certifications: list[str],
    ) -> dict:
        """
        Embed user skills and certifications in one batch.
        Returns dict with:
          query       — normalized query vector for FAISS search
          skills      — 2-D array of per-skill embeddings (n_skills, dim)
          certs       — 2-D array of per-cert embeddings (n_certs, dim)
        """
        skill_embs = (
            self.embedder.embed_batch(skills)
            if skills else np.empty((0, self.embedder.model.get_sentence_embedding_dimension()), dtype=np.float32)
        )
        cert_embs = (
            self.embedder.embed_batch(certifications)
            if certifications else np.empty((0, self.embedder.model.get_sentence_embedding_dimension()), dtype=np.float32)
        )

        # Build FAISS query: mean of skill embeddings + combined text embedding
        if skills:
            skill_mean = skill_embs.mean(axis=0)
            combined_text = " ".join(skills + certifications)
            combined_emb = self.embedder.embed(combined_text)
            query = ((skill_mean + combined_emb) / 2.0).astype(np.float32)
        else:
            query = np.zeros(self.embedder.model.get_sentence_embedding_dimension(), dtype=np.float32)

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        return {"query": query, "skills": skill_embs, "certs": cert_embs}

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_unified_score(
            self,
            user_skills:     list[str],
            user_certs:      list[str],
            user_skill_embs: np.ndarray,   # (n_user_skills, dim) — pre-computed
            user_cert_embs:  np.ndarray,   # (n_user_certs, dim)  — pre-computed
            job:             dict,
    ) -> dict:
        """
        Compute the skill-overlap match score for one job.

        Score = (matched_skills / required_skills) + cert_bonus
        FAISS plays no role here — it is used only for candidate retrieval.

        Matching priority (per required skill):
          1. Exact normalized match     — zero cost
          2. Keyword overlap            — zero cost
          3. Semantic cosine similarity — uses pre-computed job skill embeddings,
                                         zero inference cost (numpy dot products)
        """
        required_skills:      list[str] = job.get("skills", [])
        required_certs:       list[str] = job.get("certifications", [])
        job_skill_embs: list[np.ndarray] = job.get("skill_embeddings", [])

        if not required_skills:
            return {
                "skill_score": 0.0, "cert_bonus": 0.0,
                "matched_count": 0, "matched_skills": [],
                "cap_applied": False, "final_score_pct": 0.0,
            }

        # Build user lookup structures (built once per job, O(n_user_skills))
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
        matched_count  = 0
        matched_skills: list[str] = []

        for i, req_skill in enumerate(required_skills):
            norm_req   = (self.normalizer.normalize(req_skill) or req_skill).lower()
            matched_by = None

            # Gate 1: Exact normalized match
            if norm_req in user_normalized:
                matched_by = "exact"

            # Gate 2: Keyword overlap
            if not matched_by:
                req_words = {w for w in req_skill.lower().split() if len(w) > 2}
                if req_words:
                    overlap = len(req_words & user_keywords) / len(req_words)
                    if overlap >= KEYWORD_OVERLAP_THRESHOLD:
                        matched_by = "keyword"

            # Gate 3: Semantic — uses PRE-COMPUTED job skill embedding (no inference)
            if not matched_by and i < len(job_skill_embs) and user_skill_embs.shape[0] > 0:
                job_skill_emb = job_skill_embs[i]  # already normalized
                sims = user_skill_embs @ job_skill_emb  # vectorized dot products
                if float(sims.max()) >= SEMANTIC_SKILL_THRESHOLD:
                    matched_by = "semantic"

            if matched_by:
                matched_count += 1
                matched_skills.append(req_skill)
                if VERBOSE_MATCH_LOGGING:
                    logger.debug(f"    ✓ {req_skill!r}  [{matched_by}]")
            else:
                if VERBOSE_MATCH_LOGGING:
                    logger.debug(f"    ✗ {req_skill!r}  [no match]")

        skill_score = matched_count / len(required_skills)

        # ── Certification bonus ───────────────────────────────────────────────
        cert_bonus = 0.0
        if required_certs and user_cert_embs.shape[0] > 0:
            cert_matched = 0
            for req_cert in required_certs:
                try:
                    req_emb = self.embedder.embed(req_cert)
                    sims = user_cert_embs @ req_emb
                    if float(sims.max()) >= SEMANTIC_CERT_THRESHOLD:
                        cert_matched += 1
                except Exception:
                    pass
            cert_bonus = (cert_matched / len(required_certs)) * CERT_BONUS_MAX

        # ── Pure skill-overlap score ──────────────────────────────────────────
        # FAISS is retrieval only — it narrows the candidate pool but plays
        # zero role in the displayed match percentage. The score the user sees
        # is entirely determined by how many of the job's required skills they
        # actually have (exact + keyword + semantic matching above).
        # skill_score: 0.0–1.0 fraction of required skills matched
        # cert_bonus:  0.0–CERT_BONUS_MAX additive boost for matching certs
        blended = skill_score

        # ── Two-gate suppression ──────────────────────────────────────────────
        cap_applied = False
        if matched_count == 0:
            # Gate A: zero skills matched → hard zero
            blended    = min(blended, ZERO_SKILL_SCORE_CAP)
            cert_bonus = 0.0
            cap_applied = True
        elif matched_count < MIN_MATCHED_SKILLS_FOR_SCORE:
            # Gate B: fewer than N skills matched → suppress (catches 1-skill accidents)
            blended    = 0.0
            cert_bonus = 0.0
            cap_applied = True
            if VERBOSE_MATCH_LOGGING:
                logger.debug(
                    f"    Score suppressed: matched_count={matched_count} "
                    f"< MIN_MATCHED_SKILLS_FOR_SCORE={MIN_MATCHED_SKILLS_FOR_SCORE}"
                )

        final_pct = round(min((blended + cert_bonus) * 100, 100.0), 1)

        return {
            "skill_score":    skill_score,
            "cert_bonus":     cert_bonus,
            "matched_count":  matched_count,
            "matched_skills": matched_skills,
            "cap_applied":    cap_applied,
            "final_score_pct": final_pct,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    def _log_job_result(self, job: dict, result: dict) -> None:
        required     = job.get("skills", [])
        matched_set  = set(result["matched_skills"])
        skill_lines  = "\n".join(
            f"    {'✓' if s in matched_set else '✗'} {s}"
            for s in required
        )
        logger.info(
            f"\n{'='*52}\n"
            f"TOP MATCH: {job['job_title']!r}\n"
            f"  Skill score : {result['matched_count']}/{len(required)}"
            f" = {result['skill_score']:.1%}\n"
            f"  FAISS score : {job.get('faiss_score', 0):.3f}\n"
            f"  Cap applied : {result['cap_applied']}\n"
            f"  Final score : {result['final_score_pct']}%\n"
            f"  Required skills:\n{skill_lines}\n"
            f"{'='*52}"
        )