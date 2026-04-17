"""
services/resume_service.py
Full resume analysis pipeline: OCR → NLP → Embed → FAISS → Skill Score

PIPELINE (unchanged steps 1-4, upgraded step 5):
  Step 1  OCR         Extract raw text from PDF or image
  Step 2  NLP         Section-aware extraction of skills + certifications
                      + dynamic KNOWN_SKILLS from Firestore job corpus
  Step 3  Embed       Cluster-aware query embedding (domain centroids, not flat mean)
                      embed_batch(user_skills) — disk-cached
  Step 4  FAISS       Top-K candidates via job index + MMR diversity re-ranking
  Step 5  Score       Upgraded unified scoring:
                        Gate 1: Exact normalized match
                        Gate 2: Keyword overlap
                        Gate 3: Skill-index fuzzy match (FAISS skill index, not dot loop)
                        Gate 3b: Contextual embedding match (job_title: skill phrase)
                      IDF-weighted skill score (rare skills count more)
                      FAISS cosine used as continuous confidence multiplier (not binary gate)
                      Cert bonus unchanged

OCR correction pass (new):
  After OCR, each extracted token that almost-matches a known skill
  (cosine sim 0.88–0.99) is auto-corrected before NLP extraction.
  Catches Tesseract misreads like "Pythan" → "Python".
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger

from ai_modules.ocr_extractor import extract_text
from ai_modules.nlp_processor import NLPProcessor
from ai_modules.embedding_engine import EmbeddingEngine
from ai_modules.faiss_index import FAISSIndex
from ai_modules.skill_clusterer import SkillClusterer
from utils.skill_normalizer import SkillNormalizer, SKILL_SYNONYMS

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

# FAISS confidence sigmoid parameters
# Maps FAISS cosine score → confidence multiplier in [0.7, 1.0]
# At faiss_score=0.5: multiplier≈0.70 (30% discount)
# At faiss_score=0.9: multiplier≈0.99 (nearly full credit)
_SIGMOID_CENTER = 0.5
_SIGMOID_SCALE = 8.0
_CONF_MIN = 0.7

# OCR correction: auto-correct tokens this similar to a known skill
_OCR_CORRECTION_THRESHOLD_LOW  = 0.88
_OCR_CORRECTION_THRESHOLD_HIGH = 0.995  # below 1.0 so exact matches aren't re-mapped


class ResumeService:

    def __init__(self, faiss_index: FAISSIndex):
        self.faiss      = faiss_index
        self.nlp        = NLPProcessor()
        self.embedder   = EmbeddingEngine()
        self.normalizer = SkillNormalizer()
        self.clusterer  = SkillClusterer()

        # Canonical skill vocabulary for OCR correction
        self._skill_vocab: list[str] = list(SKILL_SYNONYMS.keys())
        self._skill_vocab_embs: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, file_bytes: bytes, content_type: str) -> dict:
        """
        Full pipeline: OCR → OCR-correct → NLP → Embed (cluster-aware) →
                       FAISS (MMR) → Score (IDF + confidence-weighted).
        """
        # ── Step 1: OCR ───────────────────────────────────────────────────────
        logger.info("Step 1: OCR")
        raw_text = extract_text(file_bytes, content_type)
        if not raw_text or len(raw_text.strip()) < 20:
            raise ValueError(
                "Could not extract meaningful text from the uploaded file."
            )

        # ── Step 1b: OCR correction ────────────────────────────────────────────
        logger.info("Step 1b: OCR correction")
        raw_text = self._ocr_correct(raw_text)

        # ── Step 2: NLP ───────────────────────────────────────────────────────
        logger.info("Step 2: NLP")
        extracted      = self.nlp.process(raw_text)
        skills         = extracted["skills"]
        certifications = extracted["certifications"]

        if VERBOSE_MATCH_LOGGING:
            logger.info(f"  Extracted skills: {skills}")
            logger.info(f"  Extracted certs : {certifications}")

        # ── Step 3: Cluster-aware query embedding ──────────────────────────────
        logger.info("Step 3: Cluster-aware embedding")
        user_embs = self._embed_user_profile_clustered(skills, certifications)

        # ── Step 4: FAISS candidate retrieval (MMR diversity) ─────────────────
        logger.info("Step 4: FAISS retrieval + MMR re-ranking")
        faiss_results = self.faiss.search(user_embs["query"], top_k=5)

        # ── Step 5: Upgraded scoring ───────────────────────────────────────────
        logger.info("Step 5: IDF + confidence-weighted scoring")
        scored_jobs = []
        for job in faiss_results:
            raw_faiss = job.get("faiss_score", 0.0)

            if raw_faiss < FAISS_MIN_SCORE_THRESHOLD:
                logger.debug(
                    f"Skipping '{job['job_title']}': "
                    f"faiss={raw_faiss:.3f} < {FAISS_MIN_SCORE_THRESHOLD}"
                )
                continue

            result = self._compute_unified_score(
                user_skills=skills,
                user_certs=certifications,
                user_skill_embs=user_embs["skills"],
                user_cert_embs=user_embs["certs"],
                job=job,
                faiss_score=raw_faiss,
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
    # OCR correction
    # ──────────────────────────────────────────────────────────────────────────

    def _get_skill_vocab_embs(self) -> np.ndarray:
        if self._skill_vocab_embs is None:
            self._skill_vocab_embs = self.embedder.embed_batch(self._skill_vocab)
        return self._skill_vocab_embs

    def _ocr_correct(self, text: str) -> str:
        """
        Correct likely OCR misreads of skill names using embedding similarity.
        Tokens that are close-but-not-exact to a known canonical skill are replaced.
        Only corrects individual word tokens (not phrases) to avoid false rewrites.
        """
        try:
            vocab_embs = self._get_skill_vocab_embs()
            words = text.split()
            corrected_words = []
            corrections = 0

            for word in words:
                # Only attempt correction on plausibly skill-like tokens
                clean = word.strip(".,;:()[]\"'")
                if len(clean) < 3 or not any(c.isalpha() for c in clean):
                    corrected_words.append(word)
                    continue

                try:
                    emb = self.embedder.embed(clean)
                    sims = vocab_embs @ emb
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])

                    if _OCR_CORRECTION_THRESHOLD_LOW <= best_sim < _OCR_CORRECTION_THRESHOLD_HIGH:
                        corrected = self._skill_vocab[best_idx]
                        logger.debug(
                            f"OCR correction: '{clean}' → '{corrected}' (sim={best_sim:.3f})"
                        )
                        # Preserve surrounding punctuation
                        corrected_words.append(
                            word.replace(clean, corrected)
                        )
                        corrections += 1
                        continue
                except Exception:
                    pass

                corrected_words.append(word)

            if corrections:
                logger.info(f"OCR correction: {corrections} token(s) auto-corrected.")

            return " ".join(corrected_words)

        except Exception as e:
            logger.warning(f"OCR correction pass failed (non-fatal): {e}")
            return text

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster-aware embedding
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_user_profile_clustered(
        self,
        skills: list[str],
        certifications: list[str],
    ) -> dict:
        """
        Build a cluster-aware FAISS query vector.

        Instead of a flat mean of all skill embeddings (which collapses
        multi-domain profiles into a generic blob), cluster skills into
        domains first, then average the per-domain centroids. This preserves
        the shape of a candidate's multi-domain profile.

        Returns:
          query   — (dim,) normalized query vector for FAISS
          skills  — (n_skills, dim) per-skill embeddings
          certs   — (n_certs,  dim) per-cert embeddings
        """
        dim = self.embedder.model.get_sentence_embedding_dimension()

        skill_embs = (
            self.embedder.embed_batch(skills)
            if skills else np.empty((0, dim), dtype=np.float32)
        )
        cert_embs = (
            self.embedder.embed_batch(certifications)
            if certifications else np.empty((0, dim), dtype=np.float32)
        )

        if not skills:
            return {
                "query": np.zeros(dim, dtype=np.float32),
                "skills": skill_embs,
                "certs": cert_embs,
            }

        # Cluster skills → domain centroids
        try:
            clusters = self.clusterer.cluster_skills(skills)
            domain_vecs = []
            for domain_skills in clusters.values():
                if domain_skills:
                    domain_vecs.append(
                        self.embedder.embed_skill_list(domain_skills)
                    )

            if domain_vecs:
                query = np.mean(domain_vecs, axis=0).astype(np.float32)
            else:
                query = skill_embs.mean(axis=0).astype(np.float32)

        except Exception as e:
            logger.warning(f"Clustering failed, falling back to mean embedding: {e}")
            query = skill_embs.mean(axis=0).astype(np.float32)

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
        user_skill_embs: np.ndarray,
        user_cert_embs:  np.ndarray,
        job:             dict,
        faiss_score:     float = 0.0,
    ) -> dict:
        """
        IDF-weighted, confidence-adjusted skill-overlap score.

        Matching gates per required skill (in priority order):
          1. Exact normalized match
          2. Keyword overlap  (>= KEYWORD_OVERLAP_THRESHOLD)
          3a. Skill-index fuzzy match via FAISS skill_index (new)
          3b. Contextual embedding match (job_title: skill phrase)
          3c. Bare embedding dot-product fallback

        IDF weighting: rare skills count more than common ones.
        FAISS confidence: multiplies the skill score (0.7–1.0 range).
        """
        required_skills:       list[str]       = job.get("skills", [])
        required_certs:        list[str]       = job.get("certifications", [])
        job_skill_embs:        list[np.ndarray] = job.get("skill_embeddings", [])
        job_skill_ctx_embs:    list[np.ndarray] = job.get("skill_ctx_embeddings", [])

        if not required_skills:
            return {
                "skill_score": 0.0, "cert_bonus": 0.0,
                "matched_count": 0, "matched_skills": [],
                "cap_applied": False, "final_score_pct": 0.0,
            }

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
        matched_count   = 0
        matched_skills: list[str] = []
        matched_weights: list[float] = []
        total_weight    = 0.0

        idf = self.faiss.idf_weights

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

            # Gate 3a: FAISS skill-index fuzzy match (per-skill search)
            if not matched_by and user_skill_embs.shape[0] > 0:
                # Try each user skill embedding against the skill FAISS index
                for user_emb in user_skill_embs:
                    closest_skill, sim = self.faiss.find_closest_skill(
                        user_emb, threshold=SEMANTIC_SKILL_THRESHOLD
                    )
                    if closest_skill and closest_skill.lower() == norm_req:
                        matched_by = f"skill_index({sim:.2f})"
                        break

            # Gate 3b: Contextual embedding match
            if not matched_by and i < len(job_skill_ctx_embs) and user_skill_embs.shape[0] > 0:
                ctx_emb = job_skill_ctx_embs[i]
                sims_ctx = user_skill_embs @ ctx_emb
                if float(sims_ctx.max()) >= SEMANTIC_SKILL_THRESHOLD:
                    matched_by = "ctx_embed"

            # Gate 3c: Bare embedding dot-product fallback
            if not matched_by and i < len(job_skill_embs) and user_skill_embs.shape[0] > 0:
                job_skill_emb = job_skill_embs[i]
                sims = user_skill_embs @ job_skill_emb
                if float(sims.max()) >= SEMANTIC_SKILL_THRESHOLD:
                    matched_by = "bare_embed"

            # IDF weight for this skill
            skill_weight = idf.get(req_skill.lower(), 1.0)
            total_weight += skill_weight

            if matched_by:
                matched_count += 1
                matched_skills.append(req_skill)
                matched_weights.append(skill_weight)
                if VERBOSE_MATCH_LOGGING:
                    logger.debug(f"  ✓ '{req_skill}' [{matched_by}] (idf={skill_weight:.2f})")
            else:
                if VERBOSE_MATCH_LOGGING:
                    logger.debug(f"  ✗ '{req_skill}' [no match] (idf={skill_weight:.2f})")

        # IDF-weighted skill score
        skill_score = (
            sum(matched_weights) / total_weight
            if total_weight > 0 else 0.0
        )

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

        # ── FAISS confidence multiplier (continuous, not binary) ──────────────
        # sigmoid maps faiss_score → multiplier in [_CONF_MIN, 1.0]
        sigmoid = 1.0 / (1.0 + math.exp(
            -_SIGMOID_SCALE * (faiss_score - _SIGMOID_CENTER)
        ))
        confidence = _CONF_MIN + (1.0 - _CONF_MIN) * sigmoid
        blended = skill_score * confidence

        # ── Two-gate suppression (unchanged) ─────────────────────────────────
        cap_applied = False
        if matched_count == 0:
            blended    = min(blended, ZERO_SKILL_SCORE_CAP)
            cert_bonus = 0.0
            cap_applied = True
        elif matched_count < MIN_MATCHED_SKILLS_FOR_SCORE:
            blended    = 0.0
            cert_bonus = 0.0
            cap_applied = True
            if VERBOSE_MATCH_LOGGING:
                logger.debug(
                    f"  Score suppressed: matched_count={matched_count} "
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
            "confidence":     round(confidence, 3),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    def _log_job_result(self, job: dict, result: dict) -> None:
        required    = job.get("skills", [])
        matched_set = set(result["matched_skills"])
        skill_lines = "\n".join(
            f"    {'✓' if s in matched_set else '✗'} {s}"
            for s in required
        )
        logger.info(
            f"\n{'='*52}\n"
            f"TOP MATCH: {job['job_title']!r}\n"
            f"  Skill score  : {result['matched_count']}/{len(required)}"
            f" = {result['skill_score']:.1%} (IDF-weighted)\n"
            f"  FAISS score  : {job.get('faiss_score', 0):.3f}"
            f"  → confidence={result.get('confidence', 1.0):.3f}\n"
            f"  Cap applied  : {result['cap_applied']}\n"
            f"  Final score  : {result['final_score_pct']}%\n"
            f"  Required skills:\n{skill_lines}\n"
            f"{'='*52}"
        )
