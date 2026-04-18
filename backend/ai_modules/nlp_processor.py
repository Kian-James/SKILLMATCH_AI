"""
services/resume_service.py
Full resume analysis pipeline: OCR → NLP → Embed → FAISS → Score

MEMORY NOTES:
  - SkillClusterer is a module-level singleton (not per-request instance)
  - OCR correction vocabulary embeddings are module-level (computed once)
  - SkillNormalizer canonical embs are lazy, cached in the normalizer instance
  - No skill embeddings stored in job metadata dicts — all via FAISS skill_index
  - ctx embeddings fetched from compact float16 matrix in FAISSIndex
"""

from __future__ import annotations

import gc
import math

import numpy as np
from loguru import logger

from ai_modules.ocr_extractor import extract_text, extract_text_flagged
from ai_modules.nlp_processor import NLPProcessor
from ai_modules.embedding_engine import EmbeddingEngine
from ai_modules.faiss_index import FAISSIndex
from ai_modules.skill_clusterer import SkillClusterer
from utils.skill_normalizer import SkillNormalizer, _CANONICAL_LIST
from ai_modules.nlp_processor import _FIRESTORE_SKILLS

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

_SIGMOID_CENTER = 0.5
_SIGMOID_SCALE  = 8.0
_CONF_MIN       = 0.7

_OCR_CORR_LOW  = 0.88
_OCR_CORR_HIGH = 0.995

# Common English words to skip during OCR correction candidate selection.
# These will never be misread skill names — filtering them avoids unnecessary
# embed calls and keeps the candidate batch small.
_COMMON_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy",
    "did", "she", "too", "use", "with", "this", "that", "have", "from",
    "they", "will", "been", "more", "when", "made", "like", "time", "just",
    "know", "take", "into", "year", "your", "good", "some", "could", "them",
    "than", "then", "look", "only", "come", "over", "also", "back", "after",
    "work", "first", "well", "even", "want", "because", "these", "give",
    "most", "such", "make", "any", "said", "each", "which", "their", "about",
    "would", "there", "other", "many", "very", "should", "through", "before",
    "where", "right", "while", "large", "often", "must", "high", "every",
    "near", "both", "those", "always", "between", "experience", "strong",
    "years", "skills", "ability", "team", "project", "projects", "using",
    "based", "tools", "system", "systems", "design", "business", "user",
    "users", "software", "technology", "technical", "knowledge", "understanding",
    "responsible", "developing", "working", "including", "support", "build",
    "built", "role", "environment", "across", "within", "multiple", "various",
    "excellent", "required", "development", "management", "services", "solutions",
    "proficient", "familiar", "exposure", "hands", "demonstrated", "proven",
})

# ── Module-level singletons (shared across requests, allocated once) ──────────
_shared_embedder:   EmbeddingEngine | None = None
_shared_clusterer:  SkillClusterer  | None = None
_shared_normalizer: SkillNormalizer | None = None

# OCR correction: vocabulary comes from Firestore job skills (_FIRESTORE_SKILLS),
# populated when NLPProcessor.build_matcher_from_firestore() runs at warmup.
_ocr_vocab_embs: np.ndarray | None = None   # (n_vocab, dim) float32


def _get_ocr_vocab_embs() -> np.ndarray | None:
    """
    Embedding matrix for OCR correction vocabulary.
    Source: _FIRESTORE_SKILLS (all skills from Firestore job corpus).
    Returns None if vocabulary not yet loaded (warmup hasn't run yet).
    """
    global _ocr_vocab_embs
    vocab = _FIRESTORE_SKILLS or _CANONICAL_LIST
    if not vocab:
        return None
    # Rebuild if vocab grew (e.g. after a FAISS index rebuild adds new jobs)
    if _ocr_vocab_embs is None or _ocr_vocab_embs.shape[0] != len(vocab):
        logger.debug(f"Computing OCR vocab embeddings ({len(vocab)} Firestore skills)...")
        _ocr_vocab_embs = _get_embedder().embed_batch(vocab)
    return _ocr_vocab_embs

def _get_embedder() -> EmbeddingEngine:
    global _shared_embedder
    if _shared_embedder is None:
        _shared_embedder = EmbeddingEngine()
    return _shared_embedder


def _get_clusterer() -> SkillClusterer:
    global _shared_clusterer
    if _shared_clusterer is None:
        _shared_clusterer = SkillClusterer()
    return _shared_clusterer


def _get_normalizer() -> SkillNormalizer:
    global _shared_normalizer
    if _shared_normalizer is None:
        _shared_normalizer = SkillNormalizer()
    return _shared_normalizer


def _get_ocr_vocab_embs() -> np.ndarray:
    global _ocr_vocab_embs
    if _ocr_vocab_embs is None:
        logger.debug("Computing OCR correction vocabulary embeddings (once)...")
        _ocr_vocab_embs = _get_embedder().embed_batch(_ocr_vocab)
    return _ocr_vocab_embs


class ResumeService:

    def __init__(self, faiss_index: FAISSIndex):
        self.faiss      = faiss_index
        self.nlp        = NLPProcessor()
        self.embedder   = _get_embedder()    # shared singleton
        self.normalizer = _get_normalizer()  # shared singleton
        self.clusterer  = _get_clusterer()   # shared singleton

    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, file_bytes: bytes, content_type: str) -> dict:
        # Step 1: OCR — returns (text, used_tesseract)
        logger.info("Step 1: OCR")
        raw_text, extracted_via_ocr = extract_text_flagged(file_bytes, content_type)
        if not raw_text or len(raw_text.strip()) < 20:
            raise ValueError("Could not extract meaningful text from the uploaded file.")

        # Step 1b: OCR correction — only for image/scan paths, not clean PDF text.
        # pdfplumber produces clean text; Tesseract produces noisy text.
        # We flag which path was used so we can skip correction for clean extractions.
        if extracted_via_ocr:
            logger.info("Step 1b: OCR correction (scan/image detected)")
            raw_text = self._ocr_correct(raw_text)
        else:
            logger.info("Step 1b: OCR correction skipped (clean PDF text)")

        # Step 2: NLP
        logger.info("Step 2: NLP")
        extracted      = self.nlp.process(raw_text)
        skills         = extracted["skills"]
        certifications = extracted["certifications"]

        if VERBOSE_MATCH_LOGGING:
            logger.info(f"  Skills : {skills}")
            logger.info(f"  Certs  : {certifications}")

        # Step 3: Cluster-aware embedding
        logger.info("Step 3: Cluster-aware embedding")
        user_embs = self._embed_user_profile_clustered(skills, certifications)

        # Step 4: FAISS retrieval + MMR
        logger.info("Step 4: FAISS retrieval + MMR")
        faiss_results = self.faiss.search(user_embs["query"], top_k=5)

        # Step 5: Scoring
        logger.info("Step 5: Scoring")
        scored_jobs = []
        for job in faiss_results:
            raw_faiss = job.get("faiss_score", 0.0)
            if raw_faiss < FAISS_MIN_SCORE_THRESHOLD:
                logger.debug(f"Skip '{job['job_title']}': faiss={raw_faiss:.3f}")
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

    # ── OCR correction ────────────────────────────────────────────────────────

    def _ocr_correct(self, text: str) -> str:
        """
        Correct likely OCR misreads of skill names.
        Vocab source: _FIRESTORE_SKILLS (skills from Firestore job corpus).
        One embed_batch call for all candidates — fast, not one call per word.
        """
        try:
            vocab_embs = _get_ocr_vocab_embs()
            if vocab_embs is None:
                return text   # vocab not loaded yet — skip silently

            vocab  = _FIRESTORE_SKILLS or _CANONICAL_LIST
            words  = text.split()

            candidate_positions: list[int] = []
            candidate_tokens:    list[str] = []

            for i, word in enumerate(words):
                clean = word.strip(".,;:()[]\"'")
                if len(clean) < 3 or not any(c.isalpha() for c in clean):
                    continue
                if clean.islower() and len(clean) > 12:
                    continue
                if clean.lower() in _COMMON_WORDS:
                    continue
                candidate_positions.append(i)
                candidate_tokens.append(clean)

            if not candidate_tokens:
                return text

            candidate_embs = self.embedder.embed_batch(candidate_tokens)
            sim_matrix     = candidate_embs @ vocab_embs.T

            corrections = 0
            for pos, token, sims in zip(candidate_positions, candidate_tokens, sim_matrix):
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if _OCR_CORR_LOW <= best_sim < _OCR_CORR_HIGH:
                    corrected = vocab[best_idx]
                    logger.debug(f"OCR: '{token}' → '{corrected}' ({best_sim:.3f})")
                    words[pos] = words[pos].replace(token, corrected)
                    corrections += 1

            if corrections:
                logger.info(f"OCR correction: {corrections} token(s) corrected.")
            return " ".join(words)

        except Exception as e:
            logger.warning(f"OCR correction failed (non-fatal): {e}")
            return text

    # ── Cluster-aware embedding ───────────────────────────────────────────────

    def _embed_user_profile_clustered(
            self, skills: list[str], certifications: list[str]
    ) -> dict:
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
                "query":  np.zeros(dim, dtype=np.float32),
                "skills": skill_embs,
                "certs":  cert_embs,
            }

        try:
            clusters    = self.clusterer.cluster_skills(skills)
            domain_vecs = [
                self.embedder.embed_skill_list(cs)
                for cs in clusters.values() if cs
            ]
            query = (
                np.mean(domain_vecs, axis=0).astype(np.float32)
                if domain_vecs
                else skill_embs.mean(axis=0).astype(np.float32)
            )
        except Exception as e:
            logger.warning(f"Clustering failed, using mean: {e}")
            query = skill_embs.mean(axis=0).astype(np.float32)

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        return {"query": query, "skills": skill_embs, "certs": cert_embs}

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_unified_score(
            self,
            user_skills:     list[str],
            user_certs:      list[str],
            user_skill_embs: np.ndarray,
            user_cert_embs:  np.ndarray,
            job:             dict,
            faiss_score:     float = 0.0,
    ) -> dict:
        required_skills: list[str] = job.get("skills", [])
        required_certs:  list[str] = job.get("certifications", [])

        if not required_skills:
            return {
                "skill_score": 0.0, "cert_bonus": 0.0,
                "matched_count": 0, "matched_skills": [],
                "cap_applied": False, "final_score_pct": 0.0,
            }

        # Build user lookup
        user_normalized: set[str] = set()
        user_keywords:   set[str] = set()
        for s in user_skills:
            norm = self.normalizer.normalize(s)
            if norm:
                user_normalized.add(norm.lower())
            for w in s.lower().split():
                if len(w) > 2:
                    user_keywords.add(w)
        for cert in user_certs:
            for w in cert.lower().split():
                if len(w) > 3:
                    user_keywords.add(w)

        # Find the global skill offset for this job so we can look up ctx embeddings
        job_idx = next(
            (i for i, m in enumerate(self.faiss.metadata) if m is job),
            None
        )
        ctx_start = self.faiss.get_skill_global_offset(job_idx) if job_idx is not None else -1

        # Pre-compute semantic similarities for ALL required skills at once.
        # Strategy: fetch this job's skill embedding vectors from the ctx_embs matrix
        # and do ONE matrix multiply: user_embs (n_user, dim) @ job_embs (dim, n_req)
        # → sim_matrix (n_user, n_req). Column j gives all user-skill sims against
        # required skill j. max over rows = best user-skill match for each required skill.
        # Cost: one numpy matmul instead of N_user × N_req FAISS calls.
        per_req_skill_sims: np.ndarray | None = None  # shape (n_req,)
        if user_skill_embs.shape[0] > 0 and ctx_start >= 0 and self.faiss._ctx_embs is not None:
            n_req = len(required_skills)
            ctx_end = ctx_start + n_req
            if ctx_end <= len(self.faiss._ctx_embs):
                # (n_req, dim) float16 → float32
                job_ctx_embs = self.faiss._ctx_embs[ctx_start:ctx_end].astype(np.float32)
                # (n_user, dim) @ (dim, n_req) = (n_user, n_req)
                sim_matrix = user_skill_embs @ job_ctx_embs.T
                # Best user-skill similarity for each required skill
                per_req_skill_sims = sim_matrix.max(axis=0)  # (n_req,)

        matched_count   = 0
        matched_skills: list[str] = []
        matched_weights: list[float] = []
        total_weight    = 0.0
        idf             = self.faiss.idf_weights

        for i, req_skill in enumerate(required_skills):
            norm_req   = (self.normalizer.normalize(req_skill) or req_skill).lower()
            matched_by = None

            # Gate 1: Exact
            if norm_req in user_normalized:
                matched_by = "exact"

            # Gate 2: Keyword overlap
            if not matched_by:
                req_words = {w for w in req_skill.lower().split() if len(w) > 2}
                if req_words:
                    overlap = len(req_words & user_keywords) / len(req_words)
                    if overlap >= KEYWORD_OVERLAP_THRESHOLD:
                        matched_by = "keyword"

            # Gate 3: Semantic — contextual embedding matrix (correct direction)
            # per_req_skill_sims[i] = max cosine sim between any user skill and
            # the contextual embedding of required skill i ("job_title: req_skill")
            # This correctly answers "does any user skill semantically cover req_skill i?"
            if not matched_by and per_req_skill_sims is not None:
                sim = float(per_req_skill_sims[i])
                if sim >= SEMANTIC_SKILL_THRESHOLD:
                    matched_by = f"semantic({sim:.2f})"

            skill_weight = idf.get(req_skill.lower(), 1.0)
            total_weight += skill_weight

            if matched_by:
                matched_count += 1
                matched_skills.append(req_skill)
                matched_weights.append(skill_weight)
                if VERBOSE_MATCH_LOGGING:
                    logger.debug(f"  ✓ '{req_skill}' [{matched_by}] idf={skill_weight:.2f}")
            else:
                if VERBOSE_MATCH_LOGGING:
                    logger.debug(f"  ✗ '{req_skill}' [no match] idf={skill_weight:.2f}")

        skill_score = sum(matched_weights) / total_weight if total_weight > 0 else 0.0

        # Cert bonus
        cert_bonus = 0.0
        if required_certs and user_cert_embs.shape[0] > 0:
            cert_matched = sum(
                1 for rc in required_certs
                if self._cert_matches(rc, user_cert_embs)
            )
            cert_bonus = (cert_matched / len(required_certs)) * CERT_BONUS_MAX

        # FAISS confidence sigmoid multiplier
        sigmoid    = 1.0 / (1.0 + math.exp(-_SIGMOID_SCALE * (faiss_score - _SIGMOID_CENTER)))
        confidence = _CONF_MIN + (1.0 - _CONF_MIN) * sigmoid
        blended    = skill_score * confidence

        # Suppression gates
        cap_applied = False
        if matched_count == 0:
            blended    = min(blended, ZERO_SKILL_SCORE_CAP)
            cert_bonus = 0.0
            cap_applied = True
        elif matched_count < MIN_MATCHED_SKILLS_FOR_SCORE:
            blended    = 0.0
            cert_bonus = 0.0
            cap_applied = True

        final_pct = round(min((blended + cert_bonus) * 100, 100.0), 1)

        return {
            "skill_score":     skill_score,
            "cert_bonus":      cert_bonus,
            "matched_count":   matched_count,
            "matched_skills":  matched_skills,
            "cap_applied":     cap_applied,
            "final_score_pct": final_pct,
            "confidence":      round(confidence, 3),
        }

    def _cert_matches(self, req_cert: str, user_cert_embs: np.ndarray) -> bool:
        try:
            req_emb = self.embedder.embed(req_cert)
            sims    = user_cert_embs @ req_emb
            return float(sims.max()) >= SEMANTIC_CERT_THRESHOLD
        except Exception:
            return False

    def _log_job_result(self, job: dict, result: dict) -> None:
        required    = job.get("skills", [])
        matched_set = set(result["matched_skills"])
        skill_lines = "\n".join(
            f"    {'✓' if s in matched_set else '✗'} {s}" for s in required
        )
        logger.info(
            f"\n{'='*52}\n"
            f"TOP MATCH: {job['job_title']!r}\n"
            f"  Score  : {result['matched_count']}/{len(required)}"
            f" = {result['skill_score']:.1%} (IDF-weighted)\n"
            f"  FAISS  : confidence={result.get('confidence',1.0):.3f}\n"
            f"  Final  : {result['final_score_pct']}%\n"
            f"  Skills :\n{skill_lines}\n"
            f"{'='*52}"
        )