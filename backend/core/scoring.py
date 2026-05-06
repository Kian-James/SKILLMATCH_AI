"""
core/scoring.py
===============
Stage 3 (FAISS side) — SBERT embedding + hybrid scoring.

Hybrid score = 0.70 * cosine_similarity + 0.30 * skill_match_ratio
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
from tqdm import tqdm

from config import TOP_K
from core.utils import normalize
from core.validation import fuzzy_skill_verify
from core.ocr_correction import get_sbert

log = logging.getLogger(__name__)


def identify_skill_gap(resume_skills: list[str], job_skills: list[str]) -> list[str]:
    """Return skills present in job_skills but absent from resume_skills."""
    have = {normalize(s) for s in resume_skills}
    return [s for s in job_skills if normalize(s) not in have]


def _encode_jobs(jobs_corpus: list[dict]) -> np.ndarray:
    """Encode job documents with SBERT; reuse stored embeddings when available."""
    sbert   = get_sbert()
    vectors: list[np.ndarray] = []

    for job in tqdm(jobs_corpus, desc="Encoding jobs", leave=False):
        stored = job.get("embedding") or []
        if isinstance(stored, list) and len(stored) > 0:
            vec = np.array(stored, dtype=np.float32)
        else:
            text = (
                f"{job.get('title','')} {job.get('description','')} "
                f"{' '.join(job.get('required_skills', []))} "
                f"{' '.join(job.get('required_certifications', []))}"
            )
            vec = sbert.encode(text, normalize_embeddings=True).astype(np.float32)
            job["embedding"] = vec.tolist()
        vectors.append(vec)

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix


def calculate_cosine_similarity(
        resume_text:   str,
        resume_skills: list[str],
        jobs_corpus:   list[dict],
        top_k:         int = TOP_K,
) -> list[dict]:
    """
    Stage 3 — hybrid scoring + per-job NLP validation.

    Returns top_k job dicts sorted by score descending, each including
    an 'nlp_validation' key with per-job accuracy breakdown.
    """
    sbert = get_sbert()

    q_text = f"{resume_text[:500]} {' '.join(resume_skills)}"
    q_vec  = sbert.encode(q_text, normalize_embeddings=True).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)

    job_matrix = _encode_jobs(jobs_corpus)
    dim        = job_matrix.shape[1]
    index      = faiss.IndexFlatIP(dim)
    index.add(job_matrix)

    k                = min(top_k, index.ntotal)
    cos_scores, idxs = index.search(q_vec, k)
    resume_norms     = {normalize(s) for s in resume_skills}
    results: list[dict] = []

    for cosine_score, idx in zip(cos_scores[0], idxs[0]):
        if idx < 0:
            continue
        job     = jobs_corpus[idx]
        req     = job.get("required_skills", [])
        matched = [s for s in req if normalize(s) in resume_norms]
        missing = identify_skill_gap(resume_skills, req)

        match_ratio = len(matched) / len(req) if req else 0.0
        hybrid      = round(0.70 * float(cosine_score) + 0.30 * match_ratio, 4)

        results.append({
            "job_id":          job.get("id", job.get("_id", "")),
            "title":           job.get("title", ""),
            "industry":        job.get("industry", ""),
            "score":           hybrid,
            "cosine_score":    round(float(cosine_score), 4),
            "match_ratio":     round(match_ratio, 4),
            "gap_count":       len(missing),
            "required_skills": req,
            "matched_skills":  matched,
            "missing_skills":  missing,
            "nlp_validation":  fuzzy_skill_verify(
                discovered_list   = resume_skills,
                ground_truth_list = req,
            ),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    log.info(f"[Stage 3] {len(results)} jobs scored; top = {results[0]['score'] if results else 'N/A'}.")
    return results