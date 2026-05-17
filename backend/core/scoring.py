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


def _fit_reason_fallback(
        title:          str,
        matched_skills: list[str],
        cosine_score:   float,
        match_ratio:    float,
        industry:       str,
) -> str:
    """Rule-based fallback used when the LLM call fails."""
    industry_suffix = f" in the {industry} industry" if industry else ""
    if matched_skills:
        showcase   = matched_skills[:3]
        skills_str = ", ".join(showcase)
        tail       = f" and {len(matched_skills) - 3} more" if len(matched_skills) > 3 else ""
        if match_ratio >= 0.75:
            return (
                f"Strong fit{industry_suffix} — your resume covers {skills_str}{tail}, "
                f"matching {round(match_ratio * 100)}% of what {title} requires."
            )
        elif match_ratio >= 0.45:
            return (
                f"Good alignment{industry_suffix} — skills like {skills_str}{tail} "
                f"overlap well with the {title} role."
            )
        else:
            return (
                f"Partial match{industry_suffix} — your experience with {skills_str}{tail} "
                f"is relevant to {title}, though more skills would help."
            )
    if cosine_score >= 0.75:
        return f"Semantically strong match{industry_suffix} for {title}."
    elif cosine_score >= 0.5:
        return f"Moderate semantic fit{industry_suffix} with the {title} role."
    return f"{title} is an aspirational direction; targeted upskilling would improve your fit."


def generate_fit_reason(
        title:          str,
        matched_skills: list[str],
        missing_skills: list[str],
        resume_skills:  list[str],
        resume_text:    str,
        job_description: str,
        cosine_score:   float,
        match_ratio:    float,
        industry:       str = "",
) -> str:
    """
    Generate a single tailored sentence explaining why this specific job
    fits this specific resume, using the LLM with full context.

    Falls back to rule-based generation if the LLM call fails.
    """
    from core.utils import call_llama

    # Trim resume text to a meaningful excerpt (first 600 chars covers summary + early experience)
    resume_excerpt = resume_text[:600].strip()

    # Cap skill lists to avoid bloating the prompt
    matched_str = ", ".join(matched_skills[:10]) if matched_skills else "none"
    missing_str = ", ".join(missing_skills[:5])  if missing_skills else "none"
    all_skills  = ", ".join(resume_skills[:15])  if resume_skills  else "none"

    # Use job description excerpt if available, else fall back to title+industry
    job_context = (job_description[:300].strip() if job_description
                   else f"{title} role in the {industry} industry")

    system = (
        "You are a career advisor writing concise, personalized resume feedback. "
        "Your tone is direct, specific, and encouraging — like a mentor who has "
        "actually read this person's resume. Never use generic phrases like "
        "'strong candidate' or 'great fit'. Always reference something concrete "
        "from the resume or the job. Output exactly one sentence, no more."
    )

    user = (
        f"Job title: {title}\n"
        f"Industry: {industry}\n"
        f"Job description excerpt: {job_context}\n\n"
        f"Candidate's resume excerpt:\n{resume_excerpt}\n\n"
        f"Skills on the resume: {all_skills}\n"
        f"Skills that match this job: {matched_str}\n"
        f"Key skills still missing: {missing_str}\n"
        f"Skill match ratio: {round(match_ratio * 100)}%\n"
        f"Semantic similarity score: {round(cosine_score * 100)}%\n\n"
        "Write exactly one sentence (max 35 words) explaining specifically why "
        "this resume is a good fit for this job — referencing actual skills or "
        "experience from the resume. Do not start with 'Your resume' or 'You'. "
        "Do not use bullet points or line breaks."
    )

    try:
        reason = call_llama(system, user, max_tokens=80)
        # Strip any accidental quotes, newlines, or bullet artifacts
        reason = reason.strip().strip('"\'').replace("\n", " ").strip()
        # Sanity-check: if the model returned something too short or too long, fall back
        word_count = len(reason.split())
        if 5 <= word_count <= 60:
            return reason
        log.warning(f"[FitReason] LLM output outside expected length ({word_count} words) — using fallback.")
    except Exception as exc:
        log.warning(f"[FitReason] LLM call failed: {exc} — using fallback.")

    return _fit_reason_fallback(title, matched_skills, cosine_score, match_ratio, industry)


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

        title       = job.get("title", "")
        industry    = job.get("industry", "")
        job_desc    = job.get("description", "")

        results.append({
            "job_id":          job.get("id", job.get("_id", "")),
            "title":           title,
            "industry":        industry,
            "score":           hybrid,
            "cosine_score":    round(float(cosine_score), 4),
            "match_ratio":     round(match_ratio, 4),
            "gap_count":       len(missing),
            "required_skills": req,
            "matched_skills":  matched,
            "missing_skills":  missing,
            "fit_reason":      generate_fit_reason(
                title           = title,
                matched_skills  = matched,
                missing_skills  = missing,
                resume_skills   = resume_skills,
                resume_text     = resume_text,
                job_description = job_desc,
                cosine_score    = float(cosine_score),
                match_ratio     = match_ratio,
                industry        = industry,
            ),
            "nlp_validation":  fuzzy_skill_verify(
                discovered_list   = resume_skills,
                ground_truth_list = req,
            ),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    log.info(f"[Stage 3] {len(results)} jobs scored; top = {results[0]['score'] if results else 'N/A'}.")
    return results