"""
core/scoring.py
===============
Stage 3 (FAISS side) — SBERT embedding + hybrid scoring.

Hybrid score formula (revised)
-------------------------------
Old formula had a structural ceiling of ~87 and compressed all real-world
scores into a narrow 55–80 band, making the number meaningless as a signal.

New formula:
  1. Rescale cosine similarity from SBERT's real range (0.30–0.90) → 0–1
  2. Use skill match ratio as a MULTIPLIER, not an additive term
     — a candidate missing 80% of skills can't ride a high cosine to a pass
  3. Apply a gap penalty that only activates when >40% of required skills
     are missing, capping at a 50-point drag
  4. Multiply by 100 for a 0–100 output with a meaningful 60-point threshold

Typical score ranges with new formula:
  Strong match   (cosine 0.82, skills 0.90, gap 0.10)  →  ~91
  Good match     (cosine 0.72, skills 0.75, gap 0.25)  →  ~73
  Semantic only  (cosine 0.80, skills 0.30, gap 0.70)  →  ~18
  Weak all round (cosine 0.45, skills 0.20, gap 0.80)  →   ~0
  Perfect        (cosine 0.88, skills 1.00, gap 0.00)  →  ~97

Recommended min_match_score threshold: 60  (was 50)

Changes (LLM removal): removed call_llama from generate_fit_reason.
  The LLM was used to write a personalised one-sentence fit reason but the
  HF Inference API is no longer in use. generate_fit_reason now calls
  _fit_reason_fallback directly — the rule-based generator that was already
  the fallback path. The function signature and all callers are unchanged.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
from rapidfuzz import fuzz as _fuzz
from tqdm import tqdm

from config import TOP_K
from core.utils import normalize
from core.validation import fuzzy_skill_verify
from core.ocr_correction import get_sbert

log = logging.getLogger(__name__)


# ── Scoring constants ─────────────────────────────────────────────────────────
COSINE_MIN        = 0.30   # SBERT inner-product floor for real documents
COSINE_MAX        = 0.90   # SBERT inner-product ceiling for real documents
GAP_PENALTY_FLOOR = 0.40   # gap penalty only activates above this missing-skill ratio
GAP_PENALTY_SCALE = 1.50   # how steeply the penalty grows past the floor
GAP_PENALTY_CAP   = 0.50   # maximum penalty (50 points), prevents negative scores
FUZZY_THRESHOLD   = 85     # rapidfuzz token_sort_ratio threshold, matches evaluator


# ── Skill matching ─────────────────────────────────────────────────────────────
def _fuzzy_skill_match(job_skill: str, resume_skills_raw: list[str]) -> bool:
    """
    Return True if job_skill fuzzy-matches any skill in resume_skills_raw.
    Uses the same 85-point threshold as the offline evaluator so scoring
    and evaluation are consistent.
    """
    jn = normalize(job_skill)
    for rs in resume_skills_raw:
        rn = normalize(rs)
        if jn == rn:
            return True
        if jn in rn or rn in jn:
            return True
        if _fuzz.token_sort_ratio(jn, rn) >= FUZZY_THRESHOLD:
            return True
    return False


def _partition_skills(
        req:               list[str],
        resume_skills_raw: list[str],
) -> tuple[list[str], list[str]]:
    """Split required skills into (matched, missing) using fuzzy matching."""
    matched = [s for s in req if     _fuzzy_skill_match(s, resume_skills_raw)]
    missing = [s for s in req if not _fuzzy_skill_match(s, resume_skills_raw)]
    return matched, missing


# ── Hybrid score ──────────────────────────────────────────────────────────────
def _compute_hybrid_score(
        cosine_sim:  float,
        match_ratio: float,
        gap_ratio:   float,
) -> float:
    """
    Compute a 0–1 hybrid score from the three stage inputs.

    Parameters
    ----------
    cosine_sim  : raw SBERT inner product (normalized vectors), ~0.30–0.90
    match_ratio : fuzzy-matched skills / total required skills, 0–1
    gap_ratio   : missing skills / total required skills, 0–1

    Returns
    -------
    float in [0, 1] — multiply by 100 for the display score
    """
    # Step 1: stretch SBERT's real operating range to fill 0–1
    cosine_scaled = float(np.clip(
        (cosine_sim - COSINE_MIN) / (COSINE_MAX - COSINE_MIN),
        0.0, 1.0,
        ))

    # Step 2: skill match as a multiplier
    #   sqrt softens the curve so partial matches aren't brutally penalised:
    #   0.25 match → 0.50 multiplier
    #   0.50 match → 0.71 multiplier
    #   0.80 match → 0.89 multiplier
    #   1.00 match → 1.00 multiplier
    skill_multiplier = float(np.sqrt(match_ratio))

    # Step 3: gap penalty — only fires when >40% of required skills are absent
    excess      = max(0.0, gap_ratio - GAP_PENALTY_FLOOR)
    gap_penalty = min(excess * GAP_PENALTY_SCALE, GAP_PENALTY_CAP)

    # Step 4: combine
    raw = cosine_scaled * skill_multiplier - gap_penalty
    return round(max(0.0, raw), 4)


# ── Resume query builder ──────────────────────────────────────────────────────
def _build_query_text(resume_text: str, resume_skills: list[str],
                      chunk_words: int = 400, max_chunks: int = 3) -> str:
    """
    Build the SBERT query string from the resume.

    The old approach truncated to 500 characters, missing everything after
    the header. This splits the resume into up to max_chunks word-chunks and
    appends the skill list to each so experience buried lower in the document
    still influences the cosine score.
    """
    skills_str = " ".join(resume_skills)
    words      = resume_text.split()
    chunks: list[str] = []

    for i in range(0, min(len(words), chunk_words * max_chunks), chunk_words):
        chunk = " ".join(words[i : i + chunk_words])
        chunks.append(f"{chunk} {skills_str}")

    return " ".join(chunks) if chunks else skills_str


# ── Fit reason ────────────────────────────────────────────────────────────────
def _fit_reason_fallback(
        title:          str,
        matched_skills: list[str],
        cosine_score:   float,
        match_ratio:    float,
        industry:       str,
) -> str:
    """Rule-based fit reason generator."""
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
        title:           str,
        matched_skills:  list[str],
        missing_skills:  list[str],
        resume_skills:   list[str],
        resume_text:     str,
        job_description: str,
        cosine_score:    float,
        match_ratio:     float,
        industry:        str = "",
) -> str:
    """
    Return a single tailored sentence explaining why this job fits this resume.

    Previously called the HF Inference LLM; now delegates directly to the
    rule-based fallback since the LLM is no longer in use. Function signature
    is unchanged so all callers (pipeline.py, scan_pipeline.py) require no edits.
    """
    return _fit_reason_fallback(
        title          = title,
        matched_skills = matched_skills,
        cosine_score   = cosine_score,
        match_ratio    = match_ratio,
        industry       = industry,
    )


# ── Skill gap ─────────────────────────────────────────────────────────────────
def identify_skill_gap(resume_skills: list[str], job_skills: list[str]) -> list[str]:
    """Return skills present in job_skills but absent from resume_skills (fuzzy)."""
    return [s for s in job_skills if not _fuzzy_skill_match(s, resume_skills)]


# ── Job encoding ──────────────────────────────────────────────────────────────
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
                f"{job.get('title', '')} {job.get('description', '')} "
                f"{' '.join(job.get('required_skills', []))} "
                f"{' '.join(job.get('required_certifications', []))}"
            )
            vec = sbert.encode(text, normalize_embeddings=True).astype(np.float32)
            job["embedding"] = vec.tolist()
        vectors.append(vec)

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix


# ── Main scoring function ─────────────────────────────────────────────────────
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

    # Build query from full resume (chunked), not just first 500 chars
    q_text = _build_query_text(resume_text, resume_skills)
    q_vec  = sbert.encode(q_text, normalize_embeddings=True).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)

    job_matrix = _encode_jobs(jobs_corpus)
    dim        = job_matrix.shape[1]
    index      = faiss.IndexFlatIP(dim)
    index.add(job_matrix)

    k                = min(top_k, index.ntotal)
    cos_scores, idxs = index.search(q_vec, k)

    results: list[dict] = []

    for cosine_score, idx in zip(cos_scores[0], idxs[0]):
        if idx < 0:
            continue

        job     = jobs_corpus[idx]
        req     = job.get("required_skills", [])

        # Fuzzy skill partition — consistent with the offline evaluator
        matched, missing = _partition_skills(req, resume_skills)

        match_ratio = len(matched) / len(req) if req else 0.0
        gap_ratio   = len(missing) / len(req) if req else 0.0

        # New hybrid score: rescaled cosine × skill multiplier − gap penalty
        hybrid = _compute_hybrid_score(
            cosine_sim  = float(cosine_score),
            match_ratio = match_ratio,
            gap_ratio   = gap_ratio,
        )

        title    = job.get("title", "")
        industry = job.get("industry", "")
        job_desc = job.get("description", "")

        results.append({
            "job_id":          job.get("id", job.get("_id", "")),
            "title":           title,
            "industry":        industry,
            "score":           hybrid,
            "cosine_score":    round(float(cosine_score), 4),
            "cosine_scaled":   round(float(np.clip(
                (cosine_score - COSINE_MIN) / (COSINE_MAX - COSINE_MIN),
                0.0, 1.0)), 4),
            "match_ratio":     round(match_ratio, 4),
            "gap_ratio":       round(gap_ratio, 4),
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
    log.info(
        f"[Stage 3] {len(results)} jobs scored; "
        f"top = {results[0]['score'] if results else 'N/A'} "
        f"(cosine_scaled={results[0].get('cosine_scaled','?')}, "
        f"match_ratio={results[0].get('match_ratio','?')})"
        if results else "[Stage 3] No results."
    )
    return results