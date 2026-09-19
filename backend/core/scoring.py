"""
core/scoring.py
===============
Stage 3 (FAISS side) — SBERT embedding + hybrid scoring.

Hybrid score formula
---------------------
Cosine similarity and skill match are combined as a weighted average
(50/50 by default), not multiplied together. A multiplicative formula lets
either signal single-handedly crush the whole score to near-zero even when
the other signal is strong — e.g. a 100% skill match getting dragged below
threshold by a merely-good cosine of 0.55, or a strong cosine getting
zeroed out by one bad skill-extraction pass. A weighted average lets each
signal partially carry the score on its own. A gap penalty is still
subtracted afterward so a near-total skill mismatch can't be rescued by
semantic similarity alone.

Formula:
  1. Rescale cosine similarity from SBERT's real range (0.30–0.90) → 0–1
  2. score = cosine_weight * cosine_scaled + skill_weight * match_ratio
     (cosine_weight/skill_weight default to WEIGHT_COSINE/WEIGHT_SKILL,
     but are re-blended to 1.0/0.0 for a given job when that job has no
     required_skills to score against — see "Skill/cosine coupling" below)
  3. Apply a gap penalty that only activates when >40% of required skills
     are missing, capping at a 50-point drag
  4. Multiply by 100 for a 0–100 output with a meaningful 60-point threshold

The result is already on a 0-100 scale, self-consistent with the
recommended min_match_score threshold of 60 — no rescaling needed by callers.

Typical score ranges:
  Perfect        (cosine 0.88, skills 1.00, gap 0.00)  →  ~93
  Strong match   (cosine 0.82, skills 0.90, gap 0.10)  →  ~88
  Good match     (cosine 0.72, skills 0.75, gap 0.25)  →  ~72
  Full skills,
   modest cosine (cosine 0.55, skills 1.00, gap 0.00)  →  ~71
  Semantic only  (cosine 0.80, skills 0.30, gap 0.70)  →  ~12 (gap penalty
                                                                does the work)
  Weak all round (cosine 0.45, skills 0.20, gap 0.80)  →   ~6

Skill/cosine coupling (IMPORTANT — read before touching the `req` branch below)
--------------------------------------------------------------------------------
This pipeline is meant to run as a genuine sequence: OCR text -> NLP-extracted
resume_skills -> FAISS/hybrid score. That only holds if the skill term in the
hybrid score actually reflects what Stage 2 (NLP) found.

A job with an EMPTY `required_skills` list (a corpus data-quality gap, not a
statement about the candidate) previously caused this function to hard-code
match_ratio = 1.0 for every resume scored against it — including resumes
where Stage 2 extracted zero skills. That made a 0-skill extraction score
identical to a 10/10 perfect match for any job with missing corpus data,
silently decoupling Stage 3 from Stage 2's output for those jobs.

Fix: when a job has no required_skills, the skill term is dropped from the
blend entirely (skill_weight = 0.0, cosine_weight = 1.0) rather than being
faked as a perfect match. The result carries a `skill_scored: False` flag so
downstream consumers (evaluator, API responses, UI) can tell "skill match
genuinely computed" apart from "skill component was skipped, score is
cosine-only for this job". Never treat skill_scored=False the same as a
verified skill match.

fit_reason generation: generate_fit_reason calls the HF Inference API
  (meta-llama/Llama-3.1-8B-Instruct via the HuggingFace router) to produce
  a personalised 2-3 sentence description grounded in the candidate's actual
  extracted skills. If the API call fails for any reason (timeout, quota,
  missing token) it falls back to the rule-based _fit_reason_fallback so the
  pipeline never errors out.
"""

from __future__ import annotations

import logging
import requests

import faiss
import numpy as np
from tqdm import tqdm

from config import TOP_K, HF_TOKEN, HF_API_URL, HF_MODEL
from core.utils import normalize
from core.validation import semantic_verify
from core.ocr_correction import get_sbert

log = logging.getLogger(__name__)


# ── LLM fit-reason circuit breaker ─────────────────────────────────────────────
# Once _call_llama_fit_reason fails (timeout, quota, bad token, bad response,
# etc.), don't keep re-trying it for every remaining job in the same batch —
# that would mean eating a fresh ~15s timeout per job for a corpus of
# hundreds of jobs. Trip the breaker on first failure within a process and
# go straight to the rule-based fallback for everything after that. Call
# reset_llm_fit_reason_breaker() (e.g. at the start of a new scoring run,
# or after fixing the underlying issue) to allow the LLM path to be tried again.
_LLM_FIT_REASON_DISABLED = False


def reset_llm_fit_reason_breaker() -> None:
    """Re-enable the LLM fit-reason path after a prior failure tripped the breaker."""
    global _LLM_FIT_REASON_DISABLED
    _LLM_FIT_REASON_DISABLED = False


# ── FAISS index cache ──────────────────────────────────────────────────────────
# Keyed by (id(jobs_corpus), len(jobs_corpus)) so repeat calls against the same
# loaded corpus (e.g. batch evaluation scoring many resumes in one process)
# skip re-encoding jobs and rebuilding the index every call. Invalidates
# automatically if a *different* corpus list is passed, or if the same list
# object's length changes (a cheap signal something was added/removed).
#
# NOT safe across processes or for corpora that mutate in place without a
# length change (e.g. replacing an element) — call clear_index_cache() if
# jobs_corpus content changes in place.
_INDEX_CACHE: dict[tuple[int, int], tuple["faiss.IndexFlatIP", np.ndarray]] = {}


def clear_index_cache() -> None:
    """Drop all cached FAISS indexes. Call this if a cached jobs_corpus's
    content changes in place without its identity or length changing."""
    _INDEX_CACHE.clear()


def _get_or_build_index(jobs_corpus: list[dict]) -> tuple["faiss.IndexFlatIP", np.ndarray]:
    """
    Return a cached (index, job_matrix) pair for this jobs_corpus, building
    and caching it on first use. Subsequent calls with the same corpus list
    (same object, same length) reuse the cached FAISS index instead of
    re-encoding every job and rebuilding the index from scratch.
    """
    cache_key = (id(jobs_corpus), len(jobs_corpus))
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    job_matrix = _encode_jobs(jobs_corpus)
    dim        = job_matrix.shape[1]
    index      = faiss.IndexFlatIP(dim)
    index.add(job_matrix)

    _INDEX_CACHE[cache_key] = (index, job_matrix)
    return index, job_matrix


# ── Scoring constants ─────────────────────────────────────────────────────────
COSINE_MIN        = 0.30   # SBERT inner-product floor for real documents
COSINE_MAX        = 0.90   # SBERT inner-product ceiling for real documents
WEIGHT_COSINE     = 0.50   # weight of rescaled cosine similarity in the blend
WEIGHT_SKILL      = 0.50   # weight of skill match_ratio in the blend (must sum to 1.0 with above)
GAP_PENALTY_FLOOR = 0.40   # gap penalty only activates above this missing-skill ratio
GAP_PENALTY_SCALE = 1.50   # how steeply the penalty grows past the floor
GAP_PENALTY_CAP   = 0.50   # maximum penalty (50 points on the 0-1 scale), prevents negative scores
ITEM_SEMANTIC_THRESHOLD = 0.80  # SBERT cosine floor for a semantic skill/cert match,
# matches the convention used in validation.py / extractor_base.py


# ── Item matching (skills and certs share this — semantic, not string-fuzzy) ──
def _encode_norm(texts: list[str]) -> np.ndarray:
    """Batch-encode normalized embeddings via the shared SBERT model."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    sbert = get_sbert()
    return sbert.encode(texts, normalize_embeddings=True).astype(np.float32)


def _build_index(vecs: np.ndarray) -> "faiss.IndexFlatIP | None":
    """
    Build a FAISS IndexFlatIP over already-normalized candidate embeddings.

    Returns None for an empty candidate set (nothing to search — same
    "no match possible" case _semantic_item_match previously handled via
    `candidate_vecs.shape[0] == 0`). vecs are re-normalized defensively
    (same pattern as _get_or_build_index / build_ocr_corrector) so this is
    correct even if a caller passes in embeddings that weren't normalized
    at encode time.
    """
    if vecs.shape[0] == 0:
        return None
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def _semantic_item_match(
        target_item:     str,
        candidate_norms:  list[str],
        candidate_index:  "faiss.IndexFlatIP | None",
        threshold:        float = ITEM_SEMANTIC_THRESHOLD,
) -> bool:
    """
    Return True if target_item matches any candidate semantically.

    Exact-equality and substring checks are kept as cheap pre-filters (they
    are plain string comparisons, not fuzzy scoring) before falling back to
    a FAISS top-1 nearest-neighbour search over the candidate embeddings for
    anything not already obviously the same.
    """
    tn = normalize(target_item)
    if tn in candidate_norms:
        return True
    if any(tn in cn or cn in tn for cn in candidate_norms):
        return True
    if candidate_index is None or candidate_index.ntotal == 0:
        return False
    target_vec = _encode_norm([tn])
    faiss.normalize_L2(target_vec)
    D, I = candidate_index.search(target_vec, 1)
    return bool(I[0][0] >= 0 and float(D[0][0]) >= threshold)


## NOTE: the old _partition_items()/_partition_skills() (single-shot FAISS
## top-1 lookup, no phrase-growth, no consumption tracking) has been removed.
## Skill/cert partitioning for scoring now goes through semantic_verify()
## (core/validation.py) — see calculate_cosine_similarity() below — so there
## is one semantic-matching implementation feeding both match_ratio/
## fit_reason and nlp_validation, instead of two that could disagree.
## _semantic_item_match / _encode_norm / _build_index are kept as-is: they're
## still imported directly by evaluate_pipeline.py's offline evaluator.


# ── Hybrid score ──────────────────────────────────────────────────────────────
def _compute_hybrid_score(
        cosine_sim:    float,
        match_ratio:   float,
        gap_ratio:     float,
        cosine_weight: float = WEIGHT_COSINE,
        skill_weight:  float = WEIGHT_SKILL,
) -> float:
    """
    Compute a 0–100 hybrid score from the three stage inputs.

    Parameters
    ----------
    cosine_sim    : raw SBERT inner product (normalized vectors), ~0.30–0.90
    match_ratio   : fuzzy-matched skills / total required skills, 0–1
    gap_ratio     : missing skills / total required skills, 0–1
    cosine_weight : weight of the rescaled cosine term in the blend
    skill_weight  : weight of the skill match_ratio term in the blend

    cosine_weight/skill_weight default to the module-level WEIGHT_COSINE /
    WEIGHT_SKILL (the normal 50/50 blend) but callers can re-blend per job —
    e.g. calculate_cosine_similarity passes (1.0, 0.0) for a job with no
    required_skills, so the skill term isn't faked as a perfect match when
    it was never actually evaluated. Callers are responsible for keeping
    cosine_weight + skill_weight == 1.0.

    Returns
    -------
    float in [0, 100] — already on display/threshold scale, no further
    rescaling needed by callers.
    """
    # Step 1: stretch SBERT's real operating range to fill 0–1
    cosine_scaled = float(np.clip(
        (cosine_sim - COSINE_MIN) / (COSINE_MAX - COSINE_MIN),
        0.0, 1.0,
        ))

    # Step 2: weighted average of cosine and skill match, not a multiplier —
    # see module docstring for why. Each signal can partially carry the
    # score on its own.
    blended = cosine_weight * cosine_scaled + skill_weight * match_ratio

    # Step 3: gap penalty — only fires when >40% of required skills are
    # absent. This is what actually punishes severe skill mismatches now
    # that match_ratio is additive rather than multiplicative. When
    # skill_weight is 0 (no required_skills for this job), gap_ratio is
    # always 0 too, so this naturally no-ops.
    excess      = max(0.0, gap_ratio - GAP_PENALTY_FLOOR)
    gap_penalty = min(excess * GAP_PENALTY_SCALE, GAP_PENALTY_CAP)

    # Step 4: combine and scale to 0-100
    raw = blended - gap_penalty
    return round(max(0.0, raw) * 100, 2)


# ── Resume query builder ──────────────────────────────────────────────────────
def _build_query_text(resume_text: str, resume_skills: list[str],
                      chunk_words: int = 400, max_chunks: int = 3) -> str:
    """
    Build the SBERT query string from the resume.

    Splits the resume into up to max_chunks word-chunks and appends the
    skill list to each, so experience buried lower in the document (not
    just the header) still influences the cosine score.
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
        missing_skills: list[str],
        cosine_score:   float,
        match_ratio:    float,
        industry:       str,
) -> str:
    """Rule-based fit reason — used when the LLM call fails or HF_TOKEN is unset."""
    industry_suffix = f" in the {industry} industry" if industry else ""
    if matched_skills:
        showcase   = matched_skills[:3]
        skills_str = ", ".join(showcase)
        tail       = f" and {len(matched_skills) - 3} more" if len(matched_skills) > 3 else ""
        gap_str    = (", ".join(missing_skills[:2]) + " among others") if missing_skills else ""
        if match_ratio >= 0.75:
            base = (
                f"Strong fit{industry_suffix} — your resume covers {skills_str}{tail}, "
                f"matching {round(match_ratio * 100)}% of what {title} requires."
            )
        elif match_ratio >= 0.45:
            base = (
                f"Good alignment{industry_suffix} — skills like {skills_str}{tail} "
                f"overlap well with the {title} role."
            )
        else:
            base = (
                f"Partial match{industry_suffix} — your experience with {skills_str}{tail} "
                f"is relevant to {title}, though more skills would help."
            )
        if gap_str:
            base += f" Consider developing {gap_str} to strengthen your candidacy."
        return base
    if cosine_score >= 0.75:
        return f"Semantically strong match{industry_suffix} for {title}."
    elif cosine_score >= 0.5:
        return f"Moderate semantic fit{industry_suffix} with the {title} role."
    return f"{title} is an aspirational direction; targeted upskilling would improve your fit."


def _call_llama_fit_reason(
        title:           str,
        matched_skills:  list[str],
        missing_skills:  list[str],
        resume_skills:   list[str],
        job_description: str,
        match_ratio:     float,
        industry:        str,
) -> str:
    """
    Call meta-llama/Llama-3.1-8B-Instruct via the HF Inference router to
    generate a personalised fit reason grounded in the candidate's skills.

    Raises on any network/API error so the caller can fall back gracefully.
    """
    industry_clause = f" in the {industry} industry" if industry else ""
    matched_str     = ", ".join(matched_skills) if matched_skills else "none"
    missing_str     = ", ".join(missing_skills[:5]) if missing_skills else "none"
    all_skills_str  = ", ".join(resume_skills)

    prompt = (
        f"You are a career advisor writing a short job-fit summary for a resume screening app.\n\n"
        f"Candidate skills: {all_skills_str}\n"
        f"Target role: {title}{industry_clause}\n"
        f"Skills matched: {matched_str}\n"
        f"Skills missing: {missing_str}\n"
        f"Match ratio: {round(match_ratio * 100)}%\n\n"
        f"Write exactly 2-3 sentences explaining why this role fits (or doesn't fit) "
        f"this candidate based specifically on their skills. "
        f"Be direct, honest, and constructive. "
        f"Do not use bullet points or headers. Plain prose only. "
        f"Do not start with 'I' or 'The candidate'."
    )

    response = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type":  "application/json",
        },
        json={
            "model": HF_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 180,
            "temperature": 0.5,
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


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
    Return a personalised 2-3 sentence description explaining why this job
    fits this resume, grounded in the candidate's extracted skills.

    Primary path  : meta-llama/Llama-3.1-8B-Instruct via HF Inference router.
    Fallback path : rule-based _fit_reason_fallback (no external call).

    Falls back automatically if HF_TOKEN is missing or the API errors. Only
    ever attempts the LLM call once per process: if it fails, a circuit
    breaker trips so every subsequent call (this job, and every later job in
    the same batch) skips the LLM and goes straight to the fallback instead
    of re-trying (and re-timing-out) on each one. See reset_llm_fit_reason_breaker().
    """
    global _LLM_FIT_REASON_DISABLED

    if HF_TOKEN and not _LLM_FIT_REASON_DISABLED:
        try:
            return _call_llama_fit_reason(
                title           = title,
                matched_skills  = matched_skills,
                missing_skills  = missing_skills,
                resume_skills   = resume_skills,
                job_description = job_description,
                match_ratio     = match_ratio,
                industry        = industry,
            )
        except Exception as exc:
            _LLM_FIT_REASON_DISABLED = True
            log.warning(
                f"[FitReason] LLM call failed ({exc}); switching to rule-based "
                f"fallback for the rest of this run."
            )

    return _fit_reason_fallback(
        title          = title,
        matched_skills = matched_skills,
        missing_skills = missing_skills,
        cosine_score   = cosine_score,
        match_ratio    = match_ratio,
        industry        = industry,
    )


# ── Skill gap ─────────────────────────────────────────────────────────────────
def identify_skill_gap(resume_skills: list[str], job_skills: list[str]) -> list[str]:
    """Return skills present in job_skills but absent from resume_skills (semantic)."""
    resume_norms = [normalize(s) for s in resume_skills]
    resume_index = _build_index(_encode_norm(resume_norms))
    return [s for s in job_skills if not _semantic_item_match(s, resume_norms, resume_index)]


# ── Job encoding ──────────────────────────────────────────────────────────────
def _encode_jobs(jobs_corpus: list[dict]) -> np.ndarray:
    """
    Encode job documents with SBERT; reuse stored embeddings when available.

    SIDE EFFECT: writes job["embedding"] back onto each dict in jobs_corpus
    in place, so repeat calls against the same corpus list skip re-encoding.
    This is a deliberate caching optimization for batch evaluation (e.g. the
    offline evaluator re-scoring many resumes against one loaded corpus),
    but it means jobs_corpus is NOT safely shared across concurrent callers
    — if two requests score against the same cached corpus list at once,
    this mutation is a race condition. Callers that share a corpus across
    threads/async tasks should either lock around scoring calls or key
    embeddings by job id in a separate cache instead of mutating the dicts.
    """
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
        resume_certs:  list[str] | None = None,
) -> list[dict]:
    """
    Stage 3 — hybrid scoring + per-job NLP validation.

    Returns top_k job dicts sorted by score descending, each including
    an 'nlp_validation' key with per-job accuracy breakdown for skills, and
    (when resume_certs is provided) a parallel 'nlp_validation_certs' key
    plus matched_certs/missing_certs/required_certifications.

    resume_certs is optional and defaults to an empty list for backward
    compatibility with existing callers that don't pass it — in that case
    cert fields are still present in each result but empty/N-A rather than
    omitted, so downstream code can rely on the keys always existing.

    NOTE: certs are matched and reported but deliberately NOT folded into
    the hybrid "score" — the score formula (WEIGHT_COSINE/WEIGHT_SKILL/gap
    penalty) is unchanged. If you want certs to influence ranking, that's a
    separate change to _compute_hybrid_score's weighting.

    SKILL/COSINE COUPLING: each result carries a `skill_scored` flag.
    skill_scored=True means match_ratio/gap_ratio were computed against a
    real required_skills list for that job and genuinely contributed to the
    score. skill_scored=False means that job had no required_skills data
    (a corpus gap), the skill term was excluded from the blend, and `score`
    is cosine-only for that job — match_ratio/gap_ratio are reported as None
    in that case rather than a faked 1.0/0.0, so downstream code (including
    evaluate_pipeline.py) can distinguish a verified skill match from an
    unscored one instead of silently treating them the same. See the module
    docstring for why this matters.

    PERFORMANCE NOTE: the FAISS index for jobs_corpus is built once and
    cached (see _get_or_build_index) — repeat calls with the same corpus
    list reuse the cached index instead of re-encoding every job and
    rebuilding the index from scratch. This matters for batch evaluation
    (many resumes scored against one loaded corpus in a single process).
    Call clear_index_cache() if jobs_corpus content changes in place.
    """
    resume_certs = resume_certs or []
    sbert = get_sbert()

    # Build query from full resume (chunked), not just first 500 chars
    q_text = _build_query_text(resume_text, resume_skills)
    q_vec  = sbert.encode(q_text, normalize_embeddings=True).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)

    index, job_matrix = _get_or_build_index(jobs_corpus)

    k                = min(top_k, index.ntotal)
    cos_scores, idxs = index.search(q_vec, k)

    results: list[dict] = []

    for cosine_score, idx in zip(cos_scores[0], idxs[0]):
        if idx < 0:
            continue

        job     = jobs_corpus[idx]
        req     = job.get("required_skills", [])

        # Skill partition now runs through the same semantic_verify()
        # sliding-window matcher used to report nlp_validation, instead of
        # a separate _partition_items() single-shot FAISS lookup. Previously
        # these were two independent implementations of "is this skill
        # matched" (different consumption/growth behaviour), so match_ratio
        # / fit_reason (derived from matched/missing here) and
        # nlp_validation.NLP_ACCURACY could — and did — disagree on the same
        # resume/job pair. Computing it once and reusing the result for both
        # also avoids encoding the same discovered-skill list twice.
        skill_verify = semantic_verify(
            discovered_list   = resume_skills,
            ground_truth_list = req,
        )
        matched = skill_verify["matched_skills"]
        missing = skill_verify["unmatched_skills"]

        # Same treatment for certifications.
        req_certs = job.get("required_certifications", [])
        cert_verify = semantic_verify(
            discovered_list   = resume_certs,
            ground_truth_list = req_certs,
        )
        matched_certs = cert_verify["matched_skills"]
        missing_certs = cert_verify["unmatched_skills"]

        if req:
            # Normal case: this job has real required_skills data, so the
            # skill term genuinely reflects Stage 2's extraction quality.
            match_ratio   = len(matched) / len(req)
            gap_ratio     = len(missing) / len(req)
            cosine_weight = WEIGHT_COSINE
            skill_weight  = WEIGHT_SKILL
            skill_scored  = True
        else:
            # Job has no listed required_skills — a corpus data-quality gap,
            # not a candidate mismatch, and NOT evidence the candidate's
            # skills are a perfect fit.
            #
            # Previously this hard-coded match_ratio = 1.0 here, which meant
            # ANY resume scored against this job — including one where
            # Stage 2 (NLP) extracted zero skills — got a perfect skill
            # score. That silently decoupled Stage 3 from Stage 2's output
            # for every job with missing corpus data: a 0-skill extraction
            # scored identically to a 10/10 match. Fixed by dropping the
            # skill term from the blend entirely (skill_weight = 0.0) and
            # letting cosine alone carry the full weight for this job,
            # rather than fabricating a match that was never evaluated.
            match_ratio   = 0.0
            gap_ratio     = 0.0
            cosine_weight = 1.0
            skill_weight  = 0.0
            skill_scored  = False

        # Hybrid score: weighted average of rescaled cosine and skill
        # match_ratio, minus gap penalty — returned already on a 0-100 scale.
        # cosine_weight/skill_weight fall back to the normal 50/50 split
        # unless this job had no required_skills (see above).
        hybrid = _compute_hybrid_score(
            cosine_sim    = float(cosine_score),
            match_ratio   = match_ratio,
            gap_ratio     = gap_ratio,
            cosine_weight = cosine_weight,
            skill_weight  = skill_weight,
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
            # None (rather than a faked number) when skill_scored is False —
            # this job's score is cosine-only and there is no real skill
            # match/gap ratio to report.
            "match_ratio":     round(match_ratio, 4) if skill_scored else None,
            "gap_ratio":       round(gap_ratio, 4) if skill_scored else None,
            "skill_scored":    skill_scored,
            "gap_count":       len(missing),
            "required_skills": req,
            "matched_skills":  matched,
            "missing_skills":  missing,
            "required_certifications": req_certs,
            "matched_certs":   matched_certs,
            "missing_certs":   missing_certs,
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
            # Reuse the same semantic_verify() calls made above for
            # matched/missing — this is now the single source of truth for
            # both the score/fit_reason path and the reported accuracy, so
            # the two can no longer drift apart on the same resume/job pair.
            "nlp_validation":       skill_verify,
            "nlp_validation_certs": cert_verify,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    unscored = sum(1 for r in results if not r["skill_scored"])
    log.info(
        f"[Stage 3] {len(results)} jobs scored "
        f"({unscored} cosine-only due to missing corpus required_skills); "
        f"top = {results[0]['score'] if results else 'N/A'} "
        f"(cosine_scaled={results[0].get('cosine_scaled','?')}, "
        f"match_ratio={results[0].get('match_ratio','?')}, "
        f"skill_scored={results[0].get('skill_scored','?')})"
        if results else "[Stage 3] No results."
    )
    return results