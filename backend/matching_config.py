"""
matching_config.py
All tunable matching thresholds and weights in one place.
Edit these values to calibrate the matching system without touching logic code.

HOW SCORING WORKS
-----------------
Final match score = weighted blend of:
  1. Skill match score   — how many required skills the user has
  2. FAISS cosine score  — scaled down when few skills match (see FAISS_SKILL_SCALE_THRESHOLD)

Each skill match is attempted in order:
  a. Exact normalized match  (e.g. "express.js" == "express.js")
  b. Keyword overlap         (e.g. "Flutter" overlaps "Flutter and Dart")
  c. Semantic embedding      (e.g. "RESTful APIs" ~ "REST API Development")

Certification bonus is added on top if user holds relevant certs.

FIXES (v2):
  FIX 1 — FAISS_MIN_SCORE_THRESHOLD raised from 0.0 → 0.35
           Jobs with low cosine similarity are now excluded from Top 5,
           preventing stale FAISS metadata from bypassing ZERO_SKILL_SCORE_CAP.

  FIX 2 — FAISS_SKILL_SCALE_THRESHOLD added (new parameter)
           FAISS contribution is now scaled by how many skills matched.
           When skill_score < FAISS_SKILL_SCALE_THRESHOLD, FAISS is scaled
           down proportionally so it cannot inflate the score when almost
           no skills match. At 0 skills → FAISS contributes 0%.
"""

# ── Skill matching thresholds ─────────────────────────────────────────────────

# Minimum keyword overlap fraction to count as a skill match
# 0.6 = 60% of the required skill's words must appear in user skills
# Lower → more lenient (catches more), Higher → stricter (fewer false matches)
KEYWORD_OVERLAP_THRESHOLD: float = 0.75

# Minimum cosine similarity for semantic skill matching
# 0.75 = skills must be 75% semantically similar to count as matched
# Lower → more lenient, Higher → stricter
SEMANTIC_SKILL_THRESHOLD: float = 0.75

# Minimum cosine similarity for semantic CERTIFICATION matching
SEMANTIC_CERT_THRESHOLD: float = 0.78

# ── FAISS / job retrieval thresholds ─────────────────────────────────────────

# Minimum FAISS cosine similarity to include a job in top-5 results
# FIX 1: Raised from 0.0 → 0.35
# Jobs below this threshold are excluded even if they rank in top 5.
# This prevents jobs with stale/empty skill metadata from bypassing
# ZERO_SKILL_SCORE_CAP and appearing with inflated scores (e.g. 16%
# from FAISS alone when 0 skills matched).
# 0.35 = job embedding must be at least 35% cosine-similar to resume
FAISS_MIN_SCORE_THRESHOLD: float = 0.35

# Minimum embedding similarity for fuzzy job title matching
# Used when user types a target job in Skill Gap screen
JOB_TITLE_MATCH_THRESHOLD: float = 0.72

# ── Score blending weights ────────────────────────────────────────────────────
# The final displayed match % is a weighted blend of:
#   skill_score * SKILL_WEIGHT + scaled_faiss_score * FAISS_WEIGHT
# Must sum to 1.0

# How much weight to give to skill-by-skill matching (0.0 to 1.0)
SKILL_WEIGHT: float = 0.85

# How much weight to give to FAISS embedding similarity (0.0 to 1.0)
FAISS_WEIGHT: float = 0.15

assert abs(SKILL_WEIGHT + FAISS_WEIGHT - 1.0) < 1e-6, \
    "SKILL_WEIGHT + FAISS_WEIGHT must equal 1.0"

# FIX 2: FAISS contribution scaling threshold (new parameter)
# FAISS is only allowed to contribute its full FAISS_WEIGHT when
# skill_score >= FAISS_SKILL_SCALE_THRESHOLD.
# Below this threshold, FAISS contribution is scaled down linearly:
#   faiss_multiplier = min(1.0, skill_score / FAISS_SKILL_SCALE_THRESHOLD)
#   actual_faiss_contribution = faiss_score * FAISS_WEIGHT * faiss_multiplier
#
# Effect on real scores (Kian James resume example):
#   Mobile App Developer   1/10 skills → multiplier=0.30 → FAISS adds ~5% not 18%
#   Full Stack Developer   2/15 skills → multiplier=0.40 → FAISS adds ~3% not 8%
#   Developer (Blockchain) 1/15 skills → multiplier=0.20 → FAISS adds ~2% not 12%
#   0 skills matched → multiplier=0.0 → FAISS adds 0% (consistent with ZERO cap)
#
# 0.33 = FAISS fully contributes only when ≥33% of required skills are matched
FAISS_SKILL_SCALE_THRESHOLD: float = 0.33

# ── Certification bonus ───────────────────────────────────────────────────────

# Maximum bonus added to skill score when user holds matching certifications
# 0.10 = certs can add up to 10% to the final score
CERT_BONUS_MAX: float = 0.10

# ── Score display ─────────────────────────────────────────────────────────────

# If skill_score is 0 (zero skills matched), cap the displayed score at this value
# Prevents FAISS similarity from inflating score when no skills actually match
# 0.0 = always show 0% when no skills matched (strictest)
# 0.15 = allow up to 15% from FAISS even with no skill matches
ZERO_SKILL_SCORE_CAP: float = 0.0

# Minimum number of required skills that must actually match before showing ANY
# positive score for a job. This is the second line of defense against false
# positives caused by semantic overlap in the embedding space.
# Example: a resume with Python experience has an embedding vaguely similar to a
# Blockchain Engineer job. FAISS finds it (score 0.38). The keyword matcher finds
# 1 overlapping term ("development"), bypassing ZERO_SKILL_SCORE_CAP.
# With MIN_MATCHED_SKILLS_FOR_SCORE=2, that job is suppressed to 0%.
# 1 = at least 1 skill must match (minimal filtering)
# 2 = at least 2 skills must match (recommended — eliminates most false positives)
MIN_MATCHED_SKILLS_FOR_SCORE: int = 2

# ── Logging ───────────────────────────────────────────────────────────────────

# Set to True to log detailed per-skill match breakdown
# Useful for debugging — shows WHY each skill matched or didn't
VERBOSE_MATCH_LOGGING: bool = True