"""
matching_config.py
All tunable matching thresholds and weights in one place.
Edit these values to calibrate the matching system without touching logic code.

HOW SCORING WORKS
-----------------
Final match score = weighted blend of:
  1. Skill match score   — how many required skills the user has
  2. FAISS cosine score  — how similar the resume embedding is to the job embedding

Each skill match is attempted in order:
  a. Exact normalized match  (e.g. "express.js" == "express.js")
  b. Keyword overlap         (e.g. "Flutter" overlaps "Flutter and Dart")
  c. Semantic embedding      (e.g. "RESTful APIs" ~ "REST API Development")

Certification bonus is added on top if user holds relevant certs.
"""

# ── Skill matching thresholds ─────────────────────────────────────────────────

# Minimum keyword overlap fraction to count as a skill match
# 0.6 = 60% of the required skill's words must appear in user skills
# Lower → more lenient (catches more), Higher → stricter (fewer false matches)
KEYWORD_OVERLAP_THRESHOLD: float = 0.6

# Minimum cosine similarity for semantic skill matching
# 0.75 = skills must be 75% semantically similar to count as matched
# Lower → more lenient, Higher → stricter
SEMANTIC_SKILL_THRESHOLD: float = 0.75

# Minimum cosine similarity for semantic CERTIFICATION matching
SEMANTIC_CERT_THRESHOLD: float = 0.78

# ── FAISS / job retrieval thresholds ─────────────────────────────────────────

# Minimum FAISS cosine similarity to include a job in top-5 results
# Jobs below this threshold are excluded even if they rank in top 5
# 0.0 = include all top 5 regardless of score
# 0.3 = only include jobs with at least 30% embedding similarity
FAISS_MIN_SCORE_THRESHOLD: float = 0.0

# Minimum embedding similarity for fuzzy job title matching
# Used when user types a target job in Skill Gap screen
JOB_TITLE_MATCH_THRESHOLD: float = 0.72

# ── Score blending weights ────────────────────────────────────────────────────
# The final displayed match % is a weighted blend of:
#   skill_score * SKILL_WEIGHT + faiss_score * FAISS_WEIGHT
# Must sum to 1.0

# How much weight to give to skill-by-skill matching (0.0 to 1.0)
# Higher = score is more influenced by which specific skills you have
SKILL_WEIGHT: float = 0.85

# How much weight to give to FAISS embedding similarity (0.0 to 1.0)
# Higher = score is more influenced by overall resume-job similarity
FAISS_WEIGHT: float = 0.15

assert abs(SKILL_WEIGHT + FAISS_WEIGHT - 1.0) < 1e-6, \
    "SKILL_WEIGHT + FAISS_WEIGHT must equal 1.0"

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

# ── Logging ───────────────────────────────────────────────────────────────────

# Set to True to log detailed per-skill match breakdown
# Useful for debugging — shows WHY each skill matched or didn't
VERBOSE_MATCH_LOGGING: bool = True