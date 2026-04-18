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
# Lowered from 0.75 → 0.60: catches more legitimate partial matches
# e.g. "REST API Development" matching user skill "REST API"
KEYWORD_OVERLAP_THRESHOLD: float = 0.60

# Minimum cosine similarity for semantic skill matching
# Lowered from 0.75 → 0.68: reduces false misses for semantically similar
# but differently-worded skills e.g. "RESTful APIs" vs "REST API Development"
# Minimum cosine similarity for semantic skill matching.
# 0.65: catches framework→language implications (React→JavaScript ~0.72,
# Flutter→Dart ~0.74, Node.js→JavaScript ~0.78) while still rejecting
# genuinely different skills (Python→Java ~0.58, React→Vue ~0.61).
SEMANTIC_SKILL_THRESHOLD: float = 0.65

# Minimum cosine similarity for semantic CERTIFICATION matching
# Lowered slightly from 0.78 → 0.75 for better cert matching
SEMANTIC_CERT_THRESHOLD: float = 0.75

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

# ── Score blending weights — DEPRECATED ──────────────────────────────────────
# FAISS is now used for candidate RETRIEVAL ONLY, not scoring.
# The displayed match % = skill_score + cert_bonus (pure NLP-based).
# These constants are kept so other modules (routes_skills, growth_service)
# that import matching_config don't break, but resume_service no longer uses them.
SKILL_WEIGHT: float = 1.0   # kept for import compatibility
FAISS_WEIGHT: float = 0.0   # FAISS contributes 0% to displayed score
FAISS_SKILL_SCALE_THRESHOLD: float = 0.33  # kept for import compatibility

# ── Certification bonus ───────────────────────────────────────────────────────

# Maximum bonus added to skill score when user holds matching certifications
# 0.10 = certs can add up to 10% to the final score
CERT_BONUS_MAX: float = 0.10

# ── Score display ─────────────────────────────────────────────────────────────

# Hard zero when no skills matched — prevents any leakage into displayed score
ZERO_SKILL_SCORE_CAP: float = 0.0

# Minimum skills that must match before showing any positive score.
# Eliminates single-keyword false positives (e.g. "development" matching
# a job the candidate has no real overlap with).
# 2 = recommended. Raise to 3 to be stricter on sparse resumes.
MIN_MATCHED_SKILLS_FOR_SCORE: int = 2

# ── Logging ───────────────────────────────────────────────────────────────────

# Set to True to log detailed per-skill match breakdown
# Useful for debugging — shows WHY each skill matched or didn't
VERBOSE_MATCH_LOGGING: bool = True