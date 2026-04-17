"""
services/growth_service.py
Skill growth tracking with embedding-based career trajectory detection.

NEW: Each snapshot stores a profile_vector (mean of skill embeddings).
     get_growth_data() now computes:
       - trajectory_score: cosine similarity between oldest and newest profile vectors.
         A score near 1.0 means the candidate is deepening in one direction.
         A lower score means they are broadening or pivoting.
       - career_direction: the job cluster the latest profile vector is closest to
         (uses the FAISS job index to find the nearest job title).
     These are additive fields — all existing fields are preserved.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from loguru import logger

from services.firebase_service import get_user_snapshots, save_analysis_snapshot


class GrowthService:

    def get_growth_data(self, user_id: str) -> dict:
        snapshots = get_user_snapshots(user_id)

        if not snapshots:
            return {
                "has_data": False,
                "message": "No analysis history yet. Upload your resume to start tracking growth.",
                "snapshots": [],
            }

        # ── Time series ───────────────────────────────────────────────────────
        time_series: list[dict] = []
        for snap in snapshots:
            time_series.append({
                "timestamp":            snap["timestamp"],
                "skill_count":          snap.get("skill_count", 0),
                "best_match_score":     snap.get("best_match_score", 0.0),
                "certifications_count": len(snap.get("certifications", [])),
                "skills_sample":        snap.get("skills", [])[:5],
            })

        # ── Deltas ────────────────────────────────────────────────────────────
        first  = snapshots[0]
        latest = snapshots[-1]

        skill_count_delta = (
            latest.get("skill_count", 0) - first.get("skill_count", 0)
        )
        score_delta = round(
            latest.get("best_match_score", 0.0) - first.get("best_match_score", 0.0), 4
        )
        first_skills  = set(first.get("skills", []))
        latest_skills = set(latest.get("skills", []))
        new_skills    = sorted(latest_skills - first_skills)

        # ── Improvement streak ────────────────────────────────────────────────
        streak = max_streak = 0
        for i in range(1, len(snapshots)):
            if snapshots[i].get("skill_count", 0) > snapshots[i-1].get("skill_count", 0):
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        # ── Trajectory (embedding-based) ──────────────────────────────────────
        trajectory_score   = None
        career_direction   = None

        if len(snapshots) >= 2:
            try:
                trajectory_score, career_direction = self._compute_trajectory(
                    first, latest
                )
            except Exception as e:
                logger.warning(f"Trajectory computation failed (non-fatal): {e}")

        result = {
            "has_data":                 True,
            "total_snapshots":          len(snapshots),
            "first_analysis":           first["timestamp"],
            "latest_analysis":          latest["timestamp"],
            "current_skill_count":      latest.get("skill_count", 0),
            "current_best_match_score": latest.get("best_match_score", 0.0),
            "skill_count_delta":        skill_count_delta,
            "score_delta":              score_delta,
            "new_skills_added":         new_skills,
            "new_skills_count":         len(new_skills),
            "improvement_streak":       max_streak,
            "time_series":              time_series,
        }

        if trajectory_score is not None:
            result["trajectory_score"]  = round(float(trajectory_score), 3)
            result["career_direction"]  = career_direction

        return result

    # ── Trajectory helpers ────────────────────────────────────────────────────

    @staticmethod
    def _compute_trajectory(
        first_snap: dict, latest_snap: dict
    ) -> tuple[float, str | None]:
        """
        Compare profile vectors from first and latest snapshots.

        trajectory_score: cosine similarity between the two profile vectors.
          ~1.0 = deepening in a consistent direction
          ~0.5 = broadening / pivoting significantly

        career_direction: name of the nearest job from the FAISS index to the
          latest profile vector (requires FAISS to be loaded).
        """
        # Profile vectors are stored as lists in Firestore
        vec_first  = first_snap.get("profile_vector")
        vec_latest = latest_snap.get("profile_vector")

        trajectory_score: float | None = None
        if vec_first and vec_latest:
            a = np.array(vec_first,  dtype=np.float32)
            b = np.array(vec_latest, dtype=np.float32)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                trajectory_score = float(np.dot(a / norm_a, b / norm_b))

        # Nearest job direction from FAISS (best-effort)
        career_direction: str | None = None
        if vec_latest:
            try:
                from ai_modules.faiss_index import FAISSIndex
                # Attempt to use the app-state FAISS instance if available
                # Falls back gracefully if not accessible outside request context
                from fastapi import Request
                import inspect, gc
                faiss_instance = None
                for obj in gc.get_referrers(FAISSIndex):
                    if isinstance(obj, FAISSIndex) and obj.index is not None:
                        faiss_instance = obj
                        break

                if faiss_instance and faiss_instance.metadata:
                    vec = np.array(vec_latest, dtype=np.float32).reshape(1, -1)
                    scores, indices = faiss_instance.index.search(vec, 1)
                    if indices[0][0] >= 0:
                        career_direction = faiss_instance.metadata[indices[0][0]]["job_title"]
            except Exception:
                pass  # Non-fatal — career_direction is optional enrichment

        return trajectory_score, career_direction
