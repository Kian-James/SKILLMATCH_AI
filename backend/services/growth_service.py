"""
services/growth_service.py
Computes skill growth metrics by comparing historical analysis snapshots.
All data read from Firestore user_progress collection.
"""

from __future__ import annotations

from datetime import datetime
from loguru import logger

from services.firebase_service import get_user_snapshots


class GrowthService:
    """Tracks user skill growth over time via stored analysis snapshots."""

    def get_growth_data(self, user_id: str) -> dict:
        """
        Retrieve and compute growth metrics for a user.
        Returns time-series data for charting + summary statistics.
        """
        snapshots = get_user_snapshots(user_id)

        if not snapshots:
            return {
                "has_data": False,
                "message": "No analysis history yet. Upload your resume to start tracking growth.",
                "snapshots": [],
            }

        # Build time-series points
        time_series: list[dict] = []
        for snap in snapshots:
            time_series.append({
                "timestamp": snap["timestamp"],
                "skill_count": snap.get("skill_count", 0),
                "best_match_score": snap.get("best_match_score", 0.0),
                "certifications_count": len(snap.get("certifications", [])),
                "skills_sample": snap.get("skills", [])[:5],  # first 5 for display
            })

        # Compute growth deltas (first vs latest)
        first = snapshots[0]
        latest = snapshots[-1]

        skill_count_delta = latest.get("skill_count", 0) - first.get("skill_count", 0)
        score_delta = round(
            latest.get("best_match_score", 0.0) - first.get("best_match_score", 0.0), 4
        )

        # New skills added since first snapshot
        first_skills = set(first.get("skills", []))
        latest_skills = set(latest.get("skills", []))
        new_skills = sorted(latest_skills - first_skills)

        # Longest skill streak (consecutive snapshots with skill count increase)
        streak = 0
        max_streak = 0
        for i in range(1, len(snapshots)):
            prev_count = snapshots[i - 1].get("skill_count", 0)
            curr_count = snapshots[i].get("skill_count", 0)
            if curr_count > prev_count:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        return {
            "has_data": True,
            "total_snapshots": len(snapshots),
            "first_analysis": first["timestamp"],
            "latest_analysis": latest["timestamp"],
            "current_skill_count": latest.get("skill_count", 0),
            "current_best_match_score": latest.get("best_match_score", 0.0),
            "skill_count_delta": skill_count_delta,
            "score_delta": score_delta,
            "new_skills_added": new_skills,
            "new_skills_count": len(new_skills),
            "improvement_streak": max_streak,
            "time_series": time_series,
        }
