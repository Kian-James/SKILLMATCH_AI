"""
api/routes_growth.py
Endpoint: GET /growth-tracking  (🔒 Auth required)
Returns historical skill growth data for the authenticated user.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from loguru import logger

from api.auth_middleware import get_current_user, AuthenticatedUser
from services.growth_service import GrowthService

router = APIRouter()
_growth_service = GrowthService()


class TimeSeriesPoint(BaseModel):
    timestamp: str
    skill_count: int
    best_match_score: float
    certifications_count: int
    skills_sample: list[str]


class GrowthResponse(BaseModel):
    has_data: bool
    message: str
    total_snapshots: int
    first_analysis: str | None
    latest_analysis: str | None
    current_skill_count: int
    current_best_match_score: float
    skill_count_delta: int
    score_delta: float
    new_skills_added: list[str]
    new_skills_count: int
    improvement_streak: int
    time_series: list[TimeSeriesPoint]
    trajectory_score: float | None = None   # cosine sim oldest→newest profile
    career_direction: str | None = None     # nearest job title to latest profile


@router.get("/growth-tracking", response_model=GrowthResponse)
async def growth_tracking(
    user: AuthenticatedUser = Depends(get_current_user),  # 🔒 Auth required
):
    """
    Returns all historical analysis snapshots for the authenticated user.
    UID is taken from the verified Firebase token — no query param needed.
    Used to render the Growth Chart screen in Flutter.
    """
    data = _growth_service.get_growth_data(user.uid)

    if not data.get("has_data"):
        return GrowthResponse(
            has_data=False,
            message=data.get("message", "No data yet."),
            total_snapshots=0,
            first_analysis=None,
            latest_analysis=None,
            current_skill_count=0,
            current_best_match_score=0.0,
            skill_count_delta=0,
            score_delta=0.0,
            new_skills_added=[],
            new_skills_count=0,
            improvement_streak=0,
            time_series=[],
        )

    return GrowthResponse(
        has_data=True,
        message="Growth data loaded successfully.",
        total_snapshots=data["total_snapshots"],
        first_analysis=data["first_analysis"],
        latest_analysis=data["latest_analysis"],
        current_skill_count=data["current_skill_count"],
        current_best_match_score=data["current_best_match_score"],
        skill_count_delta=data["skill_count_delta"],
        score_delta=data["score_delta"],
        new_skills_added=data["new_skills_added"],
        new_skills_count=data["new_skills_count"],
        improvement_streak=data["improvement_streak"],
        time_series=[TimeSeriesPoint(**p) for p in data["time_series"]],
        trajectory_score=data.get("trajectory_score"),
        career_direction=data.get("career_direction"),
    )
