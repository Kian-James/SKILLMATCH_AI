"""
api/routes_skills.py
Endpoints:
  POST /skill-gap                — Compute skill gap for a target job (🔒 Auth required)
  GET  /skill-resources/{skill}  — Learning resources for a skill  (🔒 Auth required)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from api.auth_middleware import get_current_user, AuthenticatedUser
from services.job_service import JobService
from ai_modules.llm_resources import generate_learning_resources

router = APIRouter()
_job_service = JobService()


class SkillGapRequest(BaseModel):
    target_job: str = Field(..., description="Target job title")
    user_skills: list[str] = Field(..., min_items=1)
    user_certifications: list[str] = Field(default_factory=list)


class SkillGapResponse(BaseModel):
    found: bool
    target_job: str
    completion_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    required_skills: list[str]
    required_certifications: list[str]
    match_count: int
    total_required: int
    message: str


@router.post("/skill-gap", response_model=SkillGapResponse)
async def skill_gap(
    body: SkillGapRequest,
    user: AuthenticatedUser = Depends(get_current_user),  # 🔒
):
    """Compute skill gap between user resume skills and a target job."""
    result = _job_service.compute_skill_gap(
        user_skills=body.user_skills,
        target_job_title=body.target_job,
    )

    if not result.get("found"):
        raise HTTPException(
            status_code=404,
            detail=result.get("error", f"Job '{body.target_job}' not found.")
        )

    score = result["completion_score"]
    if score >= 90:
        msg = "Excellent! You're highly qualified for this role."
    elif score >= 70:
        msg = "Great match! A few skills to fill in."
    elif score >= 50:
        msg = "Good foundation. Focus on the missing skills."
    else:
        msg = "Room to grow — start with the core missing skills."

    return SkillGapResponse(
        found=True,
        target_job=result["target_job"],
        completion_score=score,
        matched_skills=result["matched_skills"],
        missing_skills=result["missing_skills"],
        required_skills=result["required_skills"],
        required_certifications=result["required_certifications"],
        match_count=result["match_count"],
        total_required=result["total_required"],
        message=msg,
    )


class ResourceItem(BaseModel):
    title: str
    type: str
    platform: str
    duration: str
    description: str


class ResourcesResponse(BaseModel):
    skill: str
    resources: list[ResourceItem]


@router.get("/skill-resources/{skill}", response_model=ResourcesResponse)
async def skill_resources(
    skill: str,
    user: AuthenticatedUser = Depends(get_current_user),  # 🔒
):
    """
    Generate learning resources for a specific missing skill via Zephyr 7B.
    Called when user taps a missing skill chip in the Skill Gap screen.
    """
    if not skill.strip():
        raise HTTPException(status_code=400, detail="Skill name cannot be empty.")

    resources = generate_learning_resources(skill.strip())
    return ResourcesResponse(
        skill=skill,
        resources=[ResourceItem(**r) for r in resources],
    )
