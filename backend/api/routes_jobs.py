"""
api/routes_jobs.py
Endpoint: POST /recommend-jobs
Accepts a list of skills and returns top 5 matching jobs from FAISS.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from api.auth_middleware import get_current_user, AuthenticatedUser
from ai_modules.embedding_engine import EmbeddingEngine

router = APIRouter()


class RecommendRequest(BaseModel):
    skills: list[str] = Field(..., min_items=1, description="List of user skills")
    certifications: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


class JobMatch(BaseModel):
    job_title: str
    match_score: float
    skills: list[str]
    certifications: list[str]


class RecommendResponse(BaseModel):
    top_matches: list[JobMatch]
    query_skill_count: int


@router.post("/recommend-jobs", response_model=RecommendResponse)
async def recommend_jobs(
    request: Request,
    body: RecommendRequest,
    user: AuthenticatedUser = Depends(get_current_user),  # 🔒 Auth required
):
    """
    Given a list of skills (and optionally certifications),
    return the top K matching jobs using FAISS similarity search.
    """
    faiss_index = request.app.state.faiss
    embedder = EmbeddingEngine()

    # Build query embedding from skills
    skill_emb = embedder.embed_skill_list(body.skills)

    if body.certifications:
        cert_text = " ".join(body.certifications)
        cert_emb = embedder.embed(cert_text)
        import numpy as np
        combined = (skill_emb + cert_emb) / 2.0
        norm = np.linalg.norm(combined)
        query_emb = (combined / norm).astype("float32") if norm > 0 else skill_emb
    else:
        query_emb = skill_emb

    results = faiss_index.search(query_emb, top_k=body.top_k)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No jobs found. Ensure the job dataset has been generated."
        )

    matches = [
        JobMatch(
            job_title=r["job_title"],
            match_score=r["match_score"],
            skills=r.get("skills", []),
            certifications=r.get("certifications", []),
        )
        for r in results
    ]

    return RecommendResponse(
        top_matches=matches,
        query_skill_count=len(body.skills),
    )
