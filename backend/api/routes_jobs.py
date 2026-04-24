"""
api/routes_jobs.py
POST /recommend-jobs - Auth required, lazy FAISS load
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from api.auth_middleware import get_current_user, AuthenticatedUser
from ai_modules.embedding_engine import EmbeddingEngine

router = APIRouter()


class RecommendRequest(BaseModel):
    skills: list[str] = Field(..., min_items=1)
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
        user: AuthenticatedUser = Depends(get_current_user),
):
    # Lazy-load ML models on first request
    await request.app.state.ensure_faiss_ready()

    embedder = EmbeddingEngine()
    skill_emb = embedder.embed_skill_list(body.skills)

    if body.certifications:
        cert_emb = embedder.embed(" ".join(body.certifications))
        combined = (skill_emb + cert_emb) / 2.0
        norm = np.linalg.norm(combined)
        query_emb = (combined / norm).astype("float32") if norm > 0 else skill_emb
    else:
        query_emb = skill_emb

    results = request.app.state.faiss.search(query_emb, top_k=body.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No jobs found. Run the dataset generator first.")

    return RecommendResponse(
        top_matches=[JobMatch(**r) for r in results],
        query_skill_count=len(body.skills),
    )