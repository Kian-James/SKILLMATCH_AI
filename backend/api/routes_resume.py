"""
api/routes_resume.py
POST /upload-resume  - OCR + NLP extraction only (no storage, auth required)
POST /analyze        - Full analysis + saves snapshot with profile_vector (auth required)
"""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from loguru import logger

from api.auth_middleware import get_current_user, AuthenticatedUser
from services.resume_service import ResumeService
from services.firebase_service import save_analysis_snapshot, get_or_create_user

router = APIRouter()

ALLOWED_TYPES = {
    "application/pdf",
    "image/png", "image/jpeg", "image/jpg",
    "image/tiff", "image/bmp", "image/webp",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class UploadResponse(BaseModel):
    skills: list[str]
    certifications: list[str]
    word_count: int
    message: str


class AnalyzeResponse(BaseModel):
    user_id: str
    skills: list[str]
    certifications: list[str]
    top_job_matches: list[dict]
    best_match_score: float
    snapshot_id: str
    message: str


@router.post("/upload-resume", response_model=UploadResponse)
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(get_current_user),
):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PDF or image."
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum size is 10 MB."
        )

    await request.app.state.ensure_faiss_ready()

    service = ResumeService(request.app.state.faiss)
    try:
        result = service.analyze(file_bytes, content_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[user={user.uid}] Resume extract error: {e}")
        raise HTTPException(status_code=500, detail="Internal analysis error.")

    return UploadResponse(
        skills=result["skills"],
        certifications=result["certifications"],
        word_count=result["word_count"],
        message=(
            f"Extracted {len(result['skills'])} skills and "
            f"{len(result['certifications'])} certifications."
        ),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    request: Request,
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(get_current_user),
):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    await request.app.state.ensure_faiss_ready()

    service = ResumeService(request.app.state.faiss)
    try:
        result = service.analyze(file_bytes, content_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[user={user.uid}] Full analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed.")

    get_or_create_user(
        user_id=user.uid,
        display_name=user.name or "",
        email=user.email or "",
    )

    # Pass profile_vector so growth_service can compute career trajectory
    profile_vector: list[float] | None = result.get("query_embedding")

    snapshot_id = save_analysis_snapshot(
        user_id=user.uid,
        skills=result["skills"],
        certifications=result["certifications"],
        top_job_matches=result["top_job_matches"],
        best_match_score=result["best_match_score"],
        profile_vector=profile_vector,
    )

    return AnalyzeResponse(
        user_id=user.uid,
        skills=result["skills"],
        certifications=result["certifications"],
        top_job_matches=result["top_job_matches"],
        best_match_score=result["best_match_score"],
        snapshot_id=snapshot_id,
        message="Analysis complete. Snapshot saved.",
    )
