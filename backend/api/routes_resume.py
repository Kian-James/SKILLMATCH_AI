"""
api/routes_resume.py
Endpoints:
  POST /upload-resume  — OCR + NLP extraction only (returns metadata, no storage)
  POST /analyze        — Full analysis with job recommendations + saves snapshot
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
    user: AuthenticatedUser = Depends(get_current_user),  # 🔒 Auth required
):
    """
    Extract skills and certifications from an uploaded resume.
    File is read in memory only — nothing is stored anywhere.
    Available to all auth types including anonymous users.
    """
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PDF or image (PNG/JPG/TIFF/BMP/WEBP)."
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    faiss_index = request.app.state.faiss
    service = ResumeService(faiss_index)

    try:
        result = service.analyze(file_bytes, content_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[user={user.uid}] Resume extract error: {e}")
        raise HTTPException(status_code=500, detail="Internal analysis error.")

    logger.info(f"[user={user.uid}] Extracted {len(result['skills'])} skills (no save).")
    return UploadResponse(
        skills=result["skills"],
        certifications=result["certifications"],
        word_count=result["word_count"],
        message=f"Extracted {len(result['skills'])} skills and {len(result['certifications'])} certifications.",
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    request: Request,
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(get_current_user),  # 🔒 Auth required
):
    """
    Full analysis: extract → embed → FAISS match → save snapshot.
    UID comes from the verified Firebase token — clients never send their own UID.
    Resume file is never stored; only extracted metadata is persisted.
    """
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    faiss_index = request.app.state.faiss
    service = ResumeService(faiss_index)

    try:
        result = service.analyze(file_bytes, content_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[user={user.uid}] Full analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed.")

    # Ensure Firestore user document exists (upsert)
    get_or_create_user(
        user_id=user.uid,
        display_name=user.name or "",
        email=user.email or "",
    )

    # Persist metadata snapshot — resume file is NOT stored
    snapshot_id = save_analysis_snapshot(
        user_id=user.uid,
        skills=result["skills"],
        certifications=result["certifications"],
        top_job_matches=result["top_job_matches"],
        best_match_score=result["best_match_score"],
    )

    logger.info(f"[user={user.uid}] Snapshot {snapshot_id} saved.")
    return AnalyzeResponse(
        user_id=user.uid,
        skills=result["skills"],
        certifications=result["certifications"],
        top_job_matches=result["top_job_matches"],
        best_match_score=result["best_match_score"],
        snapshot_id=snapshot_id,
        message="Analysis complete. Snapshot saved.",
    )
