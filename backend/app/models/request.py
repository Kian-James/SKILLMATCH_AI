# backend/app/models/request.py
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import UploadFile, Form

class ResumeAnalyzeRequest(BaseModel):
    """
    Represents a resume analysis request.
    The file is sent as multipart/form-data in FastAPI endpoints.
    """
    # FastAPI requires UploadFile in endpoint, so model is mainly for validation
    target_job: Optional[str] = Field(
        None, description="Optional target job title for career gap analysis"
    )
    algorithm: Optional[str] = Field(
        "SBERT", description="Matching algorithm: SBERT or KNN"
    )
