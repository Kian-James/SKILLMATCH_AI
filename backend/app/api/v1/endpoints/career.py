# backend/app/api/v1/endpoints/career.py
from fastapi import APIRouter, UploadFile, Form, HTTPException
from typing import Optional

# Services
from app.services.ocr.trocr import TrOCROCR
from app.services.nlp.normalizer import normalize_text
from app.services.nlp.parser import parse_resume
from app.services.embeddings.generator import EmbeddingGenerator
from app.services.matching.selector import get_matcher
from app.services.jobs.ranker import JobRanker
from app.services.learning.gap_analysis import GapAnalyzer
from app.services.learning.recommender import LearningRecommender
from app.repositories.firestore.jobs import JobRepository

# Models
from app.models.response import ResumeAnalyzeResponse, RankedJob, GapAnalysis, LearningSuggestion

# Utils
from app.utils.memory import ephemeral_bytes
from app.utils.validators import sanitize_skills, sanitize_learning_paths
from app.utils.timing import measure_time

router = APIRouter()
ocr_engine = TrOCROCR()
job_repo = JobRepository()
gap_analyzer = GapAnalyzer()
recommender = LearningRecommender()
ranker = JobRanker()

# --------------------------
# Career Gap Endpoint Only
# --------------------------
@router.post("/career-gap", response_model=ResumeAnalyzeResponse)
@measure_time
async def career_gap(
        file: UploadFile,
        target_job: str = Form(...),
        algorithm: Optional[str] = Form("SBERT")
):
    # --- Step 1: Read file in-memory ---
    file_bytes = bytearray(await file.read())

    # --- Step 2: Ephemeral OCR & parsing ---
    with ephemeral_bytes(file_bytes) as resume_data:
        raw_text = ocr_engine.extract_text(resume_data)
        cleaned_text = normalize_text(raw_text)
        parsed_resume = parse_resume(cleaned_text)
        parsed_resume["skills"] = sanitize_skills(parsed_resume.get("skills", []))

    # --- Step 3: Fetch target job ---
    target_job_doc = next(
        (j for j in job_repo.get_all_jobs() if j.get("title").lower() == target_job.lower()),
        None
    )
    if not target_job_doc:
        raise HTTPException(status_code=404, detail="Target job not found")

    # --- Step 4: Gap Analysis ---
    gap = gap_analyzer.analyze(parsed_resume, target_job_doc)

    # --- Step 5: Recommend learning paths ---
    learning_suggestions = recommender.get_learning_paths(
        gap["missing_skills"], gap["missing_certifications"]
    )
    learning_suggestions = sanitize_learning_paths(learning_suggestions)

    # --- Step 6: Return structured response ---
    return ResumeAnalyzeResponse(
        ranked_jobs=[],  # Not computing full ranking in this endpoint
        career_gap_analysis=GapAnalysis(**gap),
        learning_paths=[LearningSuggestion(**l) for l in learning_suggestions]
    )
