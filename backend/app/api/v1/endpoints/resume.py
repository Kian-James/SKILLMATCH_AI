# backend/app/api/v1/endpoints/resume.py
from fastapi import APIRouter, UploadFile, Form, HTTPException
from typing import Optional
import time

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
from app.repositories.firestore.metrics import MetricsLogger

# Models
from app.models.response import ResumeAnalyzeResponse, RankedJob, GapAnalysis, LearningSuggestion

# Utils
from app.utils.memory import ephemeral_bytes
from app.utils.validators import sanitize_skills, sanitize_learning_paths
from app.utils.timing import measure_time
from app.core.logging import log_info

router = APIRouter()

# --------------------------
# Services initialization
# --------------------------
ocr_engine = TrOCROCR()
generator = EmbeddingGenerator()
job_repo = JobRepository()
ranker = JobRanker()
gap_analyzer = GapAnalyzer()
recommender = LearningRecommender()
metrics_logger = MetricsLogger()


# --------------------------
# Resume Analysis Endpoint
# --------------------------
@router.post("/analyze", response_model=ResumeAnalyzeResponse)
@measure_time
async def analyze_resume(
        file: UploadFile,
        target_job: Optional[str] = Form(None),
        algorithm: Optional[str] = Form("SBERT")
):
    log_info(f"Resume analysis started using {algorithm}")

    # --- Step 1: Read file in-memory ---
    file_bytes = bytearray(await file.read())

    # --- Step 2: Ephemeral OCR & parsing ---
    with ephemeral_bytes(file_bytes) as resume_data:
        raw_text = ocr_engine.extract_text(resume_data)
        cleaned_text = normalize_text(raw_text)
        parsed_resume = parse_resume(cleaned_text)
        parsed_resume["skills"] = sanitize_skills(parsed_resume.get("skills", []))

    # --- Step 3: Generate temporary resume embedding ---
    resume_embedding = generator.generate_resume_embedding(parsed_resume)

    # --- Step 4: Fetch stored job embeddings ---
    job_embeddings = job_repo.get_job_embeddings()  # dict: job_id -> {embedding, skills, title, ...}
    dataset_size = len(job_embeddings)

    # --- Step 5: Runtime algorithm selection with timing ---
    start_time = time.time()
    matcher = get_matcher(algorithm)  # SBERT or KNN
    job_embeddings = matcher.match(resume_embedding, job_embeddings)
    execution_time = time.time() - start_time

    log_info(f"Matching completed in {execution_time:.4f}s using {algorithm}")

    # --- Step 6: Rank jobs ---
    ranked_jobs_list = ranker.rank_jobs(resume_embedding, parsed_resume, job_embeddings)
    ranked_jobs = [RankedJob(**j) for j in ranked_jobs_list]

    # --- Step 7: Optional career gap analysis + learning suggestions ---
    career_gap = None
    learning_suggestions = []
    if target_job:
        # Fetch target job document
        target_job_doc = next(
            (j for j in job_repo.get_all_jobs() if j.get("title").lower() == target_job.lower()),
            None
        )
        if target_job_doc:
            # Step 7a: Gap analysis
            gap = gap_analyzer.analyze(parsed_resume, target_job_doc)
            career_gap = GapAnalysis(**gap)

            # Step 7b: Learning recommendations
            learning_paths = recommender.get_learning_paths(
                gap["missing_skills"], gap["missing_certifications"]
            )
            learning_suggestions = [LearningSuggestion(**l) for l in sanitize_learning_paths(learning_paths)]

    # --- Step 8: Log metrics to Firestore ---
    metrics_logger.log_algorithm_metrics(
        algorithm=algorithm,
        execution_time=execution_time,
        dataset_size=dataset_size,
        num_ranked_jobs=len(ranked_jobs)
    )

    log_info(f"Resume analysis completed. {len(ranked_jobs)} jobs ranked")

    # --- Step 9: Return response ---
    return ResumeAnalyzeResponse(
        ranked_jobs=ranked_jobs,
        career_gap_analysis=career_gap,
        learning_paths=learning_suggestions
    )
