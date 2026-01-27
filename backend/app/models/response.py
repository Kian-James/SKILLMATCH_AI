# backend/app/models/response.py
from pydantic import BaseModel
from typing import List, Optional, Dict

class RankedJob(BaseModel):
    job_id: str
    title: str
    company: str
    score: float
    links: List[str]

class GapAnalysis(BaseModel):
    missing_skills: List[str]
    missing_certifications: List[str]

class LearningSuggestion(BaseModel):
    title: str
    type: str  # e.g., "Course", "Project"
    skill: Optional[str]
    certification: Optional[str]
    link: Optional[str]

class ResumeAnalyzeResponse(BaseModel):
    ranked_jobs: List[RankedJob]
    career_gap_analysis: Optional[GapAnalysis] = None
    learning_paths: Optional[List[LearningSuggestion]] = None
