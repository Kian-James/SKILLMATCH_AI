from fastapi import APIRouter
from app.services.matching.resume_matcher import match_jobs
from app.database.firestore import db

router = APIRouter(prefix="/jobs")

@router.get("/match")
def match(user_id: str):
    doc = db.collection("resume_skills").document(user_id).get()
    return match_jobs(doc.to_dict()["skills"])
