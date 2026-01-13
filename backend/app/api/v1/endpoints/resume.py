from fastapi import APIRouter, UploadFile
from app.services.ocr.google_vision import extract_text
from app.services.nlp.skill_extractor import extract_skills
from app.database.firestore import db

router = APIRouter(prefix="/resume")

@router.post("/analyze")
async def analyze(file: UploadFile, user_id: str):
    text = extract_text(await file.read())
    skills = extract_skills(text)

    db.collection("resume_skills").document(user_id).set({
        "skills": skills
    })

    return skills
