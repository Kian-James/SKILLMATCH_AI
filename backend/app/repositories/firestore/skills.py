# backend/app/repositories/firestore/skills.py
from app.repositories.firestore.client import db
from typing import List, Dict, Any

SKILLS_COLLECTION = "skills"
SKILL_EMBEDDINGS_COLLECTION = "skill_embeddings"

class SkillRepository:
    """
    Firestore access for skills and associated learning resources
    """

    def get_all_skills(self) -> List[Dict[str, Any]]:
        """
        Returns list of all skills stored in Firestore
        """
        docs = db.collection(SKILLS_COLLECTION).stream()
        return [doc.to_dict() for doc in docs]

    def get_learning_resources(self, skill_name: str) -> List[Dict[str, str]]:
        """
        Returns learning resources for a specific skill
        Example resource dict: {"name": "Python Basics", "url": "https://..."}
        """
        doc_ref = db.collection(SKILLS_COLLECTION).document(skill_name.lower())
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("learning_resources", [])
        return []
