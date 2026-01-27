# backend/app/services/learning/recommender.py

from app.repositories.firestore.skills import SkillRepository
from app.repositories.firestore.certifications import CertificationRepository

class LearningRecommender:
    def __init__(self):
        self.skill_repo = SkillRepository()
        self.cert_repo = CertificationRepository()

    def get_learning_paths(self, missing_skills, missing_certs):
        suggestions = []

        for skill in missing_skills:
            resources = self.skill_repo.get_learning_resources(skill)
            suggestions.append({
                "type": "skill",
                "name": skill,
                "resources": resources
            })

        for cert in missing_certs:
            resources = self.cert_repo.get_learning_resources(cert)
            suggestions.append({
                "type": "certification",
                "name": cert,
                "resources": resources
            })

        return suggestions
