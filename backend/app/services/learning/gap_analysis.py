# backend/app/services/learning/gap_analysis.py
from typing import Dict, Any, List
from app.utils.validators import sanitize_skills

class GapAnalyzer:
    """
    Compares parsed resume with target job to identify career gaps
    """

    def analyze(self, parsed_resume: Dict[str, Any], target_job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            parsed_resume: {skills: [], certifications: [], experience: []}
            target_job: {skills: [], certifications: [], title: str, company: str}

        Returns:
            dict: missing skills, missing certifications, suggested projects, industry tools
        """
        resume_skills = sanitize_skills(parsed_resume.get("skills", []))
        resume_certs = parsed_resume.get("certifications", [])

        target_skills = target_job.get("skills", [])
        target_certs = target_job.get("certifications", [])

        missing_skills = list(set(target_skills) - set(resume_skills))
        missing_certifications = list(set(target_certs) - set(resume_certs))

        # Placeholder suggested projects and tools based on missing skills
        suggested_projects = [f"Project to practice {skill}" for skill in missing_skills]
        industry_tools = target_job.get("tools", [])  # Optional field in Firestore job doc

        return {
            "missing_skills": missing_skills,
            "missing_certifications": missing_certifications,
            "suggested_projects": suggested_projects,
            "industry_tools": industry_tools
        }
