# backend/app/services/jobs/ranker.py
from typing import Dict, Any, List
from collections import Counter
from app.utils.validators import sanitize_skills

class JobRanker:
    """
    Combines similarity score and skill matching to produce a final ranked list.
    """

    def __init__(self, skill_weight: float = 0.6, similarity_weight: float = 0.4):
        """
        Weights control how much skill match vs semantic similarity impacts ranking.
        """
        self.skill_weight = skill_weight
        self.similarity_weight = similarity_weight

    def rank_jobs(
            self,
            resume_embedding,
            parsed_resume: Dict[str, Any],
            job_embeddings: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Args:
            resume_embedding: torch.Tensor, ephemeral
            parsed_resume: dict with keys: skills, experience, certifications
            job_embeddings: dict: job_id -> {embedding, skills, similarity_score, ...}

        Returns:
            List[dict]: Sorted list of jobs with final_score and metadata
        """
        resume_skills = sanitize_skills(parsed_resume.get("skills", []))
        ranked_list = []

        for job_id, job_data in job_embeddings.items():
            job_skills = sanitize_skills(job_data.get("skills", []))

            # --- Step 1: Skill match score (0-1) ---
            if job_skills:
                matched_skills = len(set(resume_skills) & set(job_skills))
                skill_score = matched_skills / len(job_skills)
            else:
                skill_score = 0.0

            # --- Step 2: Similarity score from matcher ---
            similarity_score = job_data.get("similarity_score", 0.0)

            # --- Step 3: Weighted final score ---
            final_score = (self.skill_weight * skill_score) + (self.similarity_weight * similarity_score)

            ranked_list.append({
                "job_id": job_id,
                "title": job_data.get("title"),
                "company": job_data.get("company"),
                "skills": job_skills,
                "similarity_score": similarity_score,
                "skill_score": skill_score,
                "final_score": final_score,
                "indeed_url": job_data.get("indeed_url"),
                "linkedin_url": job_data.get("linkedin_url"),
                "jobstreet_url": job_data.get("jobstreet_url"),
                "company_url": job_data.get("company_url")
            })

        # --- Step 4: Sort by final_score descending ---
        ranked_list.sort(key=lambda x: x["final_score"], reverse=True)

        return ranked_list
