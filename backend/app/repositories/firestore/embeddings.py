# backend/app/repositories/firestore/embeddings.py
from app.repositories.firestore.client import db
from typing import Dict, Any
import torch

JOB_EMBEDDINGS_COLLECTION = "job_embeddings"
SKILL_EMBEDDINGS_COLLECTION = "skill_embeddings"
CERT_EMBEDDINGS_COLLECTION = "certification_embeddings"

class EmbeddingRepository:
    """
    Fetches embeddings from Firestore
    """

    @staticmethod
    def get_job_embeddings() -> Dict[str, Dict[str, Any]]:
        """
        Returns:
            dict: job_id -> {
                "embedding": torch.Tensor,
                "skills": list[str],
                "title": str,
                "company": str,
                "indeed_url": str,
                "linkedin_url": str,
                "jobstreet_url": str,
                "company_url": str
            }
        """
        result = {}
        docs = db.collection(JOB_EMBEDDINGS_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict()
            embedding_list = data.get("embedding", [])
            result[doc.id] = {
                "embedding": torch.tensor(embedding_list) if embedding_list else None,
                "skills": data.get("skills", []),
                "title": data.get("title"),
                "company": data.get("company"),
                "indeed_url": data.get("indeed_url"),
                "linkedin_url": data.get("linkedin_url"),
                "jobstreet_url": data.get("jobstreet_url"),
                "company_url": data.get("company_url")
            }
        return result

    @staticmethod
    def get_skill_embeddings() -> Dict[str, torch.Tensor]:
        """
        Returns: skill -> embedding tensor
        """
        result = {}
        docs = db.collection(SKILL_EMBEDDINGS_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict()
            embedding_list = data.get("embedding", [])
            if embedding_list:
                result[data["skill"]] = torch.tensor(embedding_list)
        return result

    @staticmethod
    def get_cert_embeddings() -> Dict[str, torch.Tensor]:
        """
        Returns: certification -> embedding tensor
        """
        result = {}
        docs = db.collection(CERT_EMBEDDINGS_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict()
            embedding_list = data.get("embedding", [])
            if embedding_list:
                result[data["cer]()]()
