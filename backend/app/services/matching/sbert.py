# backend/app/services/matching/sbert.py
from .base import BaseMatcher
from typing import Dict, Any
import torch
import torch.nn.functional as F

class SBERTMatcher(BaseMatcher):
    """
    Context-aware semantic similarity using SBERT
    """

    def match(self, resume_embedding: torch.Tensor, job_embeddings: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        for job_id, job_data in job_embeddings.items():
            job_emb = job_data.get("embedding")
            if resume_embedding is not None and job_emb is not None:
                score = F.cosine_similarity(resume_embedding, job_emb, dim=0).item()
            else:
                score = 0.0
            job_data["similarity_score"] = score
        return job_embeddings
