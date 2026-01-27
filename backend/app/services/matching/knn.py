# backend/app/services/matching/knn.py
from .base import BaseMatcher
from typing import Dict, Any
import torch
import torch.nn.functional as F

class KNNMatcher(BaseMatcher):
    """
    k-Nearest Neighbors baseline matcher
    """

    def __init__(self, k: int = 5):
        self.k = k

    def match(self, resume_embedding: torch.Tensor, job_embeddings: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        # Compute cosine similarity to all jobs
        scores = []
        for job_id, job_data in job_embeddings.items():
            job_emb = job_data.get("embedding")
            if resume_embedding is not None and job_emb is not None:
                score = F.cosine_similarity(resume_embedding, job_emb, dim=0).item()
            else:
                score = 0.0
            job_data["similarity_score"] = score
            scores.append((score, job_id))

        # Sort top-k
        scores.sort(reverse=True)
        top_k_ids = [job_id for _, job_id in scores[:self.k]]

        # Zero out scores of jobs outside top-k
        for job_id in job_embeddings:
            if job_id not in top_k_ids:
                job_embeddings[job_id]["similarity_score"] = 0.0

        return job_embeddings
