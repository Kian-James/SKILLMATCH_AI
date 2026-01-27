# backend/app/services/matching/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

class BaseMatcher(ABC):
    """
    Interface for runtime matchers (SBERT / KNN)
    """

    @abstractmethod
    def match(self, resume_embedding: torch.Tensor, job_embeddings: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Compute similarity scores between resume embedding and job embeddings.
        Updates job_embeddings with 'similarity_score' field.
        """
        pass
