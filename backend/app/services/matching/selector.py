# backend/app/services/matching/selector.py
from .sbert import SBERTMatcher
from .knn import KNNMatcher
from .base import BaseMatcher

def get_matcher(algorithm: str = "SBERT") -> BaseMatcher:
    """
    Returns matcher instance based on algorithm string
    """
    algorithm = algorithm.upper()
    if algorithm == "SBERT":
        return SBERTMatcher()
    elif algorithm == "KNN":
        return KNNMatcher(k=5)
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'SBERT' or 'KNN'.")
