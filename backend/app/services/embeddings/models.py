# backend/app/services/embeddings/models.py
from sentence_transformers import SentenceTransformer
import threading

class EmbeddingModel:
    """
    Singleton loader for embedding models.
    Supports SBERT for semantic similarity.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = SentenceTransformer(model_name)
        return cls._instance

    def encode(self, text, **kwargs):
        """
        Generate embedding for given text
        """
        return self.model.encode(text, **kwargs)
