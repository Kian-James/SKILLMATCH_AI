# backend/app/repositories/firestore/jobs.py
from app.repositories.firestore.embeddings import EmbeddingRepository

class JobRepository:
    """
    Wrapper around Firestore jobs collection and embeddings
    """
    def __init__(self):
        self.emb_repo = EmbeddingRepository()
        self.jobs_collection = "jobs"

    def get_all_jobs(self):
        """Returns all job documents"""
        docs = db.collection(self.jobs_collection).stream()
        return [doc.to_dict() for doc in docs]

    def get_job_embeddings(self):
        """Returns job embeddings for matching"""
        return self.emb_repo.get_job_embeddings()
