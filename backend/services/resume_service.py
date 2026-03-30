"""
services/resume_service.py
Orchestrates the full resume analysis pipeline:
  OCR → NLP extraction → Embeddings → FAISS search → Result
Note: Resume files are NOT stored. Only extracted metadata is persisted.
"""

from __future__ import annotations

from loguru import logger

from ai_modules.ocr_extractor import extract_text
from ai_modules.nlp_processor import NLPProcessor
from ai_modules.embedding_engine import EmbeddingEngine
from ai_modules.faiss_index import FAISSIndex
from utils.skill_normalizer import SkillNormalizer


class ResumeService:
    """Full pipeline from raw resume bytes → job recommendations + metadata."""

    def __init__(self, faiss_index: FAISSIndex):
        self.faiss = faiss_index
        self.nlp = NLPProcessor()
        self.embedder = EmbeddingEngine()
        self.normalizer = SkillNormalizer()

    def analyze(
        self,
        file_bytes: bytes,
        content_type: str,
    ) -> dict:
        """
        Main analysis pipeline.
        Returns extracted skills, certifications, top job matches, and query embedding.
        """
        # Step 1: OCR — extract raw text from PDF or image
        logger.info("Step 1: OCR text extraction...")
        raw_text = extract_text(file_bytes, content_type)
        if not raw_text or len(raw_text.strip()) < 20:
            raise ValueError("Could not extract meaningful text from the uploaded file.")

        # Step 2: NLP — extract skills and certifications
        logger.info("Step 2: NLP processing...")
        extracted = self.nlp.process(raw_text)
        skills: list[str] = extracted["skills"]
        certifications: list[str] = extracted["certifications"]

        if not skills:
            logger.warning("No skills extracted — resume may be image-heavy or poorly formatted.")

        # Step 3: Embed the skill profile
        logger.info("Step 3: Generating skill profile embedding...")
        skill_embedding = self.embedder.embed_skill_list(skills)

        # Also embed combined text (title keywords + skills) for richer matching
        combined_text = " ".join(skills + certifications)
        combined_embedding = self.embedder.embed(combined_text) if combined_text.strip() else skill_embedding

        # Average the two embeddings for balanced representation
        import numpy as np
        query_embedding = ((skill_embedding + combined_embedding) / 2.0).astype("float32")
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding /= norm

        # Step 4: FAISS search — find top 5 matching jobs
        logger.info("Step 4: FAISS similarity search...")
        top_jobs = self.faiss.search(query_embedding, top_k=5)

        best_score = top_jobs[0]["match_score"] if top_jobs else 0.0

        return {
            "skills": skills,
            "certifications": certifications,
            "word_count": extracted["word_count"],
            "top_job_matches": top_jobs,
            "best_match_score": best_score,
            # Store embedding as list for serialization (used for growth tracking)
            "query_embedding": query_embedding.tolist(),
        }
