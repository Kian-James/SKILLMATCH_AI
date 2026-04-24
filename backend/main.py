"""
main.py - FastAPI application entry point

Single uvicorn worker — same as the original setup that worked.

Concurrency is handled inside the worker:
  - run_in_threadpool in routes_resume.py runs analyze() in a thread pool
    so the event loop stays free for other requests during analysis.
  - Multiple /analyze calls are handled by Python's thread pool (default 40
    threads), not by multiple processes. This is correct for CPU-bound ML
    work on a single RAM-constrained server.

Why not Gunicorn multi-worker:
  Each worker loads its own copy of torch + models (~600MB each).
  3 workers = ~1.8GB just sitting idle on a 2GB server — no headroom
  for actual request processing. Crashes on startup or under any load.
"""

import asyncio
import gc

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import API_HOST, API_PORT, API_RELOAD
from services.firebase_service import init_firebase
from ai_modules.faiss_index import FAISSIndex
from api.routes_resume import router as resume_router
from api.routes_jobs import router as jobs_router
from api.routes_skills import router as skills_router
from api.routes_growth import router as growth_router

app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume analysis with skill gap detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Firebase...")
    init_firebase()

    logger.info("Loading dynamic skill synonyms from Firestore...")
    from utils.skill_normalizer import SkillNormalizer
    SkillNormalizer.load_dynamic_synonyms()

    app.state.faiss       = FAISSIndex()
    app.state.faiss_ready = False

    logger.info("Startup complete. ML will lazy-load on first request or /warmup.")


async def ensure_faiss_ready():
    """
    Load FAISS indexes and ML models on first request.
    Single worker so no cross-process concerns — just a simple asyncio lock
    to prevent duplicate loads if two requests arrive before warmup finishes.
    """
    if app.state.faiss_ready:
        return

    if not hasattr(app.state, "_faiss_lock"):
        app.state._faiss_lock = asyncio.Lock()

    async with app.state._faiss_lock:
        if app.state.faiss_ready:
            return

        logger.info("Lazy-loading FAISS + ML models...")

        await app.state.faiss.load_or_build()
        gc.collect()
        logger.info("FAISS indexes loaded.")

        try:
            from ai_modules.nlp_processor import NLPProcessor
            n_patterns = NLPProcessor.build_matcher_from_firestore()
            logger.info(f"PhraseMatcher ready: {n_patterns} skill patterns.")
        except Exception as e:
            logger.warning(f"PhraseMatcher build failed (non-fatal): {e}")

        try:
            from services.resume_service import _get_ocr_vocab_embs
            _get_ocr_vocab_embs()
            logger.info("OCR vocab embeddings pre-warmed.")
        except Exception as e:
            logger.warning(f"OCR vocab pre-warm failed (non-fatal): {e}")

        try:
            from ai_modules.skill_implications import load_implications_from_firestore
            n_implied = load_implications_from_firestore()
            logger.info(f"Skill implication map ready: {n_implied} skills covered.")
        except Exception as e:
            logger.warning(f"Implication map load failed (non-fatal): {e}")

        # Cache ResumeService once — reused by all requests
        try:
            from services.resume_service import ResumeService
            app.state.resume_service = ResumeService(app.state.faiss)
            logger.info("ResumeService cached on app.state.")
        except Exception as e:
            logger.warning(f"ResumeService pre-cache failed (non-fatal): {e}")

        app.state.faiss_ready = True
        logger.info("All ML models ready.")


app.state.ensure_faiss_ready = ensure_faiss_ready

app.include_router(resume_router, prefix="/api/v1", tags=["Resume"])
app.include_router(jobs_router,   prefix="/api/v1", tags=["Jobs"])
app.include_router(skills_router, prefix="/api/v1", tags=["Skills"])
app.include_router(growth_router, prefix="/api/v1", tags=["Growth"])


@app.get("/health")
async def health_check():
    return {
        "status":   "ok",
        "version":  "2.0.0",
        "ml_ready": getattr(app.state, "faiss_ready", False),
    }


@app.get("/warmup")
async def warmup():
    await ensure_faiss_ready()
    return {"status": "warmed up", "ml_ready": True}


@app.get("/admin/generate-synonyms")
async def generate_synonyms():
    from utils.skill_normalizer import SkillNormalizer
    processed = SkillNormalizer.generate_and_persist_synonyms()
    return {"status": "done", "skills_processed": processed}


@app.get("/admin/build-implications")
async def build_implications():
    from ai_modules.skill_implications import build_implications_with_llm
    processed = build_implications_with_llm()
    return {"status": "done", "skills_processed": processed}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )