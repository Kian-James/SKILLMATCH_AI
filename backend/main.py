"""
main.py - FastAPI application entry point

Render free tier: 512 MB RAM hard limit.
Render standard (2 GB): run with Gunicorn 3 workers + --preload (see Dockerfile).

MEMORY STRATEGY:
  - Boot: only Firebase + synonym map (no ML, no FAISS, no torch)
  - Startup kicks off warmup as a background task — ML models load immediately
    after startup without blocking the first real request.
  - All heavy singletons (EmbeddingEngine, SkillClusterer, SkillNormalizer)
    are module-level in resume_service — shared across requests, allocated once.
  - ResumeService instance cached on app.state after warmup — not re-created
    per request.
  - job_metadata.json no longer stores numpy arrays — stays ~2 MB in RAM.
  - ctx_embeddings stored as float16 .npy — ~5 MB vs ~20 MB float32.

CHANGES:
  - startup_event now fires ensure_faiss_ready() as a background asyncio task
    so ML models begin loading immediately on deploy (not on first request).
  - ResumeService is instantiated once after warmup and cached on app.state.
    Routes reuse it via _get_service() instead of constructing per request.
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
    # Step 1: Firebase only — no ML loaded yet
    logger.info("Initializing Firebase...")
    init_firebase()

    # Step 2: Load LLM-generated synonyms from Firestore (pure dict ops, no ML)
    logger.info("Loading dynamic skill synonyms from Firestore...")
    from utils.skill_normalizer import SkillNormalizer
    SkillNormalizer.load_dynamic_synonyms()

    # Step 3: FAISS + ML deferred but kicked off immediately in background.
    # This means ML models start loading right after deploy instead of blocking
    # the first real user request.
    app.state.faiss       = FAISSIndex()
    app.state.faiss_ready = False

    logger.info("Startup complete. ML warmup starting in background...")
    asyncio.create_task(ensure_faiss_ready())


async def ensure_faiss_ready():
    """
    Load FAISS indexes and ML models.
    Called automatically on startup (background task) and awaited by routes
    as a no-op guard once already ready.
    Explicit gc.collect() after each heavy step to keep peak RAM low.
    """
    if app.state.faiss_ready:
        return

    if not hasattr(app.state, "_faiss_lock"):
        app.state._faiss_lock = asyncio.Lock()

    async with app.state._faiss_lock:
        if app.state.faiss_ready:
            return

        logger.info("Lazy-loading FAISS + ML models...")

        # Load FAISS (triggers EmbeddingEngine → sentence-transformers → torch)
        await app.state.faiss.load_or_build()
        gc.collect()
        logger.info("FAISS indexes loaded.")

        # Build PhraseMatcher from Firestore job skills
        try:
            from ai_modules.nlp_processor import NLPProcessor
            n_patterns = NLPProcessor.build_matcher_from_firestore()
            logger.info(f"PhraseMatcher ready: {n_patterns} skill patterns from Firestore.")
        except Exception as e:
            logger.warning(f"PhraseMatcher build failed (non-fatal): {e}")

        # Pre-warm OCR correction vocab embeddings
        try:
            from services.resume_service import _get_ocr_vocab_embs
            _get_ocr_vocab_embs()
            logger.info("OCR vocab embeddings pre-warmed.")
        except Exception as e:
            logger.warning(f"OCR vocab pre-warm failed (non-fatal): {e}")

        # Load LLM skill implication map
        try:
            from ai_modules.skill_implications import load_implications_from_firestore
            n_implied = load_implications_from_firestore()
            logger.info(f"Skill implication map ready: {n_implied} skills covered.")
        except Exception as e:
            logger.warning(f"Implication map load failed (non-fatal): {e}")

        # Cache a single ResumeService instance on app.state so routes don't
        # re-construct it (and re-resolve all singletons) on every request.
        try:
            from services.resume_service import ResumeService
            app.state.resume_service = ResumeService(app.state.faiss)
            logger.info("ResumeService instance cached on app.state.")
        except Exception as e:
            logger.warning(f"ResumeService pre-cache failed (non-fatal): {e}")

        app.state.faiss_ready = True
        logger.info("All ML models ready. Serving requests.")


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
    """
    Pre-load FAISS + ML models after deploy.
    Safe to call multiple times — no-op if already warmed up.
    Warmup also starts automatically on startup, so this endpoint
    is mainly useful for health checks or manual verification.
    """
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