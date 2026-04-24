"""
main.py - FastAPI application entry point

Render standard (2 GB): Gunicorn 3 workers WITHOUT --preload (see Dockerfile).

WHY NO --preload:
  --preload loads the app in the master process before forking workers.
  asyncio.create_task() in startup_event does not survive the fork —
  the task runs in the master but workers each run startup_event again,
  causing 3x simultaneous model loads = OOM crash on 2 GB RAM.

  Solution: each worker lazy-loads on its FIRST request, protected by a
  per-worker asyncio.Lock(). Since you already call /warmup after deploy,
  worker 1 gets warmed up by /warmup, workers 2 and 3 warm up on their
  first real /analyze request. Memory stays safe because loads are
  sequential per worker (not all 3 at once).

MEMORY STRATEGY:
  - Boot: only Firebase + synonym map (no ML, no FAISS, no torch)
  - ML loads lazily on first request per worker, protected by a lock
  - All heavy singletons (EmbeddingEngine, SkillClusterer, SkillNormalizer)
    are module-level in resume_service — shared within each worker process
  - ResumeService instance cached on app.state after warmup — not re-created
    per request
  - job_metadata.json no longer stores numpy arrays — stays ~2 MB in RAM
  - ctx_embeddings stored as float16 .npy — ~5 MB vs ~20 MB float32
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

    # Step 3: FAISS + ML deferred to first request (or /warmup call).
    # Do NOT use asyncio.create_task here — with Gunicorn multi-worker,
    # each worker runs startup_event independently. A background task would
    # cause all 3 workers to load models simultaneously = OOM on 2 GB RAM.
    # Instead each worker loads lazily on its first request, protected by
    # a per-worker lock so only one load happens per worker at a time.
    app.state.faiss       = FAISSIndex()
    app.state.faiss_ready = False

    logger.info("Startup complete. ML will lazy-load on first request or /warmup call.")


async def ensure_faiss_ready():
    """
    Load FAISS indexes and ML models on first request.
    Per-worker: each Gunicorn worker has its own event loop and app.state,
    so this lock is per-worker (not cross-process). That is correct —
    we want each worker to load its own model copy exactly once,
    not all at the same time.
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
    Pre-load FAISS + ML models on this worker after deploy.
    Call once after deploy to warm up worker 1. Workers 2 and 3
    will warm up automatically on their first real request.
    Safe to call multiple times — no-op if already warmed up.
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