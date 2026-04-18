"""
main.py - FastAPI application entry point

Render free tier: 512 MB RAM hard limit.

MEMORY STRATEGY:
  - Boot: only Firebase + synonym map (no ML, no FAISS, no torch)
  - First request (/warmup or real): lazy-load FAISS + ML models in sequence
    with explicit gc.collect() between steps to release peak allocation.
  - All heavy singletons (EmbeddingEngine, SkillClusterer, SkillNormalizer)
    are module-level in resume_service — shared across requests, allocated once.
  - job_metadata.json no longer stores numpy arrays — stays ~2 MB in RAM.
  - ctx_embeddings stored as float16 .npy — ~5 MB vs ~20 MB float32.

STARTUP SEQUENCE:
  1. init_firebase()                         — lightweight
  2. SkillNormalizer.load_dynamic_synonyms() — dict merge from Firestore, no ML
  3. FAISSIndex lazy-load on first request   — ML models load here
  4. Dynamic KNOWN_SKILLS injection          — after FAISS loads job corpus
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

    # Step 3: FAISS + ML deferred to first request
    app.state.faiss       = FAISSIndex()
    app.state.faiss_ready = False

    logger.info("Startup complete. ML models will lazy-load on first request.")


async def ensure_faiss_ready():
    """
    Load FAISS indexes and ML models on first request.
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

        # Build PhraseMatcher from Firestore job skills (replaces all static KNOWN_SKILLS)
        try:
            from ai_modules.nlp_processor import NLPProcessor
            n_patterns = NLPProcessor.build_matcher_from_firestore()
            logger.info(f"PhraseMatcher ready: {n_patterns} skill patterns from Firestore.")
        except Exception as e:
            logger.warning(f"PhraseMatcher build failed (non-fatal): {e}")

        # Pre-warm OCR correction vocab embeddings so first real request is fast
        try:
            from services.resume_service import _get_ocr_vocab_embs
            _get_ocr_vocab_embs()
            logger.info("OCR vocab embeddings pre-warmed.")
        except Exception as e:
            logger.warning(f"OCR vocab pre-warm failed (non-fatal): {e}")

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
    """
    Pre-load FAISS + ML models after deploy.
    Call once before going live to avoid cold-start delay on first real request.
    Memory-safe: uses same lazy-load path with gc.collect() between steps.
    """
    await ensure_faiss_ready()
    return {"status": "warmed up", "ml_ready": True}


@app.post("/admin/generate-synonyms")
async def generate_synonyms():
    """
    One-time admin job: call Zephyr 7B to generate skill aliases for all
    canonical skills and persist to Firestore (skill_synonyms collection).
    Safe to re-run — skips skills that already have Firestore entries.
    Call once after first deploy; subsequent startups load from Firestore.
    """
    from utils.skill_normalizer import SkillNormalizer
    processed = SkillNormalizer.generate_and_persist_synonyms()
    return {"status": "done", "skills_processed": processed}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )