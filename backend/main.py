"""
main.py - FastAPI application entry point

Render free tier has 512MB RAM. To stay within limits:
- Firebase is initialized on startup (lightweight)
- FAISS index and ML models are lazy-loaded on first request
- spaCy uses en_core_web_sm on Render (set via SPACY_MODEL env var)

STARTUP SEQUENCE (new):
  1. init_firebase()
  2. SkillNormalizer.load_dynamic_synonyms()  — loads LLM-generated aliases from Firestore
  3. NLPProcessor dynamic KNOWN_SKILLS        — injected from Firestore job corpus
  4. FAISSIndex lazy-load                     — deferred to first request
"""

import asyncio
import os

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
    # Step 1: Firebase (lightweight — must be first)
    logger.info("Initializing Firebase...")
    init_firebase()

    # Step 2: Load LLM-generated synonyms from Firestore into SkillNormalizer
    logger.info("Loading dynamic skill synonyms...")
    from utils.skill_normalizer import SkillNormalizer
    SkillNormalizer.load_dynamic_synonyms()

    # Step 3: Create FAISS instance — lazy-load on first request
    app.state.faiss       = FAISSIndex()
    app.state.faiss_ready = False

    logger.info("Application ready. ML models will load on first request.")


async def ensure_faiss_ready():
    """
    Lazy-load FAISS index and ML models on first request.
    Thread-safe via asyncio lock.
    Also seeds NLP dynamic KNOWN_SKILLS from job corpus on first load.
    """
    if app.state.faiss_ready:
        return

    if not hasattr(app.state, "_faiss_lock"):
        app.state._faiss_lock = asyncio.Lock()

    async with app.state._faiss_lock:
        if not app.state.faiss_ready:
            logger.info("Lazy-loading FAISS index and ML models (first request)...")
            await app.state.faiss.load_or_build()

            # Inject dynamic KNOWN_SKILLS from job corpus into NLPProcessor
            try:
                from services.firebase_service import fetch_all_jobs
                from ai_modules.nlp_processor import NLPProcessor, KNOWN_SKILLS, _build_phrase_matcher, _load_nlp_model
                jobs = fetch_all_jobs()
                job_skills = {s for job in jobs for s in job.get("skills", [])}
                all_skills = list(set(KNOWN_SKILLS) | job_skills)
                nlp = _load_nlp_model()
                # Re-build matcher with expanded skill list
                NLPProcessor._dynamic_matcher = _build_phrase_matcher(nlp, all_skills)
                logger.info(
                    f"NLP dynamic KNOWN_SKILLS: {len(KNOWN_SKILLS)} static + "
                    f"{len(job_skills - set(KNOWN_SKILLS))} from Firestore = "
                    f"{len(all_skills)} total."
                )
            except Exception as e:
                logger.warning(f"Dynamic KNOWN_SKILLS injection failed (non-fatal): {e}")

            app.state.faiss_ready = True
            logger.info("FAISS index and ML models ready.")


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
    """Pre-load ML models and FAISS indexes. Call once after deploy."""
    await ensure_faiss_ready()
    return {"status": "warmed up", "ml_ready": True}


@app.post("/admin/generate-synonyms")
async def generate_synonyms():
    """
    One-time admin endpoint: call Zephyr 7B to generate skill aliases for all
    canonical skills and persist them to Firestore (skill_synonyms collection).
    Safe to re-run — skips skills that already have entries.
    Call this once after first deploy, then synonyms load from Firestore on startup.
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
