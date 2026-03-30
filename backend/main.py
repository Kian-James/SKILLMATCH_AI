"""
main.py - FastAPI application entry point

Render free tier has 512MB RAM. To stay within limits:
- Firebase is initialized on startup (lightweight)
- FAISS index and ML models are lazy-loaded on first request
- spaCy uses en_core_web_sm on Render (set via SPACY_MODEL env var)
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

# App setup
app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume analysis with skill gap detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup - only initialize lightweight services
# ML models are lazy-loaded on first request to keep boot memory low
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Firebase...")
    init_firebase()

    # Create FAISS instance but do NOT load yet - lazy load on first request
    # This prevents OOM on Render free tier during boot
    app.state.faiss = FAISSIndex()
    app.state.faiss_ready = False

    logger.info("Application ready. ML models will load on first request.")


async def ensure_faiss_ready():
    """
    Called before any route that needs FAISS.
    Loads the index on first call, then caches it.
    Thread-safe via asyncio lock.
    """
    if app.state.faiss_ready:
        return

    if not hasattr(app.state, "_faiss_lock"):
        app.state._faiss_lock = asyncio.Lock()

    async with app.state._faiss_lock:
        # Double-check inside lock
        if not app.state.faiss_ready:
            logger.info("Lazy-loading FAISS index and ML models (first request)...")
            await app.state.faiss.load_or_build()
            app.state.faiss_ready = True
            logger.info("FAISS index ready.")


# Attach the lazy loader to app state so routes can call it
app.state.ensure_faiss_ready = ensure_faiss_ready

# Routers
app.include_router(resume_router, prefix="/api/v1", tags=["Resume"])
app.include_router(jobs_router,   prefix="/api/v1", tags=["Jobs"])
app.include_router(skills_router, prefix="/api/v1", tags=["Skills"])
app.include_router(growth_router, prefix="/api/v1", tags=["Growth"])


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "ml_ready": getattr(app.state, "faiss_ready", False),
    }


@app.get("/warmup")
async def warmup():
    """
    Call this endpoint once after deploy to pre-load ML models.
    Useful to avoid slow first-request on Render free tier.
    """
    await ensure_faiss_ready()
    return {"status": "warmed up", "ml_ready": True}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )