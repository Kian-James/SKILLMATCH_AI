"""
main.py — FastAPI application entry point
Initializes Firebase, FAISS index, and mounts all routers
"""

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

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume analysis with skill gap detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Flutter app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Firebase...")
    init_firebase()

    logger.info("Loading FAISS index...")
    # This will build the index from Firestore if no cached index exists
    faiss_instance = FAISSIndex()
    await faiss_instance.load_or_build()
    # Store singleton on app state so routes can access it
    app.state.faiss = faiss_instance

    logger.info("Application ready.")

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(resume_router, prefix="/api/v1", tags=["Resume"])
app.include_router(jobs_router,   prefix="/api/v1", tags=["Jobs"])
app.include_router(skills_router, prefix="/api/v1", tags=["Skills"])
app.include_router(growth_router, prefix="/api/v1", tags=["Growth"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )
