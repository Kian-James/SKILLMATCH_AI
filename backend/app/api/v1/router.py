# backend/app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import resume, career, health

api_router = APIRouter()
api_router.include_router(resume.router, prefix="/resume", tags=["Resume"])
api_router.include_router(career.router, prefix="/career", tags=["Career Gap"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])
