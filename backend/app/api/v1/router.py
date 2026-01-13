from fastapi import APIRouter
from app.api.v1.endpoints import resume, jobs, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(resume.router)
api_router.include_router(jobs.router)
api_router.include_router(users.router)
