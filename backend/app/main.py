from fastapi import FastAPI
from app.api.v1.router import api_router

app = FastAPI(
    title="AI Resume Analyzer API",
    version="1.0.0",
)

# Register API routes
app.include_router(api_router, prefix="/api/v1")
