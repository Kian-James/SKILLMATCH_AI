# Dockerfile
# Used by Render for the backend deployment.
# Base image includes Tesseract OCR pre-installed.
# Uses CPU-only PyTorch to keep image size manageable.

FROM python:3.11-slim

# System dependencies
# tesseract-ocr: OCR engine
# poppler-utils: PDF to image conversion (pdf2image)
# libgl1: required by some image processing libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY ./backend/requirements.txt .

# Install CPU-only PyTorch first (avoids pulling CUDA build)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.0+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model (sm fits Render free tier RAM)
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create directories for FAISS index and embedding cache
RUN mkdir -p faiss_index embedding_cache

# Render injects PORT at runtime — default to 8000 for local Docker runs
ENV PORT=8000

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT