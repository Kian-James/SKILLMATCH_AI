# Dockerfile - stable + optimized for Render free tier

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Install core numeric stack (LOCKED) ─────────────────────
RUN pip install --no-cache-dir \
    numpy==1.24.4

# ── Layer 2: Install PyTorch CPU ─────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: ML dependencies (FAISS-compatible versions) ─────────────
RUN pip install --no-cache-dir \
    faiss-cpu==1.7.4 \
    scikit-learn==1.3.2 \
    sentence-transformers==2.6.1

# ── 🔥 CRITICAL: Re-lock NumPy AFTER installs ────────────────────────
RUN pip install --no-cache-dir numpy==1.24.4 --force-reinstall

# ── Layer 4: spaCy + model (stable install) ──────────────────────────
RUN pip install --no-cache-dir spacy==3.7.4 && \
    pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# ── Layer 5: App dependencies ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 🔥 CRITICAL AGAIN: Re-lock NumPy AFTER requirements ──────────────
RUN pip install --no-cache-dir numpy==1.24.4 --force-reinstall

# ── Layer 6: Application code ────────────────────────────────────────
COPY . .

RUN mkdir -p faiss_index embedding_cache

# ── Environment optimizations (VERY IMPORTANT for Render) ───────────
ENV PORT=8000
ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT