# Dockerfile - stable for Render free tier (UPDATED)

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

# ── Layer 1: Core numeric stack (FAISS-safe) ─────────────────────────
RUN pip install --no-cache-dir numpy==1.24.4 scipy==1.11.2

# ── Layer 2: PyTorch CPU ONLY (no CUDA) ──────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: ML dependencies (compatible versions) ───────────────────
RUN pip install --no-cache-dir \
    faiss-cpu==1.7.4 \
    scikit-learn==1.3.2 \
    sentence-transformers==2.6.1

# ── Layer 4: spaCy + model ───────────────────────────────────────────
RUN pip install --no-cache-dir spacy==3.7.4 typer==0.9.0 && \
    pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# ── Layer 5: App-specific dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 6: Lock NumPy AGAIN (final protection) ────────────────────
RUN pip install --no-cache-dir numpy==1.24.4 --force-reinstall

# ── App code ─────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p faiss_index embedding_cache

# ── Runtime optimizations ────────────────────────────────────────────
ENV PORT=8000
ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]