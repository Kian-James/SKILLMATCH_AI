# ── Base image ───────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

WORKDIR /app

# ── Upgrade pip ─────────────────────────────────────────────
RUN python -m pip install --upgrade pip

# ── Core numeric stack (FAISS-safe) ─────────────────────────
RUN pip install --no-cache-dir numpy==1.24.4 scipy==1.11.2

# ── PyTorch CPU ONLY ────────────────────────────────────────
RUN pip install --no-cache-dir torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu

# ── ML dependencies ─────────────────────────────────────────
RUN pip install --no-cache-dir faiss-cpu==1.7.4 scikit-learn==1.3.2 sentence-transformers==2.6.1

# ── spaCy + model ──────────────────────────────────────────
RUN pip install --no-cache-dir spacy==3.7.4 typer==0.9.0
RUN python -m spacy download en_core_web_sm

# ── App-specific dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ───────────────────────────────────────────────
COPY . .

RUN mkdir -p faiss_index embedding_cache

# ── Runtime optimizations ──────────────────────────────────
ENV PORT=8000
ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]