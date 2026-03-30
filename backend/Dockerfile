# Dockerfile - optimised for Render free tier
# Uses Python slim + installs torch CPU wheel directly (no CUDA bloat)
# Layers ordered so heavy deps are cached and rarely rebuilt

FROM python:3.11-slim

# System deps in one layer - minimised to what's actually needed
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

# ── Layer 1: Install numpy first (torch depends on it) ─────────────────────
RUN pip install --no-cache-dir numpy==1.24.4

# ── Layer 2: Install PyTorch CPU-only via direct wheel URL ─────────────────
# Direct URL avoids pip having to resolve the extra index, faster + reliable
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: Install remaining heavy ML deps ───────────────────────────────
RUN pip install --no-cache-dir \
    sentence-transformers==3.0.0 \
    faiss-cpu==1.8.0 \
    scikit-learn==1.4.2

# ── Layer 4: Install spaCy + download sm model ─────────────────────────────
RUN pip install --no-cache-dir spacy==3.7.4 && \
    pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# ── Layer 5: Install the rest of the requirements ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 6: Application code (changes most often - last layer) ────────────
COPY . .

RUN mkdir -p faiss_index embedding_cache

ENV PORT=8000

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT