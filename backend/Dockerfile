# Dockerfile - optimised for Render free tier
# Layers ordered so heavy deps are cached and rarely rebuilt

FROM python:3.11-slim

# System dependencies
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

# Layer 1: numpy first (torch depends on it)
RUN pip install --no-cache-dir numpy==1.26.4

# Layer 2: PyTorch CPU-only via direct index URL
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Layer 3: Heavy ML deps
RUN pip install --no-cache-dir \
    sentence-transformers==3.0.0 \
    faiss-cpu==1.8.0 \
    scikit-learn==1.4.2

# Layer 4: spaCy + model
# Hardcode the model name directly - do NOT use env var here,
# the SPACY_MODEL env var is only for runtime (which model to load),
# not for the build-time download command
RUN pip install --no-cache-dir spacy==3.7.4 && \
    python -m spacy download en_core_web_sm

# Layer 5: App requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 6: Application code
COPY . .

RUN mkdir -p faiss_index embedding_cache

ENV PORT=8000
ENV SPACY_MODEL=en_core_web_sm

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT