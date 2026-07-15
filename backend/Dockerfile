FROM python:3.11-slim

WORKDIR /app

# CPU-only torch, installed first so pip doesn't pull the ~2GB GPU build.
# Must be >=2.5 — transformers>=4.47 requires torch.float8_e8m0fnu, added in
# torch 2.5. (2.13.0 pinned here to match the version already confirmed
# working locally.)
RUN pip install --no-cache-dir \
    torch==2.13.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-bake the SBERT embedding model into the image so it doesn't download
# on the first real request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Force CPU-only mode and disable parallelism overhead
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV KMP_DUPLICATE_LIB_OK=TRUE

# --workers 1: each extra worker duplicates the loaded model in memory —
# not affordable on a 512MB instance. Raise this only on a bigger plan.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
