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

# Layer 1: numpy
RUN pip install --no-cache-dir numpy==1.26.4

# Layer 2: PyTorch CPU only
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Layer 3: Heavy ML deps
RUN pip install --no-cache-dir \
    sentence-transformers==3.0.0 \
    faiss-cpu==1.8.0 \
    scikit-learn==1.4.2

# Layer 4: spaCy + model via direct wheel URL
RUN pip install --no-cache-dir spacy==3.7.5 && \
    pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Layer 5: App requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pdfplumber==0.11.0

# Layer 6: Pre-download the embedding model into the image
# all-MiniLM-L6-v2 is only 80MB — fits comfortably in free tier RAM
# Baking it into the image means no download spike at runtime startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Layer 7: Application code
COPY . .

RUN mkdir -p faiss_index embedding_cache

ENV PORT=8000
ENV SPACY_MODEL=en_core_web_sm

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT