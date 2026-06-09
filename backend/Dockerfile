FROM python:3.11-slim

# System dependencies
# poppler-utils  — pdf2image
# libgl1         — OpenCV (EasyOCR dependency)
# libglib2.0-0   — also needed by OpenCV
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch 2.5.1 — has float8_e8m0fnu which transformers>=4.47 needs.
# Must install before requirements.txt so pip doesn't pull the 2GB GPU build.
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-bake EasyOCR models into the image so they never download at runtime
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Force CPU-only mode and disable parallelism overhead
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV KMP_DUPLICATE_LIB_OK=TRUE

# workers=1 — each worker spawns its own OCR child process.
# Two workers = two EasyOCR processes = instant OOM on 512 MB.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1