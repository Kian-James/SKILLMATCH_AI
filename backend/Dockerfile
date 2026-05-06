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

# Install CPU-only torch first — prevents pip from pulling the ~2GB GPU build
# which would instantly blow past 512MB at install time and at runtime
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Force CPU-only mode for torch and disable parallelism overhead
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV KMP_DUPLICATE_LIB_OK=TRUE

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2