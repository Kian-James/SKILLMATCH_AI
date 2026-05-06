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

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Render injects $PORT at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2