# ocr-service/Dockerfile
#
# Docker-based deploy is required here (not Render's native Python runtime)
# because pdf2image shells out to the `pdftoppm` binary from poppler-utils,
# which isn't available in Render's native runtime and can't be added to it.

FROM python:3.11-slim

# poppler-utils provides pdftoppm, which pdf2image calls under the hood.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT at runtime; default 8001 covers local `docker run`.
ENV PORT=8001
EXPOSE 8001

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]