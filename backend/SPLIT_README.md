# Splitting EasyOCR into its own Render service

This backend has been split into two independent services so each one
fits under Render's 512MB free-tier RAM limit:

1. **`ocr-service/`** — FastAPI app that only does EasyOCR text
   extraction. Exposes `POST /extract-text`.
2. **`backend/`** (this repo, updated) — everything else: SBERT
   embeddings, FAISS, skill/cert matching, Firestore. No longer imports
   `easyocr` or `pdf2image` — it calls `ocr-service` over HTTP instead.

Nothing else in your pipeline changed. `core/extraction.py` still exposes
`extract_text(file_path) -> (text, quality, is_native_pdf)` with the exact
same signature — it just delegates the actual OCR work to the other
service now, so `pipeline.py`, `core/skills.py`, etc. don't need edits.

## Deploy order

### 1. Deploy `ocr-service` first

- New Render **Web Service**, root directory `ocr-service/`.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Important:** `pdf2image` needs the `poppler-utils` system package
  (specifically `pdftoppm`), which isn't installed by default on Render's
  native Python environment. The included `render.yaml` adds it via
  `aptPackages`. If you deploy without `render.yaml` (dashboard-only),
  add `poppler-utils` under your service's "Native Environment" ->
  system packages, or switch to a Docker-based deploy with
  `apt-get install -y poppler-utils` in the Dockerfile.
- Once deployed, note the public URL, e.g.
  `https://your-ocr-service.onrender.com`

### 2. Deploy the main backend

- Set the env var `OCR_SERVICE_URL` to the URL from step 1, e.g.
  `OCR_SERVICE_URL=https://your-ocr-service.onrender.com`
- Deploy as before (build/start commands unchanged).

## Memory expectations

- `ocr-service`: `easyocr` + `torch` (CPU) is the main cost. This is
  usually 300–450MB resident once the reader is loaded, so it should fit
  in 512MB on its own but leaves little headroom — see "If it's still
  too tight" below.
- `backend`: now carries `sentence-transformers` + `faiss` +
  `firebase-admin`, without EasyOCR's torch models loaded alongside it.
  This alone should meaningfully cut peak RAM versus the combined
  service.

## If `ocr-service` is still too tight on 512MB

EasyOCR's own weight (torch + detection/recognition models) is the
binding constraint, not the split itself. Options, roughly in order of
effort:
- Use Render's paid 512MB→2GB tier for just `ocr-service` (cheaper than
  upgrading the whole backend).
- Swap EasyOCR for a lighter OCR engine (e.g. `rapidocr-onnxruntime`,
  which avoids the torch dependency entirely and is typically much
  smaller in memory).
- Restrict `EASYOCR_LANGUAGES` to only what you need (`en` only, as
  configured here, is already the minimal case).

## Local development

Run both services locally with `OCR_SERVICE_URL=http://localhost:8001`
(the default) and start `ocr-service` on port 8001:

```bash
# terminal 1
cd ocr-service
uvicorn main:app --port 8001

# terminal 2
cd backend
uvicorn app.main:app --port 8000
```
