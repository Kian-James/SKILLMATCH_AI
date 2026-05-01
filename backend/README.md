# AI-Based Resume Matching Backend

FastAPI backend that OCR-extracts skills from a résumé PDF, matches them against jobs stored in Firestore, and returns ranked career recommendations.

---

## Prerequisites

### 1 — Python 3.10+
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Poppler (system package — required by pdf2image)

| OS | Command |
|----|---------|
| **Ubuntu / Debian** | `sudo apt install poppler-utils` |
| **macOS** | `brew install poppler` |
| **Windows** | `winget install poppler` or `choco install poppler` — then add `bin/` to your `PATH` |

> No local zip or `POPPLER_PATH` variable needed — `pdf2image` finds it on your system `PATH` automatically.

### 3 — Firebase service-account key
1. Go to **Firebase Console → Project Settings → Service Accounts**
2. Click **Generate new private key** → save the JSON file
3. Place it anywhere accessible (e.g. project root) and point `.env` at it

---

## Setup

```bash
cp .env.example .env
# Edit .env with your actual values
```

`.env` variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FIREBASE_CREDENTIALS_PATH` | Path to your `serviceAccountKey.json` | *(required)* |
| `FIRESTORE_JOBS_COLLECTION` | Firestore collection name for jobs | `jobs` |
| `HOST` | API bind host | `0.0.0.0` |
| `PORT` | API bind port | `8000` |

---

## Seed Firestore

Run **once** (or whenever you update the job catalogue):

```bash
python generate_jobs.py
```

This writes all 7 job documents into your Firestore collection.

---

## Run the API

```bash
python main.py
# or
uvicorn main:app --reload
```

---

## Endpoint

### `POST /thesis-full-analysis`

**Form fields:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | PDF upload | The résumé to analyse |
| `expected_skills_json` | JSON string | `{"expected_skills": ["Node.js", "Docker", ...]}` |

**Response:** pipeline accuracy report, top-5 job recommendations fetched live from Firestore, skill gap analysis.

---

## Models

- **SBERT** — `sentence-transformers/all-MiniLM-L6-v2` is downloaded automatically from Hugging Face Hub on first run and cached locally by the `sentence-transformers` library. No manual download needed.
- **EasyOCR** — language models are downloaded automatically on first use.

---

## Project structure

```
.
├── main.py              # FastAPI app (fetches jobs from Firestore)
├── matcher.py           # OCR, NLP, FAISS, SBERT helpers
├── generate_jobs.py     # One-time script to seed Firestore
├── requirements.txt
├── .env.example
└── input_data_for_testing/
```
