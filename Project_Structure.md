# Resume Analyzer — Project Structure

```
resume-analyzer/
│
├── 📄 README.md                          ← Quick-start overview
├── 📄 STRUCTURE.md                       ← This file
│
├── 🐍 backend/                           ← FastAPI Python backend
│   ├── main.py                           ← App entry point, lifespan hooks
│   ├── requirements.txt                  ← All Python dependencies
│   ├── .env.example                      ← Required environment variables
│   │
│   ├── core/                             ← Shared infrastructure
│   │   ├── firebase.py                   ← Firebase Admin SDK + Firestore helpers
│   │   ├── model_loader.py               ← spaCy + SBERT singleton loader + embedding cache
│   │   └── faiss_index.py                ← FAISS vector index (build, search, rebuild)
│   │
│   ├── ai/                               ← AI / ML pipeline modules
│   │   ├── ocr.py                        ← PDF + image text extraction (pdfplumber + EasyOCR)
│   │   ├── nlp_extractor.py              ← Skills + cert extraction (regex, spaCy NER, SBERT)
│   │   ├── job_recommender.py            ← FAISS-based top-5 job recommendation
│   │   ├── skill_gap.py                  ← Skill gap computation + learning resource generation
│   │   └── growth_tracker.py             ← Historical growth analysis + trend detection
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── resume.py                 ← POST /upload-resume
│   │       ├── jobs.py                   ← GET  /recommend-jobs
│   │       ├── skills.py                 ← POST /skill-gap  +  POST /skill-gap/resources
│   │       ├── growth.py                 ← GET  /growth-tracking
│   │       └── auth.py                   ← POST /auth/verify
│   │
│   └── models/
│       └── schemas.py                    ← All Pydantic v2 request/response models
│
├── 🤖 dataset_generator/
│   └── generate_dataset.py               ← Zephyr-7B dataset generator + Firestore uploader
│
├── 📱 flutter_app/
│   ├── pubspec.yaml                      ← Flutter dependencies
│   ├── lib/
│   │   ├── main.dart                     ← App entry, Firebase init, AuthGate router
│   │   │
│   │   ├── theme/
│   │   │   └── app_theme.dart            ← Design tokens, light/dark MaterialTheme
│   │   │
│   │   ├── models/
│   │   │   └── models.dart               ← Dart data classes (AnalysisResult, JobMatch, etc.)
│   │   │
│   │   ├── services/
│   │   │   ├── api_service.dart          ← Dio HTTP client for all FastAPI endpoints
│   │   │   └── auth_service.dart         ← Firebase Auth wrapper (email, Google, anon)
│   │   │
│   │   ├── screens/
│   │   │   ├── auth/
│   │   │   │   └── login_screen.dart     ← Email/Password + Google + Anonymous login UI
│   │   │   ├── home/
│   │   │   │   └── home_screen.dart      ← Bottom-nav shell (IndexedStack)
│   │   │   ├── upload/
│   │   │   │   └── upload_screen.dart    ← File picker + upload + extracted skills view
│   │   │   ├── jobs/
│   │   │   │   └── jobs_screen.dart      ← Ranked job recommendation cards
│   │   │   ├── skill_gap/
│   │   │   │   ├── skill_gap_screen.dart ← Radial score + matched/missing skill chips
│   │   │   │   └── learning_resources_sheet.dart ← Bottom sheet with tappable resources
│   │   │   └── growth/
│   │   │       └── growth_screen.dart    ← fl_chart time-series + stat summary cards
│   │   │
│   │   └── widgets/
│   │       └── skill_chip.dart           ← SkillChip (present/missing/cert) + JobCard widget
│   │
│   └── assets/
│       └── images/
│           └── google_logo.png           ← Google sign-in button icon
│
└── 📚 docs/
    ├── STRUCTURE.md                      ← This file
    ├── SETUP.md                          ← Step-by-step installation guide
    ├── FIREBASE_SETUP.md                 ← Firebase console + Firestore rules
    └── API_REFERENCE.md                  ← All endpoint specs with examples
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flutter App                              │
│  Login → Upload → Jobs → Skill Gap → Growth                    │
│  Firebase Auth SDK  ←→  Firebase Auth Service (cloud)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS (Dio + Bearer token)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│  /upload-resume  /recommend-jobs  /skill-gap  /growth-tracking │
│                                                                 │
│  ┌────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  OCR   │  │    NLP    │  │   SBERT  │  │  FAISS Index  │   │
│  │(EasyOCR│  │  spaCy +  │  │ Embeddings│  │  (job vecs)   │   │
│  │pdfplumb│  │  regex    │  │  + cache │  │  cosine sim   │   │
│  └────────┘  └───────────┘  └──────────┘  └───────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Firebase Admin SDK
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Firebase (Google Cloud)                    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Firebase Auth│  │  Firestore   │  │  Collections:         │ │
│  │              │  │              │  │  • users              │ │
│  │  • Email/Pw  │  │  (NoSQL DB)  │  │  • resumes            │ │
│  │  • Google    │  │              │  │  • job_dataset        │ │
│  │  • Anonymous │  │              │  │  • user_progress      │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                  Dataset Generator
          (Zephyr-7B → normalise → Firestore)
```

---

## Firestore Data Model

### Collection: `users`
```json
{
  "uid": "firebase-uid",
  "email": "user@example.com",
  "display_name": "Jane Doe",
  "provider": "google.com",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### Collection: `resumes`
> Only metadata is stored — the resume file is never persisted.
```json
{
  "user_id": "firebase-uid",
  "extracted_skills": ["Python", "FastAPI", "Docker"],
  "extracted_certs": ["AWS Certified Developer"],
  "skill_count": 3,
  "top_jobs": [
    {
      "job_title": "Backend Engineer",
      "match_score": 87.3,
      "matching_skills": ["Python", "FastAPI"],
      "missing_skills": ["Kubernetes"]
    }
  ],
  "analysis_meta": {
    "filename": "resume.pdf",
    "file_size": 204800
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Collection: `job_dataset`
```json
{
  "job_title": "Machine Learning Engineer",
  "skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "FAISS"],
  "certifications": ["AWS Machine Learning Specialty", "TensorFlow Developer Certificate"],
  "_dedup_key": "md5hash"
}
```

### Collection: `user_progress`
```json
{
  "user_id": "firebase-uid",
  "date": "2024-06-01T00:00:00Z",
  "skill_count": 14,
  "top_score": 78.5,
  "skills": ["Python", "React", "Docker"]
}
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/verify` | Verify Firebase ID token, create user profile |
| `POST` | `/upload-resume` | Upload PDF/image, extract skills, return analysis |
| `GET`  | `/recommend-jobs` | Get top-5 job matches for a given analysis |
| `POST` | `/skill-gap` | Compute skill gap against a target job title |
| `POST` | `/skill-gap/resources` | Get learning resources for a single skill |
| `GET`  | `/growth-tracking` | Get full growth report (time-series + delta) |
| `GET`  | `/health` | Health check |

---

## AI Pipeline Flow

```
Resume File (PDF / Image)
        │
        ▼
   ┌─────────┐
   │   OCR   │  pdfplumber (text PDF) → EasyOCR (scanned/image)
   └────┬────┘
        │ raw text
        ▼
   ┌─────────┐
   │   NLP   │  1. Regex cert patterns
   │Extractor│  2. Vocabulary keyword match
   │         │  3. spaCy NER (ORG, PRODUCT)
   │         │  4. SBERT semantic validation
   └────┬────┘
        │ skills[], certs[]
        ▼
   ┌──────────────┐         ┌─────────────────┐
   │   SBERT      │         │  FAISS Index     │
   │  Embeddings  │────────▶│  (job vectors)   │
   │ (query vec)  │         │  cosine search   │
   └──────────────┘         └────────┬────────┘
                                     │ top-5 jobs + scores
                                     ▼
                              AnalysisResult
                          (saved to Firestore,
                           file is discarded)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FAISS IndexFlatIP** | Exact search (not approximate) for our dataset size (< 10k jobs). L2-normalised vectors → inner product = cosine similarity. |
| **EasyOCR over Tesseract** | Better accuracy on stylised resume fonts without extra training data. |
| **SBERT all-roberta-large** | Best semantic quality for skill matching. Cached to avoid recomputation. |
| **pdfplumber + OCR fallback** | Text-layer PDFs are extracted in milliseconds. OCR only activates for scanned/image PDFs. |
| **No resume file storage** | Privacy-first. Only extracted metadata (skills, certs) is stored in Firestore. |
| **Semantic skill matching** | Catches synonyms ("ML" ≈ "Machine Learning") that pure string matching misses. |
| **Zephyr-7B for dataset** | Free via HF inference API. Generates varied, realistic job/skill combinations without manual curation. |

---

## Environment Variables

### Backend (`.env`)
```bash
# Firebase
FIREBASE_SA_PATH=serviceAccountKey.json          # OR
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}

# Hugging Face (for dataset generator + LLM resources)
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Flutter (`lib/firebase_options.dart`)
Auto-generated by `flutterfire configure`. Contains:
- `apiKey`, `appId`, `messagingSenderId`, `projectId`
- Platform-specific options for Android/iOS/Web