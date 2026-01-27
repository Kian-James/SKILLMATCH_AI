backend/
├── app/
│   ├── main.py                  # FastAPI app entrypoint (/)
│   ├── core/
│   │   ├── config.py             # App configs (env, algorithm switch) (/)
│   │   ├── constants.py          # Global constants (/)
│   │   ├── logging.py            # Structured logging (X)
│   │   └── security.py           # Privacy enforcement (no persistence) (X)
│   │
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── resume.py     # Resume → Job Matching (/)
│   │   │   │   ├── career.py     # Goal-based gap analysis (X)
│   │   │   │   └── health.py     # Health checks (/)
│   │   │   └── router.py         # API router (/)
│   │
│   ├── services/
│   │   ├── ocr/
│   │   │   ├── base.py           # OCR interface (/)
│   │   │   ├── trocr.py (/)
│   │   │   ├── donut.py (X)
│   │   │   └── paddle.py (X)
│   │   │
│   │   ├── nlp/
│   │   │   ├── parser.py         # Skill / cert / experience extraction (/)
│   │   │   └── normalizer.py     # Text cleanup & PII stripping (/)
│   │   │
│   │   ├── embeddings/
│   │   │   ├── generator.py      # Embedding creation (X)
│   │   │   └── models.py         # HF embedding model loader (X)
│   │   │
│   │   ├── matching/
│   │   │   ├── base.py           # Matching interface  (/)
│   │   │   ├── sbert.py          # SBERT matcher (/)
│   │   │   ├── knn.py            # KNN matcher (/)
│   │   │   └── selector.py       # Runtime algorithm switch  (/)
│   │   │
│   │   ├── jobs/
│   │   │   ├── ranker.py         # Job scoring & ranking (X)
│   │   │   └── links.py          # Indeed / JobStreet / LinkedIn URLs (X)
│   │   │
│   │   ├── learning/
│   │   │   ├── gap_analysis.py   # Missing skills & certs  (X)
│   │   │   └── recommender.py    # Learning paths & projects (X)
│   │
│   ├── repositories/
│   │   ├── firestore/
│   │   │   ├── client.py         # Firestore init (/)
│   │   │   ├── jobs.py (/)
│   │   │   ├── skills.py (/)
│   │   │   ├── certifications.py (/)
│   │   │   ├── embeddings.py (X)
│   │   │   └── metrics.py  (/)
│   │
│   ├── models/
│   │   ├── request.py            # API request schemas (X)
│   │   └── response.py           # API response schemas (X)
│   │
│   ├── utils/
│   │   ├── memory.py             # In-memory lifecycle enforcement (X)
│   │   ├── timing.py             # Execution time metrics (X)
│   │   └── validators.py         # AI output validation (X)
│   │
│
├── requirements.txt
├── .env
└── README.md
