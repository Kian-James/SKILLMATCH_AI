"""
firestore_client.py
===================
Firestore integration for the AI Resume Analyzer.

Actual Firestore schema
-----------------------
Collection : jobs  (env: FIRESTORE_JOBS_COLLECTION, default "jobs")
  Document ID : auto-generated string
  Fields:
    title                    : str
    industry                 : str
    description              : str
    required_skills          : list[str]   — canonical skill names (this IS the allowlist)
    required_certifications  : list[str]   — canonical cert names (cert allowlist)
    embedding                : list[float] — pre-computed SBERT vector ([] when not yet set)

Collection : resume_results  (env: FIRESTORE_RESULTS_COLLECTION, default "resume_results")
  Document ID : auto-generated
  Fields:
    file_name       : str
    analyzed_at     : timestamp
    ocr_quality     : float
    skills          : list[str]
    certificates    : list[str]
    validation      : dict | None
    top_matches     : list[dict]

There is NO separate skills collection.
The skills allowlist is derived at runtime by aggregating every unique value
found in required_skills across all job documents — identical to check_skills.py.
The certs allowlist is derived the same way from required_certifications.

Environment variables (loaded from .env by the caller):
    FIREBASE_CREDENTIALS_PATH       — path to serviceAccountKey.json  (required)
    FIRESTORE_JOBS_COLLECTION       — jobs collection name             (default: jobs)
    FIRESTORE_RESULTS_COLLECTION    — results collection name          (default: resume_results)
"""

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    log.warning(
        "[firestore] firebase-admin not installed. "
        "Run: pip install firebase-admin"
    )

# ── Collection names — match your existing env vars exactly ──────────────────
FIREBASE_CREDENTIALS_PATH    = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
FIRESTORE_JOBS_COLLECTION    = os.getenv("FIRESTORE_JOBS_COLLECTION",    "jobs")
FIRESTORE_RESULTS_COLLECTION = os.getenv("FIRESTORE_RESULTS_COLLECTION", "resume_results")

_db = None   # Firestore client singleton


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    """
    Return a Firestore client, initialising the Firebase app on first call.
    Safe to call multiple times — reuses the singleton on subsequent calls.
    Also safe when main.py has already called firebase_admin.initialize_app()
    because it checks _apps before initialising.
    """
    global _db

    if _db is not None:
        return _db

    if not _FIREBASE_AVAILABLE:
        raise RuntimeError(
            "firebase-admin is not installed. Run: pip install firebase-admin"
        )

    if not firebase_admin._apps:
        if not FIREBASE_CREDENTIALS_PATH:
            raise RuntimeError(
                "FIREBASE_CREDENTIALS_PATH is not set. "
                "Add it to your .env file and restart."
            )
        if not os.path.isfile(FIREBASE_CREDENTIALS_PATH):
            raise FileNotFoundError(
                f"Service account file not found: {FIREBASE_CREDENTIALS_PATH}"
            )
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        log.info(f"[firestore] Initialised with: {FIREBASE_CREDENTIALS_PATH}")

    _db = firestore.client()
    log.info("[firestore] Client ready.")
    return _db


# ══════════════════════════════════════════════════════════════════════════════
# FETCH — JOBS CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_jobs_corpus() -> list:
    """
    Fetch all job documents from Firestore, sorted by their numeric 'id'
    field when present (mirrors the ordering in the original main.py).

    Each returned dict has the shape expected by faiss_search_and_score():
      {
        "id"                     : str          — Firestore document ID
        "title"                  : str
        "industry"               : str
        "description"            : str
        "required_skills"        : list[str]
        "required_certifications": list[str]    — [] when field absent
        "embedding"              : list[float]  — [] when not yet computed
      }

    Returns
    -------
    List of job dicts.  Raises HTTPException(503) when the collection is empty.
    """
    from fastapi import HTTPException

    db   = get_db()
    docs = db.collection(FIRESTORE_JOBS_COLLECTION).stream()

    corpus = []
    for doc in docs:
        data = doc.to_dict()
        corpus.append({
            "id":                      doc.id,
            "title":                   data.get("title",                   ""),
            "industry":                data.get("industry",                ""),
            "description":             data.get("description",             ""),
            "required_skills":         data.get("required_skills",         []),
            "required_certifications": data.get("required_certifications", []),
            # Keep [] instead of None so callers never have to guard for None
            "embedding":               data.get("embedding") or [],
        })

    if not corpus:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No jobs found in Firestore collection "
                f"'{FIRESTORE_JOBS_COLLECTION}'. "
                "Run generate_jobs.py first to seed the collection."
            ),
        )

    # Mirror the sort from the original main.py
    corpus.sort(key=lambda j: j.get("id", 0))

    log.info(
        f"[firestore] Fetched {len(corpus)} jobs "
        f"from '{FIRESTORE_JOBS_COLLECTION}'."
    )
    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# DERIVE — SKILLS ALLOWLIST FROM JOBS CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def derive_skills_from_corpus(jobs_corpus: list) -> list:
    """
    Build the skills allowlist by aggregating every unique value found in
    required_skills across all job documents.

    This is identical to what check_skills.py does — no separate skills
    collection is needed.

    Parameters
    ----------
    jobs_corpus : list returned by fetch_jobs_corpus()

    Returns
    -------
    Sorted list of unique skill strings.
    """
    all_skills: set = set()

    for job in jobs_corpus:
        for skill in job.get("required_skills", []):
            stripped = skill.strip()
            if stripped:
                all_skills.add(stripped)

    skill_list = sorted(all_skills)
    log.info(f"[firestore] Derived {len(skill_list)} unique skills from corpus.")
    return skill_list


# ══════════════════════════════════════════════════════════════════════════════
# DERIVE — CERTS ALLOWLIST FROM JOBS CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def derive_certs_from_corpus(jobs_corpus: list) -> list:
    """
    Build the certs allowlist by aggregating every unique value found in
    required_certifications across all job documents.

    If no job documents have a required_certifications field, returns [] —
    extract_certificates() handles an empty allowlist by failing open (keeping
    all extracted certs), so this is safe.

    Parameters
    ----------
    jobs_corpus : list returned by fetch_jobs_corpus()

    Returns
    -------
    Sorted list of unique certification strings.
    """
    all_certs: set = set()

    for job in jobs_corpus:
        for cert in job.get("required_certifications", []):
            stripped = cert.strip()
            if stripped:
                all_certs.add(stripped)

    cert_list = sorted(all_certs)
    log.info(f"[firestore] Derived {len(cert_list)} unique certs from corpus.")
    return cert_list


# ══════════════════════════════════════════════════════════════════════════════
# WRITE — CACHE JOB EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def cache_job_embeddings(jobs_corpus: list) -> None:
    """
    Write SBERT embeddings back to Firestore for jobs that did not previously
    have one (embedding was [] on fetch).  Jobs that already had a non-empty
    embedding are skipped to avoid redundant writes.

    Parameters
    ----------
    jobs_corpus : the list returned by fetch_jobs_corpus(), after
                  encode_jobs() has populated embeddings in-place.
                  Each dict must have "id" (str) and "embedding" (list[float]).
    """
    db    = get_db()
    batch = db.batch()
    count = 0

    for job in jobs_corpus:
        job_id    = job.get("id")
        embedding = job.get("embedding")

        # Skip if no id, still empty, or not a list
        if not job_id or not isinstance(embedding, list) or len(embedding) == 0:
            continue

        ref = db.collection(FIRESTORE_JOBS_COLLECTION).document(job_id)
        batch.update(ref, {"embedding": embedding})
        count += 1

        # Firestore batch write limit is 500 operations
        if count % 490 == 0:
            batch.commit()
            batch = db.batch()
            log.info(f"[firestore] Committed embedding batch ({count} so far).")

    if count > 0:
        batch.commit()

    log.info(f"[firestore] Cached embeddings for {count} job(s).")


# ══════════════════════════════════════════════════════════════════════════════
# WRITE — PERSIST RESUME ANALYSIS RESULT
# ══════════════════════════════════════════════════════════════════════════════

def save_resume_result(file_name: str, result: dict) -> str:
    """
    Persist a resume analysis result to the resume_results collection.

    Parameters
    ----------
    file_name : original filename of the uploaded resume
    result    : dict returned by the analyze_resume() pipeline

    Returns
    -------
    Firestore document ID of the saved result.
    """
    db = get_db()

    payload = {
        "file_name":    file_name,
        "analyzed_at":  datetime.now(tz=timezone.utc),
        "ocr_quality":  result.get("ocr_quality"),
        "skills":       result.get("skills",       []),
        "certificates": result.get("certificates", []),
        "validation":   result.get("validation"),
        "top_matches": [
            {
                "job_id":         m.get("job_id"),
                "title":          m.get("title"),
                "industry":       m.get("industry"),
                "score":          m.get("score"),
                "cosine_score":   m.get("cosine_score"),
                "match_ratio":    m.get("match_ratio"),
                "gap_count":      m.get("gap_count"),
                "matched_skills": m.get("matched_skills", []),
                "missing_skills": m.get("missing_skills", []),
            }
            for m in result.get("top_matches", [])
        ],
    }

    _, doc_ref = db.collection(FIRESTORE_RESULTS_COLLECTION).add(payload)
    log.info(
        f"[firestore] Result saved → "
        f"{FIRESTORE_RESULTS_COLLECTION}/{doc_ref.id}"
    )
    return doc_ref.id


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE — load everything needed for one analysis run
# ══════════════════════════════════════════════════════════════════════════════

def load_analyzer_data() -> tuple[list, list, list]:
    """
    Fetch the jobs corpus and derive the skills + certs allowlists in one call.

    Returns
    -------
    (firestore_skills, jobs_corpus, firestore_certs)

    firestore_skills : sorted list of every unique skill string found across
                       all job required_skills arrays — this is the allowlist
                       passed to extract_skills() in matcher.py
    jobs_corpus      : list of job dicts ready for faiss_search_and_score()
    firestore_certs  : sorted list of every unique cert string found across
                       all job required_certifications arrays — this is the
                       allowlist passed to extract_certificates() in matcher.py.
                       Returns [] when no job has required_certifications set
                       (extract_certificates fails open in that case).
    """
    jobs_corpus      = fetch_jobs_corpus()
    firestore_skills = derive_skills_from_corpus(jobs_corpus)
    firestore_certs  = derive_certs_from_corpus(jobs_corpus)
    return firestore_skills, jobs_corpus, firestore_certs