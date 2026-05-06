"""
db/jobs.py
==========
Firestore operations for the jobs collection.
"""

import logging

from config import FIRESTORE_JOBS_COLLECTION
from core.utils import normalize
from db.client import get_db

log = logging.getLogger(__name__)


def fetch_jobs_corpus() -> list[dict]:
    """Fetch all job documents from Firestore, sorted by document ID."""
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
            "embedding":               data.get("embedding") or [],
        })

    if not corpus:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No jobs found in Firestore collection '{FIRESTORE_JOBS_COLLECTION}'. "
                "Run generate_jobs.py first to seed the collection."
            ),
        )

    corpus.sort(key=lambda j: j.get("id", 0))
    log.info(f"[firestore] Fetched {len(corpus)} jobs from '{FIRESTORE_JOBS_COLLECTION}'.")
    return corpus


def derive_skills_from_corpus(jobs_corpus: list[dict]) -> list[str]:
    """Aggregate unique required_skills across all job documents."""
    all_skills: set[str] = set()
    for job in jobs_corpus:
        for skill in job.get("required_skills", []):
            if s := skill.strip():
                all_skills.add(s)
    skill_list = sorted(all_skills)
    log.info(f"[firestore] Derived {len(skill_list)} unique skills from corpus.")
    return skill_list


def derive_certs_from_corpus(jobs_corpus: list[dict]) -> list[str]:
    """Aggregate unique required_certifications across all job documents."""
    all_certs: set[str] = set()
    for job in jobs_corpus:
        for cert in job.get("required_certifications", []):
            if c := cert.strip():
                all_certs.add(c)
    cert_list = sorted(all_certs)
    log.info(f"[firestore] Derived {len(cert_list)} unique certs from corpus.")
    return cert_list


def cache_job_embeddings(jobs_corpus: list[dict]) -> None:
    """Write SBERT embeddings back to Firestore for jobs that lacked one."""
    db    = get_db()
    batch = db.batch()
    count = 0

    for job in jobs_corpus:
        job_id    = job.get("id")
        embedding = job.get("embedding")
        if not job_id or not isinstance(embedding, list) or len(embedding) == 0:
            continue

        ref = db.collection(FIRESTORE_JOBS_COLLECTION).document(job_id)
        batch.update(ref, {"embedding": embedding})
        count += 1

        if count % 490 == 0:
            batch.commit()
            batch = db.batch()
            log.info(f"[firestore] Committed embedding batch ({count} so far).")

    if count > 0:
        batch.commit()

    log.info(f"[firestore] Cached embeddings for {count} job(s).")