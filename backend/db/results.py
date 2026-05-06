"""
db/results.py
=============
Firestore operations for the resume_results collection.
"""

import logging
from datetime import datetime, timezone

from config import FIRESTORE_RESULTS_COLLECTION
from db.client import get_db

log = logging.getLogger(__name__)


def save_resume_result(file_name: str, result: dict) -> str:
    """Persist a resume analysis result. Returns the new document ID."""
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
    log.info(f"[firestore] Result saved → {FIRESTORE_RESULTS_COLLECTION}/{doc_ref.id}")
    return doc_ref.id