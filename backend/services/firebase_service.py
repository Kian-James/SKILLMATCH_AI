"""
services/firebase_service.py
All Firestore read/write operations. Resumes are NOT stored — only
extracted metadata (skills, certifications, analysis results).
"""

from __future__ import annotations

import os
import json
from typing import Any
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    get_firebase_credentials_path,
    COLLECTION_USERS,
    COLLECTION_JOB_DATASET,
    COLLECTION_USER_PROGRESS,
)

_db: firestore.Client | None = None


def init_firebase() -> None:
    """Initialize Firebase Admin SDK (idempotent)."""
    global _db
    if firebase_admin._apps:
        _db = firestore.client()
        return

    cred = credentials.Certificate(get_firebase_credentials_path())
    firebase_admin.initialize_app(cred)
    _db = firestore.client()
    logger.info("Firebase initialized successfully.")


def get_db() -> firestore.Client:
    if _db is None:
        raise RuntimeError("Firebase not initialized. Call init_firebase() first.")
    return _db


# ─── Job Dataset ──────────────────────────────────────────────────────────────

def fetch_all_jobs() -> list[dict]:
    """
    Return all job entries from Firestore.
    Each doc has: job_title, skills[], certifications[]
    """
    db = get_db()
    docs = db.collection(COLLECTION_JOB_DATASET).stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


def upsert_job(job_title: str, skills: list[str], certifications: list[str]) -> str:
    """
    Insert or update a job entry keyed by normalized title.
    Returns the document ID.
    """
    db = get_db()
    normalized_title = job_title.strip().title()
    doc_ref = db.collection(COLLECTION_JOB_DATASET).document(normalized_title)
    doc_ref.set({
        "job_title": normalized_title,
        "skills": sorted(set(skills)),
        "certifications": sorted(set(certifications)),
        "updated_at": datetime.utcnow().isoformat(),
    }, merge=True)
    return normalized_title


# ─── User Progress (Skill Growth Tracking) ────────────────────────────────────

def save_analysis_snapshot(
        user_id: str,
        skills: list[str],
        certifications: list[str],
        top_job_matches: list[dict],
        best_match_score: float,
) -> str:
    """
    Save a timestamped snapshot of extracted resume metadata.
    Does NOT store the resume file — only extracted data.
    Returns the new document ID.
    """
    db = get_db()
    doc_ref = db.collection(COLLECTION_USER_PROGRESS).document(user_id) \
        .collection("snapshots").document()

    doc_ref.set({
        "timestamp": datetime.utcnow().isoformat(),
        "skills": sorted(skills),
        "certifications": sorted(certifications),
        "skill_count": len(skills),
        "top_job_matches": top_job_matches[:5],   # store top 5 matches
        "best_match_score": round(best_match_score, 4),
    })
    logger.info(f"Snapshot saved for user {user_id}: {len(skills)} skills")
    return doc_ref.id


def get_user_snapshots(user_id: str) -> list[dict]:
    """Return all historical snapshots for a user, ordered by timestamp."""
    db = get_db()
    docs = (
        db.collection(COLLECTION_USER_PROGRESS)
        .document(user_id)
        .collection("snapshots")
        .order_by("timestamp")
        .stream()
    )
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


# ─── User Profile ─────────────────────────────────────────────────────────────

def get_or_create_user(user_id: str, display_name: str = "", email: str = "") -> dict:
    """Ensure user document exists in Firestore."""
    db = get_db()
    doc_ref = db.collection(COLLECTION_USERS).document(user_id)
    doc = doc_ref.get()
    if not doc.exists:
        data = {
            "user_id": user_id,
            "display_name": display_name,
            "email": email,
            "created_at": datetime.utcnow().isoformat(),
        }
        doc_ref.set(data)
        return data
    return doc.to_dict()