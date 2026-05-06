"""
db/client.py
============
Firestore client singleton.
"""

import logging
import os

from config import FIREBASE_CREDENTIALS_PATH, FIREBASE_CREDENTIALS_JSON

log = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    log.warning("[firestore] firebase-admin not installed. Run: pip install firebase-admin")

_db = None


def get_db():
    """Return Firestore client, initialising the Firebase app on first call."""
    global _db
    if _db is not None:
        return _db

    if not _FIREBASE_AVAILABLE:
        raise RuntimeError("firebase-admin is not installed. Run: pip install firebase-admin")

    if not firebase_admin._apps:
        if FIREBASE_CREDENTIALS_JSON:
            # Prefer inline JSON string — set FIREBASE_CREDENTIALS_JSON in Render
            # env vars as a single-line JSON, no secret file needed.
            import json
            cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS_JSON))
            firebase_admin.initialize_app(cred)
            log.info("[firestore] Initialised from FIREBASE_CREDENTIALS_JSON env var.")
        elif FIREBASE_CREDENTIALS_PATH and os.path.isfile(FIREBASE_CREDENTIALS_PATH):
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
            log.info(f"[firestore] Initialised from file: {FIREBASE_CREDENTIALS_PATH}")
        else:
            raise RuntimeError(
                "Firebase credentials not found. Set either:\n"
                "  FIREBASE_CREDENTIALS_JSON — single-line JSON string of serviceAccountKey\n"
                "  FIREBASE_CREDENTIALS_PATH — path to serviceAccountKey.json file"
            )

    _db = firestore.client()
    log.info("[firestore] Client ready.")
    return _db