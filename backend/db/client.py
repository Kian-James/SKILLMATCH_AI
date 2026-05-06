"""
db/client.py
============
Firestore client singleton.
"""

import logging
import os

from config import FIREBASE_CREDENTIALS_PATH

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
        if not FIREBASE_CREDENTIALS_PATH:
            raise RuntimeError("FIREBASE_CREDENTIALS_PATH is not set in your .env file.")
        if not os.path.isfile(FIREBASE_CREDENTIALS_PATH):
            raise FileNotFoundError(f"Service account file not found: {FIREBASE_CREDENTIALS_PATH}")
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        log.info(f"[firestore] Initialised with: {FIREBASE_CREDENTIALS_PATH}")

    _db = firestore.client()
    log.info("[firestore] Client ready.")
    return _db