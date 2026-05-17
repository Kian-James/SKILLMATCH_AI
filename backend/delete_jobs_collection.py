"""
delete_jobs_collection.py
──────────────────────────
Deletes ALL documents in the Firestore jobs collection.

Usage:
    python delete_jobs_collection.py

Requirements in .env:
    FIREBASE_CREDENTIALS_PATH  – path to your serviceAccountKey.json
    FIRESTORE_JOBS_COLLECTION  – Firestore collection name (default: jobs)
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIRESTORE_JOBS_COLLECTION = os.getenv("FIRESTORE_JOBS_COLLECTION", "jobs")
BATCH_SIZE = 100  # Firestore max batch write size

# ── Validate env ──────────────────────────────────────────────────────────────
if not FIREBASE_CREDENTIALS_PATH:
    print("[ERROR] FIREBASE_CREDENTIALS_PATH is not set.")
    sys.exit(1)
if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
    print(f"[ERROR] Credentials file not found: {FIREBASE_CREDENTIALS_PATH}")
    sys.exit(1)

# ── Firebase init ─────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ── Delete in batches ─────────────────────────────────────────────────────────

def delete_collection(collection_name: str) -> int:
    collection_ref = db.collection(collection_name)
    total_deleted  = 0

    while True:
        docs = list(collection_ref.limit(BATCH_SIZE).stream())
        if not docs:
            break

        batch = db.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()

        total_deleted += len(docs)
        print(f"  Deleted {total_deleted} document(s) so far …")

    return total_deleted


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[WARNING] This will permanently delete ALL documents in '{FIRESTORE_JOBS_COLLECTION}'.")
    confirm = input("  Type 'yes' to confirm: ").strip().lower()

    if confirm != "yes":
        print("[CANCELLED] Nothing was deleted.")
        sys.exit(0)

    print(f"\n[INFO] Deleting collection '{FIRESTORE_JOBS_COLLECTION}' …\n")
    total = delete_collection(FIRESTORE_JOBS_COLLECTION)

    print(f"\n[DONE] Deleted {total} document(s) from '{FIRESTORE_JOBS_COLLECTION}'.")