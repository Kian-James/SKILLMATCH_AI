"""
check_skills.py
───────────────
Prints all unique skills stored across all jobs in Firestore.
Usage: python check_skills.py
"""

import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIRESTORE_JOBS_COLLECTION  = os.getenv("FIRESTORE_JOBS_COLLECTION", "jobs")

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

docs = db.collection(FIRESTORE_JOBS_COLLECTION).stream()

all_skills = set()
job_count  = 0

for doc in docs:
    job = doc.to_dict()
    job_count += 1
    for skill in job.get("required_skills", []):
        all_skills.add(skill.strip())

print(f"\n✅ {job_count} jobs found in '{FIRESTORE_JOBS_COLLECTION}'")
print(f"✅ {len(all_skills)} unique skills total\n")
print("── All Skills ──────────────────────────────")
for skill in sorted(all_skills):
    print(f"  • {skill}")

# ── Check specific skills ─────────────────────────────────────────────────────
to_check = [
    "Zendesk",
    "AnyDesk",
    "SolarWinds Network",
    "Performance Monitor",
    "Malwarebytes",
    "macOS",
    "Ansible",
    "Microsoft Excel",
    "Microsoft Azure",
]

print("\n── Skill Check ─────────────────────────────")
for skill in to_check:
    found = skill in all_skills
    status = "✅ EXISTS" if found else "❌ MISSING"
    print(f"  {status} — {skill}")