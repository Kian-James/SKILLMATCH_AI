"""
generate_jobs_from_labels.py
─────────────────────────────
Reads job data from labels.json, generates ONLY the description for each job
via Llama 3.1 8B (Cerebras / HF Inference Provider), encodes each job with
SBERT (all-MiniLM-L6-v2), and stores everything in Firestore.

Source of truth (from labels.json):
  - title                  ← expected_job_title
  - required_skills        ← expected_skills
  - required_certifications← expected_certs (defaults to [] if missing)
  - industry               ← inferred from filename prefix
                             "Eng-" → Engineering, "Com-" → Computing

LLM is used ONLY to generate a short 1–2 sentence description per job.

Merge strategy (same as original generate_jobs.py):
  - If a job title already exists in Firestore:
      • required_skills are merged (union, cap 10)
      • required_certifications are merged (union, cap 5)
      • description is refreshed
      • embedding is re-generated from merged data

Usage:
    python generate_jobs_from_labels.py

Requirements in .env:
    HF_TOKEN                   – Hugging Face API token
    FIREBASE_CREDENTIALS_PATH  – path to your serviceAccountKey.json
    FIRESTORE_JOBS_COLLECTION  – Firestore collection name (default: jobs)
    LABELS_PATH                – path to labels.json (default: labels.json)
"""

import os
import sys
import json
import re
import time
import requests
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

HF_TOKEN                  = os.getenv("HF_TOKEN")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIRESTORE_JOBS_COLLECTION = os.getenv("FIRESTORE_JOBS_COLLECTION", "jobs")
LABELS_PATH               = os.getenv("LABELS_PATH", "labels.json")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BATCH_SIZE  = int(os.getenv("DESCRIPTION_BATCH_SIZE", "10"))  # jobs per LLM call

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"


# ── Validate env ──────────────────────────────────────────────────────────────
errors = []
if not HF_TOKEN:
    errors.append("HF_TOKEN is not set.")
if not FIREBASE_CREDENTIALS_PATH:
    errors.append("FIREBASE_CREDENTIALS_PATH is not set.")
elif not os.path.exists(FIREBASE_CREDENTIALS_PATH):
    errors.append(f"Credentials file not found: {FIREBASE_CREDENTIALS_PATH}")
if not os.path.exists(LABELS_PATH):
    errors.append(f"labels.json not found: {LABELS_PATH}")

if errors:
    for e in errors:
        print(f"[ERROR] {e}")
    sys.exit(1)

# ── SBERT model ───────────────────────────────────────────────────────────────
print("[INFO] Loading SBERT model (sentence-transformers/all-MiniLM-L6-v2) …")
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
print(f"[INFO] SBERT ready on {device.upper()}.\n")

# ── Firebase init ─────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ── Industry inference ────────────────────────────────────────────────────────

def infer_industry(filename: str) -> str:
    """
    Infer industry from the filename prefix.
      'Eng-' or 'Eng - ' → Engineering
      'Com-' or 'Com - ' → Computing
    Defaults to 'Computing' if unrecognised.
    """
    name = filename.strip().lower()
    if name.startswith("eng"):
        return "Engineering"
    if name.startswith("com"):
        return "Computing"
    print(f"  [WARN] Cannot infer industry from '{filename}' — defaulting to Computing.")
    return "Computing"


# ── Load & normalise labels ───────────────────────────────────────────────────

def _dedup(items: list, cap: int = None) -> list:
    """Case-insensitive dedup preserving first-seen order. Cap is optional."""
    seen_keys: set  = set()
    result:    list = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen_keys:
            seen_keys.add(key)
            result.append(item.strip())
        if cap is not None and len(result) >= cap:
            break
    return result


def load_labels(path: str) -> list[dict]:
    """
    Read labels.json and return a flat list of normalised job dicts:
      {
        title, industry, required_skills, required_certifications,
        source_file
      }
    Entries with an empty title are skipped.

    Deduplication strategy:
      - ALL skills and certs from every resume that shares the same title are
        collected into a raw pool first (no cap applied during collection).
      - Only after all files are processed is the pool deduped (case-insensitive)
        and capped — so no skill is ever lost due to insertion order or file order.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: dict = json.load(f)

    # title_key → {title, industry, source_file, _all_skills[], _all_certs[]}
    seen: dict[str, dict] = {}

    for filename, meta in raw.items():
        title = (meta.get("expected_job_title") or "").strip()
        if not title:
            print(f"  [SKIP] '{filename}' has no expected_job_title — skipping.")
            continue

        skills    = [str(s).strip() for s in (meta.get("expected_skills") or []) if str(s).strip()]
        certs     = [str(c).strip() for c in (meta.get("expected_certs")  or []) if str(c).strip()]
        industry  = infer_industry(filename)
        title_key = title.lower()

        if title_key not in seen:
            seen[title_key] = {
                "title":       title,
                "industry":    industry,
                "source_file": filename,
                "_all_skills": list(skills),   # raw pool — no cap yet
                "_all_certs":  list(certs),
            }
        else:
            # Append ALL skills/certs from this file into the pool — cap happens later
            seen[title_key]["_all_skills"].extend(skills)
            seen[title_key]["_all_certs"].extend(certs)

    # ── Dedup each job's full pool — NO cap on skills, cap=5 on certs ────────
    jobs = []
    for title_key, entry in seen.items():
        required_skills         = _dedup(entry["_all_skills"])        # no cap — store ALL unique skills
        required_certifications = _dedup(entry["_all_certs"], cap=5)

        print(f"  [INFO] '{entry['title']}': {len(required_skills)} unique skills collected")

        jobs.append({
            "title":                   entry["title"],
            "industry":                entry["industry"],
            "required_skills":         required_skills,
            "required_certifications": required_certifications,
            "source_file":             entry["source_file"],
        })

    print(f"[INFO] Loaded {len(raw)} label entries → {len(jobs)} unique job titles.\n")
    return jobs


# ── List merge helpers ────────────────────────────────────────────────────────

def merge_skills(priority: list, fallback: list) -> list:
    """Labels go first; old Firestore skills appended. No cap — store all unique skills."""
    return _dedup(priority + fallback)


def merge_certifications(priority: list, fallback: list) -> list:
    """Labels go first; old Firestore certs fill remaining slots up to cap=5."""
    return _dedup(priority + fallback, cap=5)


# ── Firestore helpers ─────────────────────────────────────────────────────────

def title_to_doc_id(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def get_existing_docs() -> dict:
    docs = db.collection(FIRESTORE_JOBS_COLLECTION).stream()
    return {doc.id: doc.to_dict() for doc in docs}


# ── LLM — description generation ─────────────────────────────────────────────

SYSTEM_PROMPT_DESC = """You are a job listing writer. Output ONLY a valid JSON array — no markdown, no code fences, no explanation.

Each element must match this schema exactly:
{"title": "...", "description": "..."}

Hard rules:
- "description" must be exactly 1–2 short sentences summarising what the role does and its impact.
- No special characters, no newlines inside the string.
- No trailing commas. No extra fields. Pure JSON only."""


def build_description_prompt(jobs: list[dict]) -> str:
    lines = "\n".join(
        f'{i+1}. Title: "{j["title"]}" | Industry: {j["industry"]} '
        f'| Skills: {", ".join(j["required_skills"][:5])}'
        for i, j in enumerate(jobs)
    )
    return (
        f"Write a 1–2 sentence job description for each of the following {len(jobs)} roles.\n\n"
        f"{lines}\n\n"
        f"Return ONLY a JSON array with exactly {len(jobs)} objects, "
        f"each with 'title' and 'description'. No other text."
    )


def call_llama(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_DESC},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 1800,
        "temperature": 0.7,
        "stream": False,
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"API returned {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


# ── Description parser ────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> list:
    objects, depth, start = [], 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(raw[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def parse_descriptions(raw: str, jobs: list[dict]) -> dict[str, str]:
    """
    Parse LLM output and return a dict of {title_lower: description}.
    Falls back to a generic description if a title is not found in the output.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    start = cleaned.find("[")
    end   = cleaned.rfind("]") + 1

    if start != -1 and end > start:
        try:
            items = json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            items = _repair_truncated_json(cleaned)
    else:
        items = _repair_truncated_json(cleaned)

    desc_map: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        t = (item.get("title") or "").strip().lower()
        d = (item.get("description") or "").replace("\n", " ").replace("\r", "").strip()
        if t and d:
            desc_map[t] = d

    # Fill any missing titles with a safe fallback
    for job in jobs:
        key = job["title"].lower()
        if key not in desc_map:
            print(f"  [WARN] No description returned for '{job['title']}' — using fallback.")
            desc_map[key] = (
                f"A {job['industry']} professional specialising in "
                f"{', '.join(job['required_skills'][:3])}."
            )

    return desc_map


# ── Description generation (batched) ─────────────────────────────────────────

def generate_descriptions(jobs: list[dict]) -> dict[str, str]:
    """
    Call the LLM in batches of BATCH_SIZE and return a {title_lower: description} map.
    """
    desc_map: dict[str, str] = {}
    total   = len(jobs)
    batches = [jobs[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    print(f"[INFO] Generating descriptions for {total} jobs in {len(batches)} batch(es) …\n")

    for idx, batch in enumerate(batches, 1):
        print(f"  [DESC BATCH {idx:02d}/{len(batches)}] {len(batch)} job(s) …")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt = build_description_prompt(batch)
                raw    = call_llama(prompt)
                result = parse_descriptions(raw, batch)
                desc_map.update(result)
                break
            except (ValueError, RuntimeError) as exc:
                print(f"    [RETRY {attempt}/{MAX_RETRIES}] Error: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    print(f"    [SKIP] Batch {idx} failed — using fallback descriptions.")
                    for job in batch:
                        key = job["title"].lower()
                        if key not in desc_map:
                            desc_map[key] = (
                                f"A {job['industry']} professional specialising in "
                                f"{', '.join(job['required_skills'][:3])}."
                            )

        time.sleep(1)   # polite rate-limit pause

    return desc_map


# ── Embedding ─────────────────────────────────────────────────────────────────

def _build_profile(job: dict) -> str:
    return (
        f"Job: {job['title']}. Industry: {job['industry']}. "
        f"Core Skills: {' '.join(job['required_skills'])}. "
        f"Certifications: {' '.join(job.get('required_certifications', []))}. "
        f"Desc: {job.get('description', '')}"
    )


def embed_jobs(jobs: list[dict]) -> list[dict]:
    profiles = [_build_profile(j) for j in jobs]
    print(f"\n[EMBED] Encoding {len(jobs)} job(s) with SBERT …")
    vecs = sbert.encode(profiles, convert_to_numpy=True, show_progress_bar=True)
    for job, vec in zip(jobs, vecs):
        job["embedding"] = vec.astype(float).tolist()
    return jobs


# ── Seeder ────────────────────────────────────────────────────────────────────

def seed_jobs(jobs: list[dict], existing_docs: dict) -> tuple[int, int]:
    """
    Write each job to Firestore.
      NEW     → full document set
      EXISTING→ labels are the authority:
                  • label skills/certs go in FIRST (they are the source of truth)
                  • any extra skills/certs already in Firestore fill remaining slots
                  • description and embedding are always refreshed

    Returns (new_count, merged_count).
    """
    collection_ref = db.collection(FIRESTORE_JOBS_COLLECTION)
    new_count    = 0
    merged_count = 0

    for job in jobs:
        doc_id   = title_to_doc_id(job["title"])
        existing = existing_docs.get(doc_id)

        if existing is None:
            # Strip internal helper key before storing
            doc = {k: v for k, v in job.items() if k != "source_file"}
            collection_ref.document(doc_id).set(doc)
            existing_docs[doc_id] = doc
            new_count += 1
            cert_note = f"  certs={len(job['required_certifications'])}" if job["required_certifications"] else ""
            print(f"  ✓ NEW     [{job['industry']}] {job['title']}{cert_note}")

        else:
            old_skills = existing.get("required_skills", [])
            old_certs  = existing.get("required_certifications", [])

            # Labels go FIRST — they are the source of truth.
            # Old Firestore skills/certs fill any remaining slots up to the cap.
            merged_skills = merge_skills(job["required_skills"], old_skills)
            merged_certs  = merge_certifications(job["required_certifications"], old_certs)

            merged_job = {
                **existing,
                "title":                   job["title"],
                "required_skills":         merged_skills,
                "required_certifications": merged_certs,
                "description":             job["description"],
            }

            # Re-encode with merged data
            vec = sbert.encode(
                [_build_profile(merged_job)],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            merged_job["embedding"] = vec.astype(float).tolist()

            doc = {k: v for k, v in merged_job.items() if k != "source_file"}
            collection_ref.document(doc_id).set(doc)
            existing_docs[doc_id] = doc
            merged_count += 1

            skill_delta = len(merged_skills) - len(old_skills)
            cert_delta  = len(merged_certs)  - len(old_certs)
            print(
                f"  ↻ MERGED  [{job['industry']}] {job['title']}  "
                f"(skills: {len(old_skills)}→{len(merged_skills)} +{skill_delta}, "
                f"certs: {len(old_certs)}→{len(merged_certs)} +{cert_delta})"
            )

    return new_count, merged_count


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[INFO] Reading labels from : {LABELS_PATH}")
    print(f"[INFO] LLM                 : {HF_MODEL}")
    print(f"[INFO] Embed               : sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Store               : Firestore / {FIRESTORE_JOBS_COLLECTION}\n")

    # 1. Load & normalise from labels.json
    jobs = load_labels(LABELS_PATH)

    # 2. Generate descriptions via LLM (batched)
    desc_map = generate_descriptions(jobs)

    # Attach descriptions to job dicts
    for job in jobs:
        job["description"] = desc_map.get(job["title"].lower(), "")

    # 3. Embed all jobs
    jobs = embed_jobs(jobs)

    # 4. Load existing Firestore docs and seed
    print("\n[INFO] Checking existing Firestore jobs …")
    existing_docs = get_existing_docs()
    print(f"[INFO] {len(existing_docs)} existing jobs found.\n")

    print("[INFO] Writing to Firestore …\n")
    new_count, merged_count = seed_jobs(jobs, existing_docs)

    # ── Summary ───────────────────────────────────────────────────────────────
    eng_count = sum(1 for j in jobs if j["industry"] == "Engineering")
    com_count = sum(1 for j in jobs if j["industry"] == "Computing")

    print(f"\n{'═'*60}")
    print(f"[DONE]  Labels read             : {len(jobs)}")
    print(f"[DONE]  New documents           : {new_count}")
    print(f"[DONE]  Merged documents        : {merged_count}")
    print(f"[DONE]  Engineering jobs        : {eng_count}")
    print(f"[DONE]  Computing jobs          : {com_count}")
    print(f"[DONE]  Total in Firestore now  : {len(existing_docs)}")
    print(f"[DONE]  Collection              : {FIRESTORE_JOBS_COLLECTION}")
    print(f"[DONE]  Model                   : {HF_MODEL}")