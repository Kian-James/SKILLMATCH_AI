"""
generate_jobs_from_labels.py
────────────────────────────
Reads job data from labels.json, generates a short description for each job
via Zephyr-7B-Beta (HF Inference / featherless-ai), encodes each job with
SBERT (all-MiniLM-L6-v2), and writes everything to Firestore.

Source of truth (labels.json):
  title                   ← expected_job_title
  required_skills         ← expected_skills
  required_certifications ← expected_certs (defaults to [] if absent)
  industry                ← inferred from filename prefix
                            "Eng" → Engineering | "Com" → Computing

LLM is used ONLY for the 1–2 sentence description field.

Merge strategy (when a title already exists in Firestore):
  • required_skills         – union of label + stored (label rows take priority)
  • required_certifications – union, capped at MAX_CERTS
  • description             – always refreshed from LLM
  • embedding               – re-generated from merged data

Usage:
    python generate_jobs_from_labels.py

Required .env variables:
    HF_TOKEN                   – Hugging Face API token
    FIREBASE_CREDENTIALS_PATH  – path to serviceAccountKey.json
    FIRESTORE_JOBS_COLLECTION  – Firestore collection name (default: jobs)
    LABELS_PATH                – path to labels.json  (default: labels.json)

Optional .env variables:
    MAX_RETRIES                – LLM call retries per batch  (default: 3)
    DESCRIPTION_BATCH_SIZE     – jobs per LLM call           (default: 4)
    MAX_CERTS                  – cap on stored certifications (default: 5)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional

import requests
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()

HF_TOKEN                  = os.getenv("HF_TOKEN", "")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
FIRESTORE_JOBS_COLLECTION = os.getenv("FIRESTORE_JOBS_COLLECTION", "jobs")
LABELS_PATH               = os.getenv("LABELS_PATH", "labels.json")
MAX_RETRIES               = int(os.getenv("MAX_RETRIES", "3"))
BATCH_SIZE                = int(os.getenv("DESCRIPTION_BATCH_SIZE", "4"))
MAX_CERTS                 = int(os.getenv("MAX_CERTS", "5"))

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "HuggingFaceH4/zephyr-7b-beta:featherless-ai"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _abort_if_invalid() -> None:
    errors: list[str] = []
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

_abort_if_invalid()

# ── SBERT ─────────────────────────────────────────────────────────────────────

print("[INFO] Loading SBERT model …")
_device = "cuda" if torch.cuda.is_available() else "cpu"
sbert   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=_device)
print(f"[INFO] SBERT ready on {_device.upper()}.\n")

# ── Firestore ─────────────────────────────────────────────────────────────────

import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dedup(items: list[str], cap: Optional[int] = None) -> list[str]:
    """Case-insensitive dedup preserving first-seen order, with an optional cap."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
        if cap is not None and len(result) >= cap:
            break
    return result


def _infer_industry(filename: str) -> str:
    name = filename.strip().lower()
    if name.startswith("eng"):
        return "Engineering"
    if name.startswith("com"):
        return "Computing"
    print(f"  [WARN] Cannot infer industry from '{filename}' – defaulting to Computing.")
    return "Computing"


def _title_to_doc_id(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & normalise labels.json
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(path: str) -> list[dict]:
    """
    Parse labels.json and return one dict per unique job title:
        {title, industry, required_skills, required_certifications, source_file}

    All resumes that share a title contribute their skills/certs to a common
    pool before dedup/cap is applied, so no data is lost due to file ordering.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    # title_lower → aggregation bucket
    buckets: dict[str, dict] = {}

    for filename, meta in raw.items():
        title = (meta.get("expected_job_title") or "").strip()
        if not title:
            print(f"  [SKIP] '{filename}' has no expected_job_title.")
            continue

        skills   = [str(s).strip() for s in (meta.get("expected_skills") or []) if str(s).strip()]
        certs    = [str(c).strip() for c in (meta.get("expected_certs")  or []) if str(c).strip()]
        industry = _infer_industry(filename)
        key      = title.lower()

        if key not in buckets:
            buckets[key] = {
                "title":       title,
                "industry":    industry,
                "source_file": filename,
                "_skills":     list(skills),
                "_certs":      list(certs),
            }
        else:
            buckets[key]["_skills"].extend(skills)
            buckets[key]["_certs"].extend(certs)

    jobs: list[dict] = []
    for bucket in buckets.values():
        skills_deduped = _dedup(bucket["_skills"])          # no cap on skills
        certs_deduped  = _dedup(bucket["_certs"], cap=MAX_CERTS)
        print(f"  [LABELS] '{bucket['title']}': {len(skills_deduped)} unique skill(s), "
              f"{len(certs_deduped)} cert(s)")
        jobs.append({
            "title":                   bucket["title"],
            "industry":                bucket["industry"],
            "required_skills":         skills_deduped,
            "required_certifications": certs_deduped,
            "source_file":             bucket["source_file"],
        })

    print(f"\n[INFO] Loaded {len(raw)} label entries → {len(jobs)} unique job title(s).\n")
    return jobs

# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM – description generation
# ─────────────────────────────────────────────────────────────────────────────

# WHY ONE JOB PER CALL?
# Zephyr-7B-Beta through the featherless-ai router reliably returns one clean
# JSON object when asked for exactly one item.  Asking for a batch of N items
# causes the model to truncate mid-array, skip titles, or echo the prompt.
# The per-call overhead is tolerable given the polite sleep between requests.
#
# WHY NO ASSISTANT SEED?
# Seeding `{"role": "assistant", "content": "["}` is supposed to prime JSON
# output, but the featherless-ai router sometimes ignores or strips it, leading
# to the seed text being embedded in the response body.  Instead we instruct
# the model explicitly in the system prompt and validate the output ourselves.

_SYSTEM_PROMPT = (
    "You are a concise technical writer. "
    "When given a job title, industry, and a list of required skills, "
    "you respond with ONLY a single valid JSON object and nothing else. "
    "The object must have exactly two keys: \"title\" and \"description\". "
    "\"description\" must be 1–2 sentences summarising what a professional "
    "in that role does. No markdown, no code fences, no extra text."
)


def _build_single_prompt(job: dict) -> str:
    skills_preview = ", ".join(job["required_skills"][:6]) or "N/A"
    return (
        f"Title: {job['title']}\n"
        f"Industry: {job['industry']}\n"
        f"Key skills: {skills_preview}\n\n"
        f"Return a JSON object with keys \"title\" and \"description\"."
    )


def _call_zephyr(user_content: str) -> str:
    """Single HTTP call to the HF chat-completions endpoint."""
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": 200,
        "temperature": 0.2,
        "stream": False,
    }
    resp = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def _extract_json_object(text: str) -> dict:
    """
    Pull the first {...} block out of `text` and parse it.
    Raises ValueError if nothing parseable is found.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().strip("`").strip()

    # Find the outermost {...}
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in: {text!r}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON parse error: {exc} – raw: {candidate!r}") from exc

    raise ValueError(f"Unbalanced braces in: {text!r}")


def _generate_description_for_job(job: dict) -> str:
    """
    Call Zephyr once for a single job and return the description string.
    Retries up to MAX_RETRIES times with exponential back-off.
    Falls back to a template string if all retries are exhausted.
    """
    prompt = _build_single_prompt(job)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw  = _call_zephyr(prompt)
            print(f"    [LLM RAW] {raw[:160]}{'…' if len(raw) > 160 else ''}")
            obj  = _extract_json_object(raw)
            desc = (obj.get("description") or "").replace("\n", " ").strip()
            if not desc:
                raise ValueError("Empty 'description' field in LLM response.")
            return desc
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"    [RETRY {attempt}/{MAX_RETRIES}] {job['title']!r}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    # Graceful fallback — never stores an empty description
    fallback = (
        f"A {job['industry']} professional specialising in "
        f"{', '.join(job['required_skills'][:3]) or 'core domain skills'}."
    )
    print(f"  [FALLBACK] '{job['title']}' – using template description.")
    return fallback


def generate_descriptions(jobs: list[dict]) -> dict[str, str]:
    """
    Iterate over every job and return {title_lower: description}.
    A small sleep between calls respects the HF rate limit.
    """
    total    = len(jobs)
    desc_map: dict[str, str] = {}

    print(f"[INFO] Generating descriptions for {total} job(s) …\n")

    for idx, job in enumerate(jobs, 1):
        print(f"  [{idx:03d}/{total}] {job['title']} …")
        desc_map[job["title"].lower()] = _generate_description_for_job(job)
        time.sleep(0.8)   # polite pause between requests

    print()
    return desc_map

# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding
# ─────────────────────────────────────────────────────────────────────────────

def _build_profile(job: dict) -> str:
    """
    Produce the text string that SBERT encodes.
    Skills and certs are joined with ', ' so the tokeniser sees them as
    distinct phrases rather than one fused token.
    """
    skills = ", ".join(job["required_skills"])
    certs  = ", ".join(job.get("required_certifications", []))
    return (
        f"Job title: {job['title']}. "
        f"Industry: {job['industry']}. "
        f"Required skills: {skills}. "
        f"Certifications: {certs}. "
        f"Description: {job.get('description', '')}"
    )


def embed_jobs(jobs: list[dict]) -> list[dict]:
    profiles = [_build_profile(j) for j in jobs]
    print(f"[EMBED] Encoding {len(jobs)} job(s) with SBERT …")
    vecs = sbert.encode(profiles, convert_to_numpy=True, show_progress_bar=True)
    for job, vec in zip(jobs, vecs):
        job["embedding"] = vec.astype(float).tolist()
    return jobs

# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_skills(label_skills: list[str], stored_skills: list[str]) -> list[str]:
    """Label-derived skills go first; stored fills the remainder. No cap."""
    return _dedup(label_skills + stored_skills)


def _merge_certs(label_certs: list[str], stored_certs: list[str]) -> list[str]:
    """Label-derived certs go first; stored fills remainder up to MAX_CERTS."""
    return _dedup(label_certs + stored_certs, cap=MAX_CERTS)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Firestore seed
# ─────────────────────────────────────────────────────────────────────────────

def get_existing_docs() -> dict[str, dict]:
    return {doc.id: doc.to_dict() for doc in db.collection(FIRESTORE_JOBS_COLLECTION).stream()}


def seed_jobs(jobs: list[dict], existing_docs: dict[str, dict]) -> tuple[int, int]:
    """
    Write every job to Firestore.
    Returns (new_count, merged_count).
    """
    col   = db.collection(FIRESTORE_JOBS_COLLECTION)
    new_c = merged_c = 0

    for job in jobs:
        doc_id   = _title_to_doc_id(job["title"])
        existing = existing_docs.get(doc_id)

        if existing is None:
            # New document – write as-is (drop the internal source_file key)
            doc = {k: v for k, v in job.items() if k != "source_file"}
            col.document(doc_id).set(doc)
            existing_docs[doc_id] = doc
            new_c += 1
            print(f"  ✓ NEW     [{job['industry']}] {job['title']}")

        else:
            # Existing document – merge skills/certs, refresh description + embedding
            merged_skills = _merge_skills(
                job["required_skills"],
                existing.get("required_skills", []),
            )
            merged_certs = _merge_certs(
                job["required_certifications"],
                existing.get("required_certifications", []),
            )

            merged_job: dict = {
                **existing,
                "title":                   job["title"],
                "industry":                job["industry"],
                "required_skills":         merged_skills,
                "required_certifications": merged_certs,
                "description":             job["description"],
            }

            # Re-embed with the merged profile
            vec = sbert.encode(
                [_build_profile(merged_job)],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            merged_job["embedding"] = vec.astype(float).tolist()

            doc = {k: v for k, v in merged_job.items() if k != "source_file"}
            col.document(doc_id).set(doc)
            existing_docs[doc_id] = doc
            merged_c += 1

            old_s = len(existing.get("required_skills", []))
            old_c = len(existing.get("required_certifications", []))
            print(
                f"  ↻ MERGED  [{job['industry']}] {job['title']}  "
                f"(skills {old_s}→{len(merged_skills)}, "
                f"certs {old_c}→{len(merged_certs)})"
            )

    return new_c, merged_c

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[INFO] Labels file : {LABELS_PATH}")
    print(f"[INFO] LLM model   : {HF_MODEL}")
    print(f"[INFO] Embed model : sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Firestore   : {FIRESTORE_JOBS_COLLECTION}")
    print(f"[INFO] Batch size  : 1 job per LLM call (most reliable for Zephyr-7B)")
    print(f"[INFO] Max retries : {MAX_RETRIES}\n")

    # Step 1 – load labels
    jobs = load_labels(LABELS_PATH)

    # Step 2 – generate descriptions (one LLM call per job)
    desc_map = generate_descriptions(jobs)
    for job in jobs:
        job["description"] = desc_map.get(job["title"].lower(), "")

    # Step 3 – embed
    jobs = embed_jobs(jobs)

    # Step 4 – write to Firestore
    print("\n[INFO] Fetching existing Firestore documents …")
    existing_docs = get_existing_docs()
    print(f"[INFO] {len(existing_docs)} existing document(s) found.\n")

    print("[INFO] Writing to Firestore …\n")
    new_count, merged_count = seed_jobs(jobs, existing_docs)

    # Summary
    eng = sum(1 for j in jobs if j["industry"] == "Engineering")
    com = sum(1 for j in jobs if j["industry"] == "Computing")
    print(f"\n{'═' * 55}")
    print(f"  Labels processed   : {len(jobs)}")
    print(f"  New documents      : {new_count}")
    print(f"  Merged documents   : {merged_count}")
    print(f"  Engineering jobs   : {eng}")
    print(f"  Computing jobs     : {com}")
    print(f"  Total in Firestore : {len(existing_docs)}")
    print(f"  Collection         : {FIRESTORE_JOBS_COLLECTION}")
    print(f"{'═' * 55}")