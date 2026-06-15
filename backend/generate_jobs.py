"""
generate_jobs.py
────────────────
Generates job listings via Zephyr-7B-Beta (HF Inference / featherless-ai),
encodes each job with SBERT (all-MiniLM-L6-v2), and writes everything to
Firestore.

Batch strategy:
  - JOB_COUNT jobs total, BATCH_SIZE per LLM call
  - Batches alternate industry: Computing → Engineering → Computing → …
  - If a title already exists in Firestore its required_skills and
    required_certifications are MERGED (LLM output takes priority over stored
    data), and the embedding is re-generated from the merged profile.

Uniqueness strategy:
  1. Avoid-list sent to the LLM contains only the most recent 60 titles so the
     prompt never overflows Zephyr's context window.
  2. Fuzzy duplicate detection via Jaccard token overlap (threshold 0.75).
  3. Titles must contain at least one tech/stack keyword.

Usage:
    python generate_jobs.py

Required .env variables:
    HF_TOKEN                   – Hugging Face API token
    FIREBASE_CREDENTIALS_PATH  – path to serviceAccountKey.json
    FIRESTORE_JOBS_COLLECTION  – Firestore collection name (default: jobs)

Optional .env variables:
    MAX_RETRIES                – retries per batch            (default: 3)
    JOB_COUNT                  – total jobs to generate       (default: 100)
    BATCH_SIZE                 – jobs per LLM call            (default: 10)
    AVOID_LIST_LIMIT           – max titles in avoid block    (default: 60)
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
MAX_RETRIES               = int(os.getenv("MAX_RETRIES", "3"))
JOB_COUNT                 = int(os.getenv("JOB_COUNT", "100"))
BATCH_SIZE                = int(os.getenv("BATCH_SIZE", "10"))
AVOID_LIST_LIMIT          = int(os.getenv("AVOID_LIST_LIMIT", "60"))

HF_API_URL     = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL       = "HuggingFaceH4/zephyr-7b-beta:featherless-ai"
INDUSTRY_ORDER = ["Computing", "Engineering"]

# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _abort_if_invalid() -> None:
    errors: list[str] = []
    if not HF_TOKEN:
        errors.append("HF_TOKEN is not set.")
    if not FIREBASE_CREDENTIALS_PATH:
        errors.append("FIREBASE_CREDENTIALS_PATH is not set.")
    elif not os.path.exists(FIREBASE_CREDENTIALS_PATH):
        errors.append(f"Credentials file not found: {FIREBASE_CREDENTIALS_PATH}")
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

# ── Style / cert examples ─────────────────────────────────────────────────────

STYLE_EXAMPLES: dict[str, list[str]] = {
    "Engineering": [
        "Embedded C Firmware Engineer",
        "FPGA Design Engineer",
        "PLC Control Systems Engineer",
        "SolidWorks Mechanical Design Engineer",
        "ROS2 Robotics Software Engineer",
        "5G RF Network Engineer",
        "CFD Fluid Dynamics Engineer",
        "Biomedical Implant Design Engineer",
        "SCADA Automation Engineer",
        "Geotechnical Foundation Engineer",
    ],
    "Computing": [
        "Python Backend Software Engineer",
        "React TypeScript Frontend Engineer",
        "PyTorch Machine Learning Engineer",
        "Kubernetes Platform Engineer",
        "Kafka Streaming Data Engineer",
        "Rust Systems Software Engineer",
        "iOS Swift Mobile Developer",
        "Penetration Testing Security Engineer",
        "dbt Analytics Engineer",
        "Solidity Smart Contract Developer",
    ],
}

CERT_EXAMPLES: dict[str, list[str]] = {
    "Computing": [
        "AWS Certified Solutions Architect – Associate",
        "Google Professional Cloud Architect",
        "Certified Kubernetes Administrator (CKA)",
        "HashiCorp Certified: Terraform Associate",
        "Certified Ethical Hacker (CEH)",
        "CompTIA Security+",
        "Microsoft Certified: Azure Developer Associate",
        "Databricks Certified Associate Developer for Apache Spark",
        "Google Professional Data Engineer",
        "Offensive Security Certified Professional (OSCP)",
    ],
    "Engineering": [
        "Professional Engineer (PE) License",
        "Certified LabVIEW Developer (CLD)",
        "IPC-A-610 Certified IPC Specialist",
        "Certified Functional Safety Engineer (CFSE)",
        "Project Management Professional (PMP)",
        "Certified Reliability Engineer (CRE)",
        "Six Sigma Green Belt (SSGB)",
        "SOLIDWORKS Certified Professional (CSWP)",
        "Certified Automation Professional (CAP)",
        "FAA Part 107 Remote Pilot Certificate",
    ],
}

# ── Tech keyword set ──────────────────────────────────────────────────────────

TECH_KEYWORDS: set[str] = {
    "python", "go", "golang", "java", "rust", "c++", "ruby", "swift", "kotlin",
    "typescript", "javascript", "scala", "elixir", "haskell", "r", "matlab",
    "fortran", "cobol", "perl", "lua", "dart", "solidity",
    "react", "vue", "angular", "next.js", "django", "fastapi", "flask",
    "spring", "rails", "express", "laravel", "pytorch", "tensorflow", "keras",
    "hugging face", "scikit", "spark", "flink", "kafka", "airflow", "dbt",
    "ros", "ros2", "unity", "unreal", "opengl", "vulkan", "webassembly",
    "kubernetes", "docker", "terraform", "ansible", "helm", "aws", "gcp",
    "azure", "ci/cd", "jenkins", "github actions", "linux", "nginx",
    "istio", "prometheus", "grafana", "datadog", "snowflake", "databricks",
    "fpga", "rtos", "arm", "cortex", "pcb", "vhdl", "verilog", "can bus",
    "plc", "scada", "dsp", "rf", "5g", "iot", "embedded", "firmware",
    "microcontroller", "avr", "stm32", "zephyr", "freertos",
    "solidworks", "catia", "ansys", "autocad", "revit", "bim", "cfd",
    "fea", "gd&t", "simulink",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
    "sqlite", "oracle", "selenium", "cypress", "pytest", "appsec",
    "penetration", "siem", "splunk",
    "mlops", "nlp", "llm", "computer vision", "reinforcement", "generative",
    "blockchain", "smart contract", "autonomous", "robotics", "mechatronics",
    "biomedical", "pharmaceutical", "geotechnical", "hydraulic", "satellite",
    "fiber optic", "voip", "antenna", "signal processing",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dedup(items: list[str], cap: Optional[int] = None) -> list[str]:
    """Case-insensitive dedup preserving first-seen order, with an optional cap."""
    seen:   set[str]  = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
        if cap is not None and len(result) >= cap:
            break
    return result


def _title_to_doc_id(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def _tokenize(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def _is_duplicate(new_title: str, existing_titles: list[str],
                  threshold: float = 0.75) -> bool:
    new_tokens = _tokenize(new_title)
    if not new_tokens:
        return True
    for existing in existing_titles:
        ex_tokens = _tokenize(existing)
        if not ex_tokens:
            continue
        union = new_tokens | ex_tokens
        if len(new_tokens & ex_tokens) / len(union) >= threshold:
            return True
    return False


def _has_tech_keyword(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in TECH_KEYWORDS)


def _is_valid_cert(item: str) -> bool:
    item = item.strip()
    if not item or len(item) < 3:
        return False
    if re.fullmatch(r"[\d\W]+", item):
        return False
    if len(item.split()) > 10:
        return False
    return True


def merge_skills(new: list[str], old: list[str], cap: int = 10) -> list[str]:
    """
    LLM-fresh skills (new) take priority; stored skills fill the remainder.
    No hard cap by default — store everything unique, caller can pass cap.
    """
    return _dedup(new + old, cap=cap if cap else None)


def merge_certs(new: list[str], old: list[str], cap: int = 6) -> list[str]:
    """LLM-fresh certs take priority; stored certs fill up to cap."""
    return _dedup(new + old, cap=cap)

# ─────────────────────────────────────────────────────────────────────────────
# Firestore
# ─────────────────────────────────────────────────────────────────────────────

def get_existing_docs() -> dict[str, dict]:
    return {doc.id: doc.to_dict() for doc in db.collection(FIRESTORE_JOBS_COLLECTION).stream()}


def get_existing_titles(existing_docs: dict[str, dict]) -> list[str]:
    return [v.get("title", "") for v in existing_docs.values()]

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a job listing generator. Output ONLY a valid JSON array — no markdown, no code fences, no explanation.

Each element must match this schema exactly:
{"title":"...","industry":"Engineering","description":"...","required_skills":["skill1","skill2","skill3","skill4","skill5"],"required_certifications":["Cert Name 1","Cert Name 2"]}

Hard rules:
- "industry" must be exactly "Engineering" or "Computing" — nothing else
- "required_skills" must be a list of 5 specific technologies, tools, or frameworks relevant to the title
- "required_certifications" must be a list of 1 to 3 real, widely-recognised certifications directly relevant to the title. Use the full official name. Do NOT invent certifications.
- "title" must always include the primary technology or stack (e.g. "Python Backend Engineer", "FPGA Design Engineer")
- Every title must be completely unique — no variations of the same role
- "description" must be 1-2 short sentences with no special characters or newlines
- No trailing commas. No extra fields. No comments. Pure JSON only."""


def build_prompt(batch_size: int, all_existing_titles: list[str], industry: str) -> str:
    style  = ", ".join(STYLE_EXAMPLES[industry])
    certs  = ", ".join(CERT_EXAMPLES[industry])

    # Cap the avoid list to the most recent AVOID_LIST_LIMIT titles so the
    # prompt never overflows Zephyr's context window.
    avoid_titles = all_existing_titles[-AVOID_LIST_LIMIT:]
    avoid_block  = (
            "\n\nDo NOT generate any of these titles or anything similar:\n"
            + "\n".join(f"- {t}" for t in avoid_titles)
    ) if avoid_titles else ""

    return (
        f"Generate EXACTLY {batch_size} UNIQUE job listings for the {industry} industry.\n\n"
        f"Title format (style guide only — do not copy verbatim):\n{style}\n\n"
        f"Certification style guide (naming convention examples — pick certs that fit each role):\n{certs}\n\n"
        f"Requirements:\n"
        f"- Every title must include the specific technology, tool, or stack\n"
        f"- Every title must be different from every other title in this batch\n"
        f"- Prefer niche, specialised, or emerging roles over common ones\n"
        f'- Set "industry" to exactly "{industry}" for every object\n'
        f"- required_certifications: 1–3 real certifications directly relevant to each title"
        f"{avoid_block}\n\n"
        f"Return ONLY a JSON array with exactly {batch_size} objects. No other text."
    )

# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _call_zephyr(prompt: str) -> str:
    resp = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type":  "application/json",
        },
        json={
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 3000,
            "temperature": 0.7,
            "stream": False,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()

# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_objects_from_text(text: str) -> list[dict]:
    """
    Walk `text` tracking brace depth and collect every balanced {...} block.

    Splitting on '"title"' (the original approach) is unreliable because the
    word 'title' can appear inside description strings, causing splits mid-object.
    Brace-depth tracking is safer and handles truncated arrays correctly.
    """
    objects: list[dict] = []
    depth  = 0
    start  = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    # Try one cheap repair: remove a trailing comma inside the object
                    repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
                    try:
                        obj = json.loads(repaired)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                start = None

    return objects


def _clean_llm_output(raw: str) -> str:
    """Strip Zephyr chat-template tokens and markdown fences."""
    text = re.sub(r"<\|[a-zA-Z_]+\|>", "", raw)          # <|assistant|> etc.
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.strip().strip("`").strip()


def _normalise_industry(value: str) -> Optional[str]:
    """
    Accept minor casing / spacing variants Zephyr sometimes emits.
    Returns the canonical string or None if unrecognisable.
    """
    v = value.strip().lower()
    if v == "computing":
        return "Computing"
    if v == "engineering":
        return "Engineering"
    return None


def parse_jobs(
        raw: str,
        batch_size: int,
        expected_industry: str,
        all_existing_titles: list[str],
) -> list[dict]:
    """
    Parse the LLM response and return validated job dicts.

    Extraction order:
      1. Try full JSON array parse on the cleaned text.
      2. Fall back to brace-depth object scanner (handles truncated arrays).
    """
    cleaned = _clean_llm_output(raw)

    # Repair a common Zephyr failure: bare industry value instead of key/value
    # e.g.  "title": "...",\n    "Computing",\n    "description": ...
    cleaned = re.sub(
        r',\s*"(Engineering|Computing)"\s*,',
        r', "industry": "\1",',
        cleaned,
    )

    # ── Attempt 1: full array parse ───────────────────────────────────────────
    jobs_raw: list[dict] = []
    array_start = cleaned.find("[")
    array_end   = cleaned.rfind("]")          # -1 if absent

    if array_start != -1 and array_end > array_start:
        try:
            jobs_raw = json.loads(cleaned[array_start : array_end + 1])
        except json.JSONDecodeError:
            pass  # fall through to scanner

    # ── Attempt 2: object scanner ─────────────────────────────────────────────
    if not jobs_raw:
        print("  [WARN] Full array parse failed — using brace-depth scanner …")
        jobs_raw = _extract_objects_from_text(cleaned)

    if not jobs_raw:
        print("  [DEBUG] Raw LLM output (first 2000 chars):")
        print(raw[:2000])
        raise ValueError("No parseable job objects found in LLM output.")

    # ── Validate each record ──────────────────────────────────────────────────
    validated:       list[dict] = []
    seen_this_batch: list[str]  = []

    for i, job in enumerate(jobs_raw):
        if not isinstance(job, dict):
            continue

        # Required fields
        missing = [k for k in ("title", "industry", "description",
                               "required_skills", "required_certifications")
                   if k not in job]
        if missing:
            print(f"  [WARN] Record {i+1} missing {missing} — skipping.")
            continue

        # Industry: accept minor casing differences, reject wrong industry
        industry = _normalise_industry(str(job["industry"]))
        if industry is None:
            print(f"  [WARN] '{job.get('title', '?')}' unrecognised industry "
                  f"'{job['industry']}' — skipping.")
            continue
        if industry != expected_industry:
            print(f"  [WARN] '{job.get('title', '?')}' is '{industry}', "
                  f"expected '{expected_industry}' — skipping.")
            continue

        # Title
        title = str(job["title"]).strip()
        if not _has_tech_keyword(title):
            print(f"  [WARN] '{title}' has no tech keyword — skipping.")
            continue
        if _is_duplicate(title, seen_this_batch):
            print(f"  [WARN] '{title}' is near-duplicate within batch — skipping.")
            continue

        # Skills: must be a list; accept 4 or 5 (trim to 5 if longer)
        skills = job["required_skills"]
        if not isinstance(skills, list):
            print(f"  [WARN] '{title}' skills is not a list — skipping.")
            continue
        skills = [str(s).strip() for s in skills if str(s).strip()]
        if len(skills) > 5:
            skills = skills[:5]
        if len(skills) < 4:
            # Fewer than 4 is too sparse to be useful
            print(f"  [WARN] '{title}' has only {len(skills)} skill(s) — skipping.")
            continue

        # Certifications
        raw_certs = job["required_certifications"]
        if not isinstance(raw_certs, list):
            print(f"  [WARN] '{title}' certifications is not a list — skipping.")
            continue
        clean_certs = [c for c in (str(c).strip() for c in raw_certs) if _is_valid_cert(c)]
        clean_certs = clean_certs[:3]
        if not clean_certs:
            print(f"  [WARN] '{title}' has no valid certifications — skipping.")
            continue

        description = str(job["description"]).replace("\n", " ").replace("\r", "").strip()

        validated.append({
            "title":                   title,
            "industry":                industry,
            "description":             description,
            "required_skills":         skills,
            "required_certifications": clean_certs,
        })
        seen_this_batch.append(title)

        if len(validated) >= batch_size:
            break

    return validated

# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def _build_profile(job: dict) -> str:
    """
    Build the SBERT input string for a job.

    Skills are joined with ', ' (not spaces and definitely not repeated 3×)
    so SBERT sees them as distinct phrases rather than one fused token.
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
    print(f"  [EMBED] Encoding {len(jobs)} job(s) …")
    vecs = sbert.encode(profiles, convert_to_numpy=True, show_progress_bar=False)
    for job, vec in zip(jobs, vecs):
        job["embedding"] = vec.astype(float).tolist()
    return jobs

# ─────────────────────────────────────────────────────────────────────────────
# Firestore seed
# ─────────────────────────────────────────────────────────────────────────────

def seed_jobs(jobs: list[dict], existing_docs: dict[str, dict]) -> tuple[int, int]:
    """
    Write each job to Firestore.
      NEW     → full document write
      EXISTING→ LLM output takes priority; stored data fills the remainder.
                Description and embedding are always refreshed.

    Returns (new_count, merged_count).
    """
    col     = db.collection(FIRESTORE_JOBS_COLLECTION)
    new_c   = 0
    merged_c = 0

    for job in jobs:
        doc_id   = _title_to_doc_id(job["title"])
        existing = existing_docs.get(doc_id)

        if existing is None:
            col.document(doc_id).set(job)
            existing_docs[doc_id] = job
            new_c += 1
            print(f"  ✓ NEW     [{job['industry']}] {job['title']}")
            print(f"            certs: {', '.join(job['required_certifications'])}")

        else:
            old_skills = existing.get("required_skills", [])
            old_certs  = existing.get("required_certifications", [])

            # LLM-fresh data goes first so it takes priority in the dedup pass
            merged_skills = merge_skills(job["required_skills"], old_skills)
            merged_certs  = merge_certs(job["required_certifications"], old_certs)

            merged_job: dict = {
                **existing,
                "title":                   job["title"],
                "industry":                job["industry"],
                "description":             job["description"],
                "required_skills":         merged_skills,
                "required_certifications": merged_certs,
            }

            vec = sbert.encode(
                [_build_profile(merged_job)],
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            merged_job["embedding"] = vec.astype(float).tolist()

            col.document(doc_id).set(merged_job)
            existing_docs[doc_id] = merged_job
            merged_c += 1

            print(
                f"  ↻ MERGED  [{job['industry']}] {job['title']}  "
                f"(skills {len(old_skills)}→{len(merged_skills)}, "
                f"certs {len(old_certs)}→{len(merged_certs)})"
            )

    return new_c, merged_c

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INDUSTRY_TARGET = JOB_COUNT // 2
    TOTAL_BATCHES   = JOB_COUNT // BATCH_SIZE

    print("[INFO] Checking existing Firestore jobs …")
    existing_docs   = get_existing_docs()
    existing_titles = get_existing_titles(existing_docs)
    print(f"[INFO] {len(existing_titles)} existing job(s) found.\n")

    print(f"[INFO] Plan    : {JOB_COUNT} jobs | {BATCH_SIZE}/batch | {TOTAL_BATCHES} batch(es)")
    print(f"[INFO] Order   : Batch 1=Computing, Batch 2=Engineering, alternating")
    print(f"[INFO] Avoid   : last {AVOID_LIST_LIMIT} titles sent per prompt")
    print(f"[INFO] LLM     : {HF_MODEL}")
    print(f"[INFO] Embed   : sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Store   : Firestore / {FIRESTORE_JOBS_COLLECTION}\n")

    industry_counts:  dict[str, int] = {"Engineering": 0, "Computing": 0}
    industry_targets: dict[str, int] = {"Engineering": INDUSTRY_TARGET,
                                        "Computing":   INDUSTRY_TARGET}

    total_new    = 0
    total_merged = 0
    all_jobs:    list[dict] = []

    for batch_num in range(1, TOTAL_BATCHES + 1):
        industry = INDUSTRY_ORDER[(batch_num - 1) % 2]

        # If this industry's quota is full, try the other one
        if industry_counts[industry] >= industry_targets[industry]:
            other = "Engineering" if industry == "Computing" else "Computing"
            if industry_counts[other] >= industry_targets[other]:
                print("[INFO] Both industry quotas met — stopping early.")
                break
            industry = other

        remaining    = industry_targets[industry] - industry_counts[industry]
        current_size = min(BATCH_SIZE, remaining)

        print(f"{'─' * 60}")
        print(f"[BATCH {batch_num:02d}/{TOTAL_BATCHES}]  {industry}  "
              f"— requesting {current_size} job(s)  "
              f"(quota: {industry_counts[industry]}/{industry_targets[industry]})")

        batch_jobs: list[dict] = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt     = build_prompt(current_size, existing_titles, industry)
                raw_output = _call_zephyr(prompt)
                batch_jobs = parse_jobs(raw_output, current_size, industry,
                                        existing_titles)
                if batch_jobs:
                    break
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] 0 valid jobs — retrying …")
            except (ValueError, RuntimeError) as exc:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [SKIP] Batch {batch_num} failed after {MAX_RETRIES} attempt(s).")

        if not batch_jobs:
            continue

        batch_jobs = embed_jobs(batch_jobs)
        n, m       = seed_jobs(batch_jobs, existing_docs)

        total_new              += n
        total_merged           += m
        industry_counts[industry] += n
        all_jobs.extend(batch_jobs)
        existing_titles.extend(j["title"] for j in batch_jobs)

        time.sleep(1)   # polite rate-limit pause

    # ── Summary ───────────────────────────────────────────────────────────────
    all_stored_certs: set[str] = {
        c
        for doc in existing_docs.values()
        for c in doc.get("required_certifications", [])
    }

    print(f"\n{'═' * 55}")
    print(f"  Jobs processed       : {len(all_jobs)}")
    print(f"  New documents        : {total_new}")
    print(f"  Merged documents     : {total_merged}")
    print(f"  Engineering new      : {industry_counts['Engineering']}")
    print(f"  Computing new        : {industry_counts['Computing']}")
    print(f"  Total in Firestore   : {len(existing_docs)}")
    print(f"  Unique certs stored  : {len(all_stored_certs)}")
    print(f"  Collection           : {FIRESTORE_JOBS_COLLECTION}")
    print(f"  Model                : {HF_MODEL}")
    print(f"{'═' * 55}")