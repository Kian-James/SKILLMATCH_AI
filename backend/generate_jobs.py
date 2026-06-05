"""
generate_jobs.py
────────────────
Generates job listings via Llama 3.1 8B (Cerebras / HF Inference Provider),
encodes each job with SBERT (all-MiniLM-L6-v2), and stores both the job data
and its embedding vector in Firestore.

Batch strategy:
  - 100 jobs total, 10 per batch (10 batches)
  - Batches alternate: Batch 1 = Computing, Batch 2 = Engineering, ...
  - If a job title already exists in Firestore, its required_skills and
    required_certifications are MERGED (union of old + new, capped at 10 skills
    / 6 certs), and the embedding is re-generated.

Uniqueness strategy:
  1. Every already-used title is sent in the avoid list every batch.
  2. Fuzzy duplicate detection via Jaccard token overlap (threshold 0.75).
  3. Titles must contain at least one tech/stack keyword.

Cert strategy:
  - The LLM picks 1–3 real, role-appropriate certifications per job title.
  - Certs are validated: must be a non-empty list of 1–3 strings, each ≤ 10
    words. Items that look like generic noise (< 3 chars, all digits) are
    dropped.
  - Merged docs re-generate their embedding using the combined cert + skill
    profile so certs.py's allowlist match stays accurate.

Usage:
    python generate_jobs.py

Requirements in .env:
    HF_TOKEN                   – Hugging Face API token
    FIREBASE_CREDENTIALS_PATH  – path to your serviceAccountKey.json
    FIRESTORE_JOBS_COLLECTION  – Firestore collection name (default: jobs)
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

JOB_COUNT   = 100
BATCH_SIZE  = 10
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"

# Batch 1 → Computing, Batch 2 → Engineering, alternating
INDUSTRY_ORDER = ["Computing", "Engineering"]

# ── Style examples ────────────────────────────────────────────────────────────
STYLE_EXAMPLES = {
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

# ── Cert examples (style guide for the LLM, not an exhaustive allowlist) ─────
# These show the naming convention the LLM should follow. The actual allowlist
# that certs.py uses at runtime is the union of all required_certifications
# stored across job documents in Firestore.
CERT_EXAMPLES = {
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

# ── Tech keywords ─────────────────────────────────────────────────────────────
TECH_KEYWORDS = {
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

# ── Validate env ──────────────────────────────────────────────────────────────
errors = []
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


# ── Firestore helpers ─────────────────────────────────────────────────────────

def title_to_doc_id(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def get_existing_docs() -> dict:
    docs = db.collection(FIRESTORE_JOBS_COLLECTION).stream()
    return {doc.id: doc.to_dict() for doc in docs}


def get_existing_titles(existing_docs: dict) -> list:
    return [v.get("title", "") for v in existing_docs.values()]


# ── Skill / cert merging ──────────────────────────────────────────────────────

def merge_list(old: list, new: list, cap: int) -> list:
    """
    Merge two lists, preserving order and removing case-insensitive duplicates.
    Old items come first; new unique items are appended. Result is capped at
    `cap` entries.
    """
    seen:   set[str]  = set()
    merged: list[str] = []
    for item in old + new:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(item.strip())
        if len(merged) >= cap:
            break
    return merged


# Back-compat aliases used throughout the rest of the file
def merge_skills(old: list, new: list, cap: int = 10) -> list:
    return merge_list(old, new, cap)


def merge_certs(old: list, new: list, cap: int = 6) -> list:
    return merge_list(old, new, cap)


# ── Duplicate detection ───────────────────────────────────────────────────────

def _tokenize(title: str) -> set:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def is_duplicate(new_title: str, existing_titles: list, threshold: float = 0.75) -> bool:
    new_tokens = _tokenize(new_title)
    if not new_tokens:
        return True
    for existing in existing_titles:
        existing_tokens = _tokenize(existing)
        if not existing_tokens:
            continue
        intersection = new_tokens & existing_tokens
        union        = new_tokens | existing_tokens
        if len(intersection) / len(union) >= threshold:
            return True
    return False


def has_tech_keyword(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in TECH_KEYWORDS)


# ── Cert item validation ──────────────────────────────────────────────────────

def _is_valid_cert(item: str) -> bool:
    """
    Cheap noise guard on a single cert string.
    Mirrors the 10-word guard in certs.py plus a minimum-length check so
    obviously bad values (empty, single digits, stray punctuation) are dropped
    before they ever reach Firestore.
    """
    item = item.strip()
    if not item:
        return False
    # Too short to be a real cert name
    if len(item) < 3:
        return False
    # All digits / punctuation — not a cert name
    if re.fullmatch(r"[\d\W]+", item):
        return False
    # certs.py drops anything > 10 words, so we match that ceiling here
    if len(item.split()) > 10:
        return False
    return True


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a job listing generator. Output ONLY a valid JSON array — no markdown, no code fences, no explanation.

Each element must match this schema exactly:
{"title":"...","industry":"Engineering","description":"...","required_skills":["skill1","skill2","skill3","skill4","skill5"],"required_certifications":["Cert Name 1","Cert Name 2"]}

Hard rules:
- "industry" must be exactly "Engineering" or "Computing" — nothing else
- "required_skills" must be exactly 5 specific technologies, tools, or frameworks that match the title
- "required_certifications" must be a list of 1 to 3 real, widely-recognised certifications that are directly relevant to the job title. Use the full official name (e.g. "AWS Certified Solutions Architect – Associate", not just "AWS cert"). Do NOT invent certifications.
- "title" must always include the primary technology or stack (e.g. "Python Backend Engineer", "FPGA Design Engineer")
- Every title must be completely unique — no variations of the same role
- "description" must be 1-2 short sentences with no special characters or newlines
- No trailing commas. No extra fields. No comments. Pure JSON only."""


def build_prompt(batch_size: int, all_existing_titles: list, industry: str) -> str:
    style_examples = ", ".join(STYLE_EXAMPLES[industry])
    cert_examples  = ", ".join(CERT_EXAMPLES[industry])

    avoid_block = ""
    if all_existing_titles:
        avoid_block = (
                f"\n\nDo NOT generate any of these titles or anything similar to them:\n"
                + "\n".join(f"- {t}" for t in all_existing_titles)
        )

    return (
        f"Generate EXACTLY {batch_size} UNIQUE job listings for the {industry} industry.\n\n"
        f"Title format (use as style guide only — do not copy these):\n{style_examples}\n\n"
        f"Certification style guide (use as examples of naming convention and relevance — "
        f"do not copy these verbatim, pick certs that fit each specific job title):\n{cert_examples}\n\n"
        f"Requirements:\n"
        f"- Every title must include the specific technology, tool, or stack\n"
        f"- Every title must be different from every other title in this batch\n"
        f"- Think of niche, specialised, or emerging roles — not just common ones\n"
        f"- Set 'industry' to exactly \"{industry}\" for every object\n"
        f"- required_certifications: 1–3 real certifications directly relevant to the job title"
        f"{avoid_block}\n\n"
        f"Return ONLY a JSON array with exactly {batch_size} objects. No other text."
    )


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llama(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 1800,   # slightly higher to accommodate cert strings
        "temperature": 0.7,
        "stream": False,
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"API returned {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


# ── Parser ────────────────────────────────────────────────────────────────────

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


def parse_jobs(raw: str, batch_size: int, expected_industry: str,
               all_existing_titles: list) -> list:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    start = cleaned.find("[")
    end   = cleaned.rfind("]") + 1

    if start != -1 and end > start:
        try:
            jobs_raw = json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            print("  [WARN] Array parse failed — using object scanner …")
            jobs_raw = _repair_truncated_json(cleaned)
    else:
        print("  [WARN] No JSON array found — using object scanner …")
        jobs_raw = _repair_truncated_json(cleaned)

    if not jobs_raw:
        raise ValueError("No parseable job objects found in model output.")

    validated       = []
    seen_this_batch = []

    for i, job in enumerate(jobs_raw):
        # ── Required field check ──────────────────────────────────────────────
        missing = [k for k in ("title", "industry", "description",
                               "required_skills", "required_certifications")
                   if k not in job]
        if missing:
            print(f"  [WARN] Job {i+1} missing {missing} — skipping.")
            continue

        # ── Industry validation ───────────────────────────────────────────────
        raw_industry = job["industry"].strip()
        if raw_industry not in ("Engineering", "Computing"):
            print(f"  [WARN] '{job['title']}' invalid industry '{raw_industry}' — skipping.")
            continue
        if raw_industry != expected_industry:
            print(f"  [WARN] '{job['title']}' is '{raw_industry}', expected '{expected_industry}' — skipping.")
            continue

        # ── Title validation ──────────────────────────────────────────────────
        title = job["title"].strip()

        if not has_tech_keyword(title):
            print(f"  [WARN] '{title}' has no tech keyword — too generic, skipping.")
            continue

        if is_duplicate(title, seen_this_batch):
            print(f"  [WARN] '{title}' is a near-duplicate within this batch — skipping.")
            continue

        # ── Skills validation ─────────────────────────────────────────────────
        skills = job["required_skills"]
        if not isinstance(skills, list):
            print(f"  [WARN] '{title}' has non-list skills — skipping.")
            continue
        if len(skills) > 5:
            skills = skills[:5]
        elif len(skills) < 5:
            print(f"  [WARN] '{title}' has {len(skills)} skills (need 5) — skipping.")
            continue

        # ── Certifications validation ─────────────────────────────────────────
        raw_certs = job["required_certifications"]
        if not isinstance(raw_certs, list):
            print(f"  [WARN] '{title}' required_certifications is not a list — skipping.")
            continue

        # Filter noise items; keep 1–3 valid entries
        clean_certs = [c for c in (str(c).strip() for c in raw_certs) if _is_valid_cert(c)]
        clean_certs = clean_certs[:3]   # hard cap at 3

        if not clean_certs:
            print(f"  [WARN] '{title}' has no valid certifications after filtering — skipping.")
            continue

        # ── Build validated record ────────────────────────────────────────────
        description = job["description"].replace("\n", " ").replace("\r", "").strip()

        validated.append({
            "title":                   title,
            "industry":                raw_industry,
            "description":             description,
            "required_skills":         [str(s).strip() for s in skills],
            "required_certifications": clean_certs,
        })
        seen_this_batch.append(title)

        if len(validated) >= batch_size:
            break

    return validated


# ── Embedding ─────────────────────────────────────────────────────────────────

def _build_profile(job: dict) -> str:
    """
    Build the SBERT input string for a job.
    Certifications are appended so that resume cert matches influence
    the embedding similarity score used in pipeline.py's scoring stage.
    """
    certs_str = " ".join(job.get("required_certifications", []))
    return (
        f"Job: {job['title']}. Industry: {job['industry']}. "
        f"Core Skills: {' '.join(job['required_skills'] * 3)}. "
        f"Certifications: {certs_str}. "
        f"Desc: {job['description']}"
    )


def embed_jobs(jobs: list) -> list:
    profiles = [_build_profile(j) for j in jobs]
    print(f"  [EMBED] Encoding {len(jobs)} job(s) with SBERT …")
    vecs = sbert.encode(profiles, convert_to_numpy=True, show_progress_bar=False)
    for job, vec in zip(jobs, vecs):
        job["embedding"] = vec.astype(float).tolist()
    return jobs


# ── Seeder (with merge logic) ─────────────────────────────────────────────────

def seed_jobs(jobs: list, existing_docs: dict) -> tuple[int, int]:
    """
    Write each job to Firestore.
      - NEW job      → full document set
      - EXISTING job → merge required_skills (cap 10) and
                       required_certifications (cap 6), refresh description
                       and embedding.

    Returns (new_count, merged_count).
    """
    collection_ref = db.collection(FIRESTORE_JOBS_COLLECTION)
    new_count    = 0
    merged_count = 0

    for job in jobs:
        doc_id   = title_to_doc_id(job["title"])
        existing = existing_docs.get(doc_id)

        if existing is None:
            # ── Brand-new job ─────────────────────────────────────────────────
            collection_ref.document(doc_id).set(job)
            existing_docs[doc_id] = job
            new_count += 1
            certs_summary = ", ".join(job["required_certifications"])
            print(f"  ✓ NEW     [{job['industry']}] {job['title']}")
            print(f"            certs: {certs_summary}")
        else:
            # ── Merge into existing job ───────────────────────────────────────
            old_skills = existing.get("required_skills", [])
            old_certs  = existing.get("required_certifications", [])

            merged_skills = merge_skills(old_skills, job["required_skills"])
            merged_certs  = merge_certs(old_certs,   job["required_certifications"])

            merged_job = {
                **existing,
                "required_skills":         merged_skills,
                "required_certifications": merged_certs,
                "description":             job["description"],
            }

            # Re-encode with the merged profile so the vector stays accurate
            profile = _build_profile(merged_job)
            vec = sbert.encode([profile], convert_to_numpy=True, show_progress_bar=False)[0]
            merged_job["embedding"] = vec.astype(float).tolist()

            collection_ref.document(doc_id).set(merged_job)
            existing_docs[doc_id] = merged_job
            merged_count += 1

            skills_added = len(merged_skills) - len(old_skills)
            certs_added  = len(merged_certs)  - len(old_certs)
            print(f"  ↻ MERGED  [{job['industry']}] {job['title']}")
            print(f"            skills: {len(old_skills)} → {len(merged_skills)} (+{skills_added})  "
                  f"certs: {len(old_certs)} → {len(merged_certs)} (+{certs_added})")

    return new_count, merged_count


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INDUSTRY_TARGET = JOB_COUNT // 2
    TOTAL_BATCHES   = JOB_COUNT // BATCH_SIZE

    print("[INFO] Checking existing Firestore jobs …")
    existing_docs   = get_existing_docs()
    existing_titles = get_existing_titles(existing_docs)
    print(f"[INFO] {len(existing_titles)} existing jobs found.\n")

    print(f"[INFO] Plan: {JOB_COUNT} jobs  |  {BATCH_SIZE}/batch  |  {TOTAL_BATCHES} batches")
    print(f"[INFO] Order: Batch 1=Computing, Batch 2=Engineering, alternating …")
    print(f"[INFO] LLM  : {HF_MODEL}")
    print(f"[INFO] Embed: sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Store: Firestore / {FIRESTORE_JOBS_COLLECTION}\n")

    industry_counts  = {"Engineering": 0, "Computing": 0}
    industry_targets = {"Engineering": INDUSTRY_TARGET, "Computing": INDUSTRY_TARGET}

    total_new    = 0
    total_merged = 0
    all_jobs     = []

    for batch_num in range(1, TOTAL_BATCHES + 1):
        industry = INDUSTRY_ORDER[(batch_num - 1) % 2]

        if industry_counts[industry] >= industry_targets[industry]:
            other = "Engineering" if industry == "Computing" else "Computing"
            if industry_counts[other] >= industry_targets[other]:
                print(f"[INFO] Both industry quotas met — stopping early.")
                break
            industry = other

        remaining    = industry_targets[industry] - industry_counts[industry]
        current_size = min(BATCH_SIZE, remaining)

        print(f"{'─'*60}")
        print(f"[BATCH {batch_num:02d}/{TOTAL_BATCHES}]  {industry}  "
              f"— requesting {current_size} job(s)  "
              f"(quota: {industry_counts[industry]}/{industry_targets[industry]})")

        jobs = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt     = build_prompt(current_size, existing_titles, industry)
                raw_output = call_llama(prompt)
                jobs       = parse_jobs(raw_output, current_size, industry,
                                        existing_titles)
                if jobs:
                    break
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] 0 valid jobs returned — retrying …")
            except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] Error: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [SKIP] Batch {batch_num} failed after {MAX_RETRIES} attempts.")

        if not jobs:
            continue

        jobs = embed_jobs(jobs)
        n, m = seed_jobs(jobs, existing_docs)

        total_new    += n
        total_merged += m

        industry_counts[industry] += n
        all_jobs.extend(jobs)

        existing_titles.extend([j["title"] for j in jobs])

        time.sleep(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    # Collect all unique certs now stored in Firestore for the summary line
    all_stored_certs: set[str] = set()
    for doc in existing_docs.values():
        for c in doc.get("required_certifications", []):
            all_stored_certs.add(c)

    print(f"\n{'═'*60}")
    print(f"[DONE]  Jobs processed        : {len(all_jobs)}")
    print(f"[DONE]  New documents         : {total_new}")
    print(f"[DONE]  Merged documents      : {total_merged}")
    print(f"[DONE]  Engineering new       : {industry_counts['Engineering']}")
    print(f"[DONE]  Computing new         : {industry_counts['Computing']}")
    print(f"[DONE]  Total in Firestore    : {len(existing_docs)}")
    print(f"[DONE]  Unique certs stored   : {len(all_stored_certs)}")
    print(f"[DONE]  Collection            : {FIRESTORE_JOBS_COLLECTION}")
    print(f"[DONE]  Model                 : {HF_MODEL}")