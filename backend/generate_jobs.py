"""
generate_jobs.py
────────────────
Generates job listings via Llama 3.1 8B (Cerebras / HF Inference Provider),
encodes each job with SBERT (all-MiniLM-L6-v2), and stores both the job data
and its embedding vector in Firestore.

Uniqueness strategy:
  1. Every already-used title is sent in the avoid list every batch — not just
     the last 15 — so the model is always aware of everything it has made.
  2. Fuzzy duplicate detection: new titles are compared against all existing
     titles using token overlap. Near-matches (e.g. "Python Backend Dev" vs
     "Python Backend Developer") are rejected before hitting Firestore.
  3. Titles are validated to contain at least one tech/stack keyword so the
     model cannot fall back to generic names.

Usage:
    python generate_jobs.py

Requirements in .env:
    HF_TOKEN                   – Hugging Face API token
    FIREBASE_CREDENTIALS_PATH  – path to your serviceAccountKey.json
    FIRESTORE_JOBS_COLLECTION  – Firestore collection name (default: jobs)
    JOB_COUNT                  – total jobs to generate (default: 50)
    BATCH_SIZE                 – jobs per API call (default: 5)
    MAX_RETRIES                – retries per failed batch (default: 3)
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
JOB_COUNT                 = int(os.getenv("JOB_COUNT", "50"))
BATCH_SIZE                = int(os.getenv("BATCH_SIZE", "5"))
MAX_RETRIES               = int(os.getenv("MAX_RETRIES", "3"))

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct:cerebras"

# ── Style examples (never sent as titles to copy — only as format guidance) ──
# The model sees these so it understands the specificity level expected.
# It is explicitly told NOT to copy these — only to match their style.
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

# Tech/stack keywords — a generated title must contain at least one of these
# so generic titles like "Software Engineer" get rejected automatically.
TECH_KEYWORDS = {
    # Languages
    "python", "go", "golang", "java", "rust", "c++", "ruby", "swift", "kotlin",
    "typescript", "javascript", "scala", "elixir", "haskell", "r", "matlab",
    "fortran", "cobol", "perl", "lua", "dart", "solidity",
    # Frameworks / libraries
    "react", "vue", "angular", "next.js", "django", "fastapi", "flask",
    "spring", "rails", "express", "laravel", "pytorch", "tensorflow", "keras",
    "hugging face", "scikit", "spark", "flink", "kafka", "airflow", "dbt",
    "ros", "ros2", "unity", "unreal", "opengl", "vulkan", "webassembly",
    # Infra / cloud / devops
    "kubernetes", "docker", "terraform", "ansible", "helm", "aws", "gcp",
    "azure", "ci/cd", "jenkins", "github actions", "linux", "nginx",
    "istio", "prometheus", "grafana", "datadog", "snowflake", "databricks",
    # Hardware / EE / embedded
    "fpga", "rtos", "arm", "cortex", "pcb", "vhdl", "verilog", "can bus",
    "plc", "scada", "dsp", "rf", "5g", "iot", "embedded", "firmware",
    "microcontroller", "avr", "stm32", "zephyr", "freertos",
    # Mechanical / civil / chem
    "solidworks", "catia", "ansys", "autocad", "revit", "bim", "cfd",
    "fea", "gd&t", "ros2", "matlab", "simulink",
    # DB / security / qa
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
    "sqlite", "oracle", "selenium", "cypress", "pytest", "appsec",
    "penetration", "siem", "splunk",
    # Specialisms (counted as specific enough on their own)
    "mlops", "nlp", "llm", "computer vision", "reinforcement", "generative",
    "blockchain", "smart contract", "autonomous", "robotics", "mechatronics",
    "biomedical", "pharmaceutical", "geotechnical", "hydraulic", "satellite",
    "fiber optic", "voip", "antenna", "signal processing",
}

# ── Validate env ─────────────────────────────────────────────────────────────
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


# ── Duplicate detection ───────────────────────────────────────────────────────

def _tokenize(title: str) -> set:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def is_duplicate(new_title: str, existing_titles: list, threshold: float = 0.75) -> bool:
    """
    Returns True if new_title is too similar to any existing title.
    Uses Jaccard similarity on word tokens.
    e.g. "Python Backend Dev" vs "Python Backend Developer" → 0.75 → duplicate.
    """
    new_tokens = _tokenize(new_title)
    if not new_tokens:
        return True
    for existing in existing_titles:
        existing_tokens = _tokenize(existing)
        if not existing_tokens:
            continue
        intersection = new_tokens & existing_tokens
        union        = new_tokens | existing_tokens
        similarity   = len(intersection) / len(union)
        if similarity >= threshold:
            return True
    return False


def has_tech_keyword(title: str) -> bool:
    """Returns True if the title contains at least one recognised tech keyword."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in TECH_KEYWORDS)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a job listing generator. Output ONLY a valid JSON array — no markdown, no code fences, no explanation.

Each element must match this schema exactly:
{"title":"...","industry":"Engineering","description":"...","required_skills":["skill1","skill2","skill3","skill4","skill5"]}

Hard rules:
- "industry" must be exactly "Engineering" or "Computing" — nothing else
- "required_skills" must be exactly 5 specific technologies, tools, or frameworks that match the title
- "title" must always include the primary technology or stack (e.g. "Python Backend Engineer", "FPGA Design Engineer")
- Every title must be completely unique — no variations of the same role
- "description" must be 1-2 short sentences with no special characters or newlines
- No trailing commas. No extra fields. No comments. Pure JSON only."""


def build_prompt(batch_size: int, all_existing_titles: list, industry: str) -> str:
    style_examples = ", ".join(STYLE_EXAMPLES[industry])

    # Send EVERY existing title so the model has full awareness of what exists
    avoid_block = ""
    if all_existing_titles:
        avoid_block = (
                f"\n\nDo NOT generate any of these titles or anything similar to them:\n"
                + "\n".join(f"- {t}" for t in all_existing_titles)
        )

    return (
        f"Generate EXACTLY {batch_size} UNIQUE job listings for the {industry} industry.\n\n"
        f"Title format (use as style guide only — do not copy these):\n{style_examples}\n\n"
        f"Requirements:\n"
        f"- Every title must include the specific technology, tool, or stack\n"
        f"- Every title must be different from every other title in this batch\n"
        f"- Think of niche, specialised, or emerging roles — not just common ones\n"
        f"- Set 'industry' to exactly \"{industry}\" for every object"
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
        "max_tokens": 1500,
        "temperature": 0.7,  # slightly higher — we WANT creative titles now
        "stream": False,
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"API returned {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> list:
    """Depth-tracking brace scanner — salvages complete objects from truncated output."""
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

    validated      = []
    seen_this_batch = []  # track titles within this batch to catch intra-batch dupes

    for i, job in enumerate(jobs_raw):
        # ── Field presence ────────────────────────────────────────────────────
        missing = [k for k in ("title", "industry", "description", "required_skills")
                   if k not in job]
        if missing:
            print(f"  [WARN] Job {i+1} missing {missing} — skipping.")
            continue

        # ── Industry check ────────────────────────────────────────────────────
        raw_industry = job["industry"].strip()
        if raw_industry not in ("Engineering", "Computing"):
            print(f"  [WARN] '{job['title']}' invalid industry '{raw_industry}' — skipping.")
            continue
        if raw_industry != expected_industry:
            print(f"  [WARN] '{job['title']}' is '{raw_industry}', expected '{expected_industry}' — skipping.")
            continue

        title = job["title"].strip()

        # ── Tech keyword check ────────────────────────────────────────────────
        if not has_tech_keyword(title):
            print(f"  [WARN] '{title}' has no tech keyword — too generic, skipping.")
            continue

        # ── Fuzzy duplicate check against ALL existing + this batch ───────────
        all_seen = all_existing_titles + seen_this_batch
        if is_duplicate(title, all_seen):
            print(f"  [WARN] '{title}' is too similar to an existing title — skipping.")
            continue

        # ── Skills check ──────────────────────────────────────────────────────
        skills = job["required_skills"]
        if not isinstance(skills, list):
            print(f"  [WARN] '{title}' has non-list skills — skipping.")
            continue
        if len(skills) > 5:
            skills = skills[:5]
        elif len(skills) < 5:
            print(f"  [WARN] '{title}' has {len(skills)} skills — skipping.")
            continue

        description = job["description"].replace("\n", " ").replace("\r", "").strip()

        validated.append({
            "title":           title,
            "industry":        raw_industry,
            "description":     description,
            "required_skills": [str(s).strip() for s in skills],
        })
        seen_this_batch.append(title)

        if len(validated) >= batch_size:
            break

    return validated


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_jobs(jobs: list) -> list:
    profiles = [
        f"Job: {j['title']}. Industry: {j['industry']}. "
        f"Core Skills: {' '.join(j['required_skills'] * 3)}. "
        f"Desc: {j['description']}"
        for j in jobs
    ]
    print(f"  [EMBED] Encoding {len(jobs)} job(s) with SBERT …")
    vecs = sbert.encode(profiles, convert_to_numpy=True, show_progress_bar=False)
    for job, vec in zip(jobs, vecs):
        job["embedding"] = vec.astype(float).tolist()
    return jobs


# ── Seeder ────────────────────────────────────────────────────────────────────

def seed_jobs(jobs: list, existing_docs: dict) -> None:
    collection_ref = db.collection(FIRESTORE_JOBS_COLLECTION)
    for job in jobs:
        doc_id = title_to_doc_id(job["title"])
        # By this point duplicates are already filtered — every job here is new
        collection_ref.document(doc_id).set(job)
        existing_docs[doc_id] = job
        print(f"  ✓  [{job['industry']}] {job['title']}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[INFO] Checking existing Firestore jobs …")
    existing_docs   = get_existing_docs()
    existing_titles = get_existing_titles(existing_docs)
    print(f"[INFO] {len(existing_titles)} existing jobs found.")

    industries_cycle = ["Engineering", "Computing"]
    industry_targets = {
        "Engineering": JOB_COUNT - JOB_COUNT // 2,
        "Computing":   JOB_COUNT // 2,
    }
    industry_counts = {"Engineering": 0, "Computing": 0}

    print(f"\n[INFO] Generating {JOB_COUNT} NEW unique jobs in batches of {BATCH_SIZE}")
    print(f"[INFO] Industries : Engineering / Computing (alternating per batch)")
    print(f"[INFO] LLM        : {HF_MODEL}")
    print(f"[INFO] Embed      : sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Store      : Firestore / {FIRESTORE_JOBS_COLLECTION}")
    print(f"[INFO] Retries    : {MAX_RETRIES} per batch\n")

    all_jobs  = []
    batch_num = 0
    ind_idx   = 0

    while len(all_jobs) < JOB_COUNT:
        industry = industries_cycle[ind_idx % 2]
        ind_idx += 1

        remaining = industry_targets[industry] - industry_counts[industry]
        if remaining <= 0:
            industry = industries_cycle[ind_idx % 2]
            ind_idx += 1
            remaining = industry_targets[industry] - industry_counts[industry]
            if remaining <= 0:
                break

        current_size = min(BATCH_SIZE, remaining)
        batch_num   += 1

        print(f"[BATCH {batch_num}] {industry} — requesting {current_size} unique job(s) …")

        jobs = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt     = build_prompt(current_size, existing_titles, industry)
                raw_output = call_llama(prompt)
                jobs       = parse_jobs(raw_output, current_size, industry,
                                        existing_titles)
                if jobs:
                    break
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] 0 valid unique jobs — retrying …")

            except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] Error: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [SKIP] Batch {batch_num} failed after {MAX_RETRIES} attempts.")

        if not jobs:
            continue

        jobs = embed_jobs(jobs)
        print(f"  [INFO] Seeding batch {batch_num} into Firestore …")
        seed_jobs(jobs, existing_docs)

        for j in jobs:
            industry_counts[j["industry"]] += 1

        all_jobs.extend(jobs)
        existing_titles.extend([j["title"] for j in jobs])

        time.sleep(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[SUCCESS] {len(all_jobs)}/{JOB_COUNT} unique job documents written.")
    print(f"[INFO]    Engineering jobs : {industry_counts['Engineering']}")
    print(f"[INFO]    Computing jobs   : {industry_counts['Computing']}")
    print(f"[INFO]    Total in Firestore now : {len(existing_docs)}")
    print(f"[INFO]    Each document includes a 384-dim SBERT embedding.")
    print(f"[INFO]    Collection : {FIRESTORE_JOBS_COLLECTION}")
    print(f"[INFO]    Model used : {HF_MODEL}")