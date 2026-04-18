"""
dataset_generator/generate_dataset.py

Fully dynamic job dataset generator using Llama-3.1-8B-Instruct
via HuggingFace's unified inference router (free tier).

- No third-party provider needed — just your HF token
- Uses OpenAI-compatible client pointed at router.huggingface.co
- Generates TECH-SPECIFIC job titles (e.g. "Python Software Engineer")
- Skills are tailored to match the exact tech stack in the title
- Covers both Computing AND Engineering fields
- Semantic deduplication via RoBERTa embeddings
- Uploads to Firestore

Run:
    cd resume_analyzer/backend
    source venv/bin/activate
    cd ../dataset_generator
    python generate_dataset.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

# Bootstrap path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import HUGGINGFACE_API_TOKEN, LLM_MODEL_ID, LLM_BASE_URL
from services.firebase_service import init_firebase, fetch_all_jobs
from utils.skill_normalizer import SkillNormalizer
from ai_modules.embedding_engine import EmbeddingEngine

normalizer = SkillNormalizer()
embedder   = EmbeddingEngine()

if not HUGGINGFACE_API_TOKEN:
    raise RuntimeError(
        "HUGGINGFACE_API_TOKEN is not set. "
        "Get yours at https://huggingface.co/settings/tokens — "
        "make sure 'Make calls to Inference Providers' permission is enabled."
    )

# OpenAI-compatible client pointed at HuggingFace's unified router.
# Your HF token is the only credential needed — no Featherless account.
_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=HUGGINGFACE_API_TOKEN,
)

# Cosine similarity threshold for deduplication.
# Set high (0.97) so tech-specific variants like
# "Python Software Engineer" and "Java Software Engineer" are kept separate.
DEDUP_THRESHOLD = 0.97


# ---------------------------------------------------------------------------
# Firestore ID sanitization
# ---------------------------------------------------------------------------

def sanitize_firestore_id(title: str) -> str:
    """
    Sanitize a job title for safe use as a Firestore document ID.

    Firestore document IDs must not contain forward slashes ('/') as they are
    interpreted as path separators, causing 'uneven path elements' errors.
    """
    if not title:
        return "unknown"
    safe = re.sub(r'[/\\.]', '-', title)
    safe = re.sub(r'-{2,}', '-', safe)
    safe = safe.strip('-')
    return safe or "unknown"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        retries: int = 4,
) -> str:
    """
    Call Llama-3.1-8B-Instruct via HuggingFace's router.
    Retries with exponential backoff on transient errors.
    """
    for attempt in range(1, retries + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.90,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "loading" in err_str or "503" in err_str or "502" in err_str:
                wait = 30 * attempt
                logger.warning(
                    f"  Model loading / gateway error — waiting {wait}s "
                    f"(attempt {attempt}/{retries})"
                )
                time.sleep(wait)
            elif "429" in err_str or "rate" in err_str:
                wait = 60 * attempt
                logger.warning(
                    f"  Rate limited — waiting {wait}s (attempt {attempt}/{retries})"
                )
                time.sleep(wait)
            else:
                logger.error(f"  LLM error (attempt {attempt}/{retries}): {e}")
                time.sleep(5 * attempt)

    logger.error("  All retries exhausted.")
    return ""


def _extract_json(text: str):
    """Extract the first valid JSON array from LLM output."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try stripping trailing commas (common LLM mistake)
            cleaned = re.sub(r",\s*([}\]])", r"\1", match.group())
            try:
                return json.loads(cleaned)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SYSTEM_RECRUITER = (
    "You are a senior technical recruiter with deep expertise across all "
    "Computing, Software, Hardware, and Engineering disciplines. "
    "You always respond with valid JSON only — no markdown, no explanation, "
    "no preamble. Output the JSON array and nothing else."
)


# ---------------------------------------------------------------------------
# Step 1 — Discover job families
# ---------------------------------------------------------------------------

def discover_job_families() -> list[str]:
    """
    Ask the LLM to freely enumerate job role families across ALL Computing
    and Engineering disciplines. No preset list — fully LLM-driven.
    """
    user = """List all major job role families across Computing and Engineering.

Cover ALL of these areas:
- Computing: software development, data science, AI/ML, cloud, cybersecurity,
  networking, embedded systems, game development, blockchain, QA/testing, etc.
- Engineering: electrical, mechanical, civil, chemical, aerospace, biomedical,
  industrial, environmental, structural, systems, automotive, robotics,
  manufacturing, petroleum, materials, nuclear, etc.

Return ONLY a JSON array of strings. Each string = a short role family name (2-5 words).
Aim for 50-70 distinct families. No duplicates. No markdown. No explanation.

Example: ["Software Engineering", "Data Science", "Electrical Engineering"]"""

    logger.info("Step 1: Discovering job families dynamically via LLM...")
    raw = _call_llm(SYSTEM_RECRUITER, user, max_tokens=1000)
    if not raw:
        logger.error("Failed to get job families from LLM.")
        return []

    families = _extract_json(raw)
    if not isinstance(families, list):
        logger.error(f"Could not parse job families. Raw output:\n{raw[:400]}")
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for f in families:
        if isinstance(f, str):
            f = f.strip().strip('"').strip()
            low = f.lower()
            if f and low not in seen and 3 < len(f) < 80:
                seen.add(low)
                cleaned.append(f)

    logger.success(f"  Discovered {len(cleaned)} job families.")
    return cleaned


# ---------------------------------------------------------------------------
# Tech stacks to generate per computing family
# ---------------------------------------------------------------------------

COMPUTING_TECH_STACKS: dict[str, list[str]] = {
    "Software Engineering": [
        "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust",
        "C++", "C#", "Ruby", "Kotlin", "Swift", "PHP",
    ],
    "Backend Engineering": [
        "Python", "Java", "Node.js", "Go", "Ruby on Rails",
        "Django", "Spring Boot", "FastAPI", "Kotlin", "Scala",
    ],
    "Frontend Engineering": [
        "React", "Vue.js", "Angular", "Next.js", "Svelte",
        "TypeScript", "JavaScript",
    ],
    "Full Stack Engineering": [
        "React and Node.js", "Vue.js and Django", "Angular and Spring Boot",
        "Next.js and FastAPI", "Ruby on Rails",
    ],
    "Data Science": [
        "Python", "R", "Scala", "SQL",
    ],
    "Machine Learning Engineering": [
        "Python TensorFlow", "Python PyTorch", "Python Scikit-learn",
        "Java Weka", "Scala Spark ML",
    ],
    "AI Engineering": [
        "Python OpenAI API", "Python LangChain", "Python Hugging Face",
        "JavaScript TensorFlow.js",
    ],
    "Data Engineering": [
        "Python Apache Spark", "Scala Apache Spark", "SQL dbt",
        "Java Apache Kafka", "Python Apache Airflow",
    ],
    "Cloud Engineering": [
        "AWS", "Azure", "Google Cloud Platform", "Multi-Cloud Terraform",
    ],
    "DevOps Engineering": [
        "Kubernetes", "Docker", "Jenkins", "GitHub Actions",
        "Ansible", "Terraform",
    ],
    "Cybersecurity Engineering": [
        "Penetration Testing Kali Linux", "Cloud Security AWS",
        "Network Security", "Application Security",
    ],
    "Mobile Development": [
        "Android Kotlin", "iOS Swift", "React Native",
        "Flutter", "Android Java",
    ],
    "Embedded Systems Engineering": [
        "C and C++", "Rust", "Python MicroPython", "RTOS",
    ],
    "Game Development": [
        "Unity C#", "Unreal Engine C++", "Godot GDScript",
    ],
    "Blockchain Engineering": [
        "Solidity Ethereum", "Rust Solana", "JavaScript Web3.js",
    ],
    "QA Engineering": [
        "Selenium Python", "Cypress JavaScript", "JUnit Java",
        "Playwright", "Appium",
    ],
    "Database Engineering": [
        "PostgreSQL", "MySQL", "MongoDB", "Cassandra",
        "Redis", "Oracle", "SQL Server",
    ],
    "Site Reliability Engineering": [
        "Python and Kubernetes", "Go and Prometheus", "Python and Terraform",
    ],
}


def _get_tech_stacks_for_family(family: str) -> list[str]:
    """Return tech stacks for this family, or [] for non-computing families."""
    for key, stacks in COMPUTING_TECH_STACKS.items():
        if key.lower() in family.lower() or family.lower() in key.lower():
            return stacks
    return []


# ---------------------------------------------------------------------------
# Step 2 — Generate roles per family
# ---------------------------------------------------------------------------

def generate_roles_for_family(family: str) -> list[dict]:
    """
    Dispatch to the appropriate generator based on whether the family
    has a known tech stack list or not.
    """
    stacks = _get_tech_stacks_for_family(family)
    if stacks:
        return _generate_roles_with_stacks(family, stacks)
    return _generate_roles_generic(family)


def _generate_roles_with_stacks(family: str, stacks: list[str]) -> list[dict]:
    """
    For each tech stack generate one highly specific role with skills
    tailored to that exact stack. Batches 8 stacks per LLM call.
    """
    all_roles: list[dict] = []
    batch_size = 8

    for i in range(0, len(stacks), batch_size):
        batch = stacks[i : i + batch_size]

        user = f"""For the job family "{family}", generate one specific job role per technology stack below.

Technology stacks:
{json.dumps(batch, indent=2)}

Return ONLY a JSON array — one object per stack — with exactly these fields:
  "job_title"      : Role name WITH the technology in the title.
                     Format: "[Technology] [Role]"
                     Examples: "Python Software Engineer", "React Frontend Engineer",
                               "AWS Cloud Architect", "PostgreSQL Database Administrator"
                     IMPORTANT: Do NOT use slash characters in titles.
                                Write "React Node.js Full Stack Engineer" not "React/Node.js".
  "technology"     : The exact technology stack string from the list above.
  "skills"         : Array of exactly 10 skills SPECIFIC to this technology stack.
                     Every skill must relate directly to the named technology.
                     Example for Python: ["Python 3.x", "Django", "FastAPI", "SQLAlchemy",
                       "pytest", "asyncio", "Celery", "NumPy", "Pandas", "Docker"]
                     Example for Java:  ["Java 17+", "Spring Boot", "Maven", "JUnit 5",
                       "Hibernate", "Jakarta EE", "Kafka", "JDBC", "Mockito", "IntelliJ IDEA"]
  "certifications" : Array of 2-3 real, recognized certifications. Use [] if none exist.

Rules:
- No generic skills like "problem solving" or "communication"
- No duplicate job titles within the array
- Return exactly {len(batch)} objects

No markdown. No explanation. JSON array only."""

        raw = _call_llm(SYSTEM_RECRUITER, user, max_tokens=1400)
        if not raw:
            logger.warning(f"  No LLM response for '{family}' batch {i // batch_size + 1}")
            continue

        parsed = _extract_json(raw)
        if not isinstance(parsed, list):
            logger.warning(
                f"  Could not parse roles for '{family}' batch {i // batch_size + 1}"
            )
            continue

        for item in parsed:
            if not isinstance(item, dict):
                continue
            title  = str(item.get("job_title", "")).strip()
            skills = item.get("skills", [])
            certs  = item.get("certifications", [])
            tech   = str(item.get("technology", "")).strip()
            if not title or not skills:
                continue
            all_roles.append({
                "raw_title":        title,
                "normalized_title": title,
                "skills":           [str(s).strip() for s in skills if isinstance(s, str) and s.strip()],
                "certifications":   [str(c).strip() for c in certs  if isinstance(c, str) and c.strip()],
                "family":           family,
                "technology":       tech,
            })

        time.sleep(2)  # be gentle with the rate limiter

    return all_roles


def _generate_roles_generic(family: str) -> list[dict]:
    """
    For non-computing families (mechanical, civil, biomedical, etc.),
    generate tool/discipline-specific variants such as
    "AutoCAD Mechanical Engineer" or "ANSYS Structural Engineer".
    """
    user = f"""For the job family "{family}", generate 6 distinct real-world job roles.

Each role should include the primary tool, software, or specialization in the title.

Examples:
- "Mechanical Engineering" → "AutoCAD Mechanical Engineer", "ANSYS FEA Engineer"
- "Electrical Engineering" → "PLC Automation Engineer", "FPGA Design Engineer"
- "Civil Engineering"      → "AutoCAD Civil 3D Engineer", "Geotechnical Engineer"

Return ONLY a JSON array. Each object must have exactly:
  "job_title"      : Specific role name (include tool/specialization).
                     IMPORTANT: No slash characters in titles.
  "skills"         : Array of exactly 10 specific technical skills for this role.
                     All skills must match the specialization in the title.
  "certifications" : Array of 2-3 real recognized certifications. Use [] if none.

Rules:
- Technical skills only — no soft skills
- No duplicate job titles
- No markdown, no explanation — JSON array only."""

    raw = _call_llm(SYSTEM_RECRUITER, user, max_tokens=1100)
    if not raw:
        return []

    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        logger.warning(f"  Could not parse generic roles for '{family}'")
        return []

    roles: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        title  = str(item.get("job_title", "")).strip()
        skills = item.get("skills", [])
        certs  = item.get("certifications", [])
        if not title or not skills:
            continue
        roles.append({
            "raw_title":        title,
            "normalized_title": title,
            "skills":           [str(s).strip() for s in skills if isinstance(s, str) and s.strip()],
            "certifications":   [str(c).strip() for c in certs  if isinstance(c, str) and c.strip()],
            "family":           family,
            "technology":       "",
        })

    return roles


# ---------------------------------------------------------------------------
# Step 3 — Semantic deduplication
# ---------------------------------------------------------------------------

def semantic_deduplicate(roles: list[dict]) -> list[dict]:
    """
    Remove near-duplicate job roles using RoBERTa cosine similarity.

    Threshold is 0.97 (very strict) so that tech-specific variants like
    "Python Software Engineer" and "Java Software Engineer" are NOT merged.
    Only truly identical titles (e.g. generated twice from different families)
    get collapsed. Skills from merged duplicates are combined.
    """
    if not roles:
        return []

    logger.info(f"  Embedding {len(roles)} titles for semantic deduplication...")
    titles     = [r["normalized_title"] for r in roles]
    embeddings = embedder.embed_batch(titles)
    sim_matrix = cosine_similarity(embeddings)

    kept_indices: list[int] = []
    dropped:      set[int]  = set()
    merge_log:    list[str] = []

    for i in range(len(roles)):
        if i in dropped:
            continue
        for j in range(i + 1, len(roles)):
            if j in dropped:
                continue
            sim = float(sim_matrix[i][j])
            if sim >= DEDUP_THRESHOLD:
                # Merge j's data into i
                roles[i]["skills"] = list(
                    set(roles[i]["skills"] + roles[j]["skills"])
                )[:15]
                roles[i]["certifications"] = list(
                    set(roles[i]["certifications"] + roles[j]["certifications"])
                )[:4]
                dropped.add(j)
                merge_log.append(
                    f"    '{roles[j]['normalized_title']}' → "
                    f"'{roles[i]['normalized_title']}' (sim={sim:.3f})"
                )
        kept_indices.append(i)

    if merge_log:
        logger.info("  Merges performed:")
        for entry in merge_log:
            logger.info(entry)

    unique_roles = [roles[i] for i in kept_indices]
    logger.success(
        f"  Done: {len(unique_roles)} unique roles "
        f"({len(roles) - len(unique_roles)} duplicates merged)"
    )
    return unique_roles


# ---------------------------------------------------------------------------
# Step 4 — Upload to Firestore
# ---------------------------------------------------------------------------

def normalize_and_upload(roles: list[dict]) -> int:
    """
    Upload each role to Firestore.
    Titles are kept as-is (e.g. "Python Software Engineer").
    Only illegal Firestore ID characters are sanitized.
    """
    import firebase_admin.firestore as _fs

    uploaded = 0
    db = _fs.client()

    for role in roles:
        title = role["normalized_title"].strip()
        if not title or len(title) < 3:
            continue

        skills = normalizer.deduplicate_skills(role.get("skills", []))
        if not skills:
            logger.warning(f"  Skipping '{title}' — no valid skills after normalization.")
            continue

        certs = sorted({
            c.strip() for c in role.get("certifications", [])
            if isinstance(c, str) and len(c.strip()) > 3
        })

        doc_id   = sanitize_firestore_id(title)
        doc_data = {
            "job_title":      title,
            "skills":         skills,
            "certifications": certs,
        }
        if role.get("technology"):
            doc_data["technology"] = role["technology"]
        if role.get("family"):
            doc_data["family"] = role["family"]

        try:
            db.collection("jobs").document(doc_id).set(doc_data)
            logger.info(f"  ok  {title}  ({len(skills)} skills, {len(certs)} certs)")
            uploaded += 1
        except Exception as e:
            logger.error(f"  Failed to upload '{title}': {e}")

    return uploaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 65)
    logger.info("  Resume Analyzer — Dynamic Dataset Generator")
    logger.info(f"  Model    : {LLM_MODEL_ID}")
    logger.info(f"  Endpoint : {LLM_BASE_URL}")
    logger.info("  Provider : HuggingFace router (no third-party account)")
    logger.info("  Mode     : Tech-specific titles + matching skill sets")
    logger.info("=" * 65)

    logger.info("\nInitializing Firebase...")
    init_firebase()

    existing              = fetch_all_jobs()
    existing_titles_lower = {j["job_title"].lower() for j in existing}
    logger.info(f"Existing Firestore entries: {len(existing_titles_lower)}")

    # Step 1 — discover families
    families = discover_job_families()
    if not families:
        logger.error("Could not discover job families. Check your HUGGINGFACE_API_TOKEN.")
        return

    logger.info(f"\nJob families discovered ({len(families)} total):")
    for f in families:
        logger.info(f"  - {f}")

    # Step 2 — generate roles
    logger.info("\nGenerating tech-specific roles for each family...")
    all_roles: list[dict] = []

    for idx, family in enumerate(families, 1):
        stacks     = _get_tech_stacks_for_family(family)
        stack_note = f"({len(stacks)} stacks)" if stacks else "(discipline variants)"
        logger.info(f"\n[{idx}/{len(families)}] {family} {stack_note}")

        roles = generate_roles_for_family(family)
        if roles:
            all_roles.extend(roles)
            logger.info(f"  Generated {len(roles)} roles")
        else:
            logger.warning(f"  No roles generated for '{family}' — skipping")

        time.sleep(2)

    logger.info(f"\nTotal raw roles collected: {len(all_roles)}")

    # Step 3 — deduplicate
    logger.info("\nRunning semantic deduplication (RoBERTa, threshold=0.97)...")
    unique_roles = semantic_deduplicate(all_roles)

    # Filter roles already in Firestore
    new_roles = [
        r for r in unique_roles
        if r["normalized_title"].lower() not in existing_titles_lower
    ]
    logger.info(f"Already in Firestore (skipping): {len(unique_roles) - len(new_roles)}")
    logger.info(f"New roles to upload            : {len(new_roles)}")

    # Step 4 — upload
    logger.info("\nUploading to Firestore...")
    count = normalize_and_upload(new_roles)

    # Save local backup
    backup_data = {
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_used":         LLM_MODEL_ID,
        "endpoint":           LLM_BASE_URL,
        "families_discovered": families,
        "stats": {
            "raw_roles_generated":   len(all_roles),
            "after_deduplication":   len(unique_roles),
            "already_in_firestore":  len(unique_roles) - len(new_roles),
            "newly_uploaded":        count,
        },
        "roles": [
            {
                "job_title":      r["normalized_title"],
                "technology":     r.get("technology", ""),
                "family":         r.get("family", ""),
                "skills":         r["skills"],
                "certifications": r["certifications"],
            }
            for r in unique_roles
        ],
    }
    backup_path = Path(__file__).parent / "generated_dataset.json"
    with open(backup_path, "w") as f:
        json.dump(backup_data, f, indent=2)

    logger.info("\n" + "=" * 65)
    logger.success("Dataset generation complete!")
    logger.info(f"  Model              : {LLM_MODEL_ID}")
    logger.info(f"  Families           : {len(families)}")
    logger.info(f"  Raw roles          : {len(all_roles)}")
    logger.info(f"  After dedup        : {len(unique_roles)}")
    logger.info(f"  Newly uploaded     : {count}")
    logger.info(f"  Local backup       : {backup_path}")
    logger.info("=" * 65)
    logger.info("\nNext step: restart the backend to rebuild the FAISS index.")


if __name__ == "__main__":
    main()