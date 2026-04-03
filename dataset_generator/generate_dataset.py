"""
dataset_generator/generate_dataset.py

Fully dynamic job dataset generator using Meta-Llama-3-8B-Instruct
via HuggingFace InferenceClient (free tier, hf-inference provider).

- NO preset job categories - the LLM discovers them itself
- Covers both Computing AND Engineering fields
- Semantic deduplication via RoBERTa embeddings
- Normalizes job titles (strips tech qualifiers like "Python Software Engineer" -> "Software Engineer")
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

import numpy as np
from huggingface_hub import InferenceClient
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

# Bootstrap path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import HUGGINGFACE_API_TOKEN, LLM_MODEL_ID
from services.firebase_service import init_firebase, upsert_job, fetch_all_jobs
from utils.skill_normalizer import SkillNormalizer
from ai_modules.embedding_engine import EmbeddingEngine

normalizer = SkillNormalizer()
embedder = EmbeddingEngine()

if not HUGGINGFACE_API_TOKEN:
    raise RuntimeError(
        "HUGGINGFACE_API_TOKEN is not set. "
        "Get yours at https://huggingface.co/settings/tokens"
    )

# HuggingFace InferenceClient - the correct modern API
_client = InferenceClient(
    model=LLM_MODEL_ID,
    token=HUGGINGFACE_API_TOKEN,
)

# Cosine similarity threshold for deduplication
DEDUP_THRESHOLD = 0.92


def sanitize_firestore_id(title: str) -> str:
    """
    Sanitize a job title for safe use as a Firestore document ID.

    Firestore document IDs must not contain forward slashes ('/') as they are
    interpreted as path separators, causing 'uneven path elements' errors.
    Other problematic characters (backslash, dot) are also replaced.

    The original title is always preserved in the document's data fields.

    Args:
        title: Raw job title string (e.g. "Ui/Ux Designer")

    Returns:
        A sanitized string safe to use as a Firestore document ID
        (e.g. "Ui-Ux Designer")
    """
    if not title:
        return "unknown"
    # Replace path separators and dot (reserved in Firestore) with a dash
    safe = re.sub(r'[/\\.]', '-', title)
    # Collapse multiple consecutive dashes into one
    safe = re.sub(r'-{2,}', '-', safe)
    # Strip leading/trailing dashes
    safe = safe.strip('-')
    # Firestore doc IDs cannot be empty strings
    return safe or "unknown"


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1000, retries: int = 4) -> str:
    """
    Call LLM via HuggingFace InferenceClient chat_completion.
    Uses the hf-inference provider (free tier) automatically.
    Retries with backoff on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            response = _client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.5,
                top_p=0.92,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if "loading" in err_str.lower() or "503" in err_str:
                wait = 30 * attempt
                logger.warning(f"  Model loading - waiting {wait}s (attempt {attempt}/{retries})")
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
            cleaned = re.sub(r",\s*([}\]])", r"\1", match.group())
            try:
                return json.loads(cleaned)
            except Exception:
                pass
    return None


SYSTEM_RECRUITER = (
    "You are a senior technical recruiter with deep expertise across all "
    "Computing, Software, Hardware, and Engineering disciplines. "
    "You always respond with valid JSON only - no markdown, no explanation."
)


def discover_job_families() -> list[str]:
    """
    Ask the LLM to freely enumerate job role families across ALL Computing
    and Engineering disciplines. No preset list - fully LLM-driven.
    """
    user = """List all major job role families that exist across the Computing and Engineering industries.

Cover ALL of these broad areas (and any others you know of):
- Computing: software development, data science, AI/ML, cloud, cybersecurity, networking, embedded systems, game development, blockchain, QA/testing, etc.
- Engineering: electrical, mechanical, civil, chemical, aerospace, biomedical, industrial, environmental, structural, systems, automotive, robotics, manufacturing, petroleum, materials, nuclear, etc.

Return ONLY a JSON array of strings. Each string is a short role family name (2-5 words).
Requirements:
- Aim for 50-70 distinct families
- No duplicates
- No markdown, no explanation - only the JSON array

Example format:
["Software Engineering", "Data Science", "Electrical Engineering", "Civil Engineering", "Robotics Engineering"]"""

    logger.info("Step 1: Asking LLM to dynamically discover job families...")
    raw = _call_llm(SYSTEM_RECRUITER, user, max_tokens=1000)
    if not raw:
        logger.error("Failed to get job families from LLM.")
        return []

    families = _extract_json(raw)
    if not isinstance(families, list):
        logger.error(f"Could not parse job families. Raw:\n{raw[:400]}")
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

    logger.success(f"  Discovered {len(cleaned)} job families dynamically.")
    return cleaned


def generate_roles_for_family(family: str) -> list[dict]:
    """
    Generate 5 concrete job roles for a given role family.
    Titles must be clean - technologies go in skills, not titles.
    """
    user = f"""For the job family "{family}", generate 5 distinct, real-world job roles.

Return ONLY a JSON array. Each object must have exactly:
  "job_title"       : Clean, canonical role name. IMPORTANT: Do NOT put technologies in the title.
                      Write "Software Engineer" NOT "Python Software Engineer".
                      Write "Data Scientist" NOT "TensorFlow Data Scientist".
  "skills"          : Array of exactly 10 specific, technical skills required for this role.
  "certifications"  : Array of 2-3 real, recognized certifications. Use empty array [] if none standard.

Rules:
- Job titles reflect the role, not the technology stack
- Skills must be technical and specific (not soft skills)
- No duplicate job titles in this list
- Certifications must be real and widely recognized

Return only the JSON array. No markdown. No explanation."""

    raw = _call_llm(SYSTEM_RECRUITER, user, max_tokens=900)
    if not raw:
        return []

    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        logger.warning(f"  Could not parse roles for '{family}'")
        return []

    roles: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        title = str(item.get("job_title", "")).strip()
        skills = item.get("skills", [])
        certs = item.get("certifications", [])
        if not title or not skills:
            continue
        roles.append({
            "raw_title": title,
            "skills": [str(s).strip() for s in skills if isinstance(s, str) and s.strip()],
            "certifications": [str(c).strip() for c in certs if isinstance(c, str) and c.strip()],
            "family": family,
        })

    return roles


def normalize_titles_batch(roles: list[dict]) -> list[dict]:
    """
    Ask the LLM to normalize a batch of raw job titles to canonical form.
    "Python Software Engineer" -> "Software Engineer"
    "Senior ML Engineer (NLP)" -> "Machine Learning Engineer"
    Processes in batches of 20.
    """
    all_titles = [r["raw_title"] for r in roles]
    normalized_map: dict[str, str] = {}
    batch_size = 20
    total_batches = (len(all_titles) + batch_size - 1) // batch_size
    logger.info(f"  Normalizing {len(all_titles)} titles in {total_batches} batches...")

    for i in range(0, len(all_titles), batch_size):
        batch = all_titles[i : i + batch_size]
        batch_num = i // batch_size + 1

        user = f"""Normalize these job titles to their clean, canonical, technology-neutral form.

Rules:
- Remove ALL technology qualifiers from titles (Python, Java, AWS, React, etc. belong in skills not titles)
- Remove seniority prefixes (Senior, Junior, Lead, Principal, Associate -> just the base role)
- Remove parenthetical qualifiers like "(NLP)", "(Backend)"
- Keep engineering discipline in the title (e.g. "Electrical Engineer" stays as-is)
- Use widely-recognized standard industry titles
- IMPORTANT: Do NOT use slash characters (/) in titles. Write "UI UX Designer" not "UI/UX Designer".

Titles to normalize:
{json.dumps(batch, indent=2)}

Return ONLY a JSON array of normalized title strings, in the same order as the input.
Array must have exactly {len(batch)} elements. No markdown. No explanation."""

        raw = _call_llm(SYSTEM_RECRUITER, user, max_tokens=400)
        normalized_batch = _extract_json(raw)

        if isinstance(normalized_batch, list) and len(normalized_batch) == len(batch):
            for original, norm in zip(batch, normalized_batch):
                if isinstance(norm, str) and norm.strip():
                    normalized_map[original] = norm.strip().title()
                else:
                    normalized_map[original] = normalizer.normalize_job_title(original)
        else:
            logger.warning(f"  Batch {batch_num}: LLM normalization failed, using rule-based fallback.")
            for title in batch:
                normalized_map[title] = normalizer.normalize_job_title(title)

        logger.debug(f"  Batch {batch_num}/{total_batches} done.")
        time.sleep(1)

    for role in roles:
        role["normalized_title"] = normalized_map.get(
            role["raw_title"],
            normalizer.normalize_job_title(role["raw_title"])
        )
    return roles


def semantic_deduplicate(roles: list[dict]) -> list[dict]:
    """
    Remove duplicate job roles using RoBERTa embedding cosine similarity.
    Catches what string matching misses:
    - "Software Engineer" vs "Software Developer"
    - "ML Engineer" vs "Machine Learning Engineer"
    When duplicates found, skills are MERGED so no info is lost.
    """
    if not roles:
        return []

    logger.info(f"  Embedding {len(roles)} titles for semantic deduplication...")
    titles = [r["normalized_title"] for r in roles]
    embeddings = embedder.embed_batch(titles)
    sim_matrix = cosine_similarity(embeddings)

    kept_indices: list[int] = []
    dropped: set[int] = set()
    merge_log: list[str] = []

    for i in range(len(roles)):
        if i in dropped:
            continue
        for j in range(i + 1, len(roles)):
            if j in dropped:
                continue
            sim = float(sim_matrix[i][j])
            if sim >= DEDUP_THRESHOLD:
                # Merge j's skills into i
                roles[i]["skills"] = list(set(roles[i]["skills"] + roles[j]["skills"]))[:15]
                roles[i]["certifications"] = list(set(roles[i]["certifications"] + roles[j]["certifications"]))[:4]
                dropped.add(j)
                merge_log.append(
                    f"    '{roles[j]['normalized_title']}' -> '{roles[i]['normalized_title']}' (sim={sim:.3f})"
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


def normalize_and_upload(roles: list[dict]) -> int:
    """Final normalization pass then upload each role to Firestore.

    Writes directly to Firestore using a sanitized document ID so that
    titles containing '/' (e.g. "UI/UX Designer") don't cause Firestore's
    "uneven number of path elements" error. The original title is preserved
    inside the document body.
    """
    import firebase_admin.firestore as _fs

    uploaded = 0
    db = _fs.client()

    for role in roles:
        title = normalizer.normalize_job_title(role["normalized_title"])
        if not title or len(title) < 3:
            continue
        skills = normalizer.deduplicate_skills(role.get("skills", []))
        if not skills:
            logger.warning(f"  Skipping '{title}' - no valid skills after normalization.")
            continue
        certs = sorted(set(
            c.strip() for c in role.get("certifications", [])
            if isinstance(c, str) and len(c.strip()) > 3
        ))

        # Sanitize for Firestore document ID: '/' is a path separator,
        # causing "uneven path elements" errors (e.g. "UI/UX Designer").
        # Original title is preserved in the document data untouched.
        doc_id = sanitize_firestore_id(title)

        try:
            db.collection("jobs").document(doc_id).set({
                "job_title": title,
                "skills": skills,
                "certifications": certs,
            })
            logger.info(f"  ok {title}  ({len(skills)} skills, {len(certs)} certs)")
            uploaded += 1
        except Exception as e:
            logger.error(f"  Failed to upload '{title}': {e}")
    return uploaded


def main():
    logger.info("=" * 65)
    logger.info("  Resume Analyzer - Dynamic Dataset Generator")
    logger.info("  Model: " + LLM_MODEL_ID)
    logger.info("  Fields: Computing + Engineering (all disciplines)")
    logger.info("  Preset categories: NONE - fully LLM-driven discovery")
    logger.info("=" * 65)

    logger.info("\nInitializing Firebase...")
    init_firebase()

    existing = fetch_all_jobs()
    existing_titles_lower = {j["job_title"].lower() for j in existing}
    logger.info(f"Existing Firestore entries: {len(existing_titles_lower)}")

    # Step 1: LLM discovers job families dynamically
    families = discover_job_families()
    if not families:
        logger.error("Could not discover job families. Check your HUGGINGFACE_API_TOKEN.")
        return

    logger.info(f"\nJob families discovered ({len(families)} total):")
    for f in families:
        logger.info(f"  - {f}")

    # Step 2: Generate roles per family
    logger.info(f"\nGenerating roles for each family...")
    all_roles: list[dict] = []

    for idx, family in enumerate(families, 1):
        logger.info(f"\n[{idx}/{len(families)}] {family}")
        roles = generate_roles_for_family(family)
        if roles:
            all_roles.extend(roles)
            logger.info(f"  Generated {len(roles)} roles")
        else:
            logger.warning(f"  No roles generated for '{family}' - skipping")
        time.sleep(2)  # Rate limit buffer

    logger.info(f"\nTotal raw roles collected: {len(all_roles)}")

    # Step 3: Normalize titles via LLM
    logger.info("\nNormalizing job titles via LLM...")
    all_roles = normalize_titles_batch(all_roles)

    # Step 4: Semantic deduplication
    logger.info("\nRunning semantic deduplication (RoBERTa embeddings)...")
    unique_roles = semantic_deduplicate(all_roles)

    # Filter already-existing entries
    new_roles = [
        r for r in unique_roles
        if normalizer.normalize_job_title(r["normalized_title"]).lower()
           not in existing_titles_lower
    ]
    logger.info(f"Already in Firestore (skipping): {len(unique_roles) - len(new_roles)}")
    logger.info(f"New roles to upload: {len(new_roles)}")

    # Step 5: Upload
    logger.info("\nUploading to Firestore...")
    count = normalize_and_upload(new_roles)

    # Save backup
    backup_data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_used": LLM_MODEL_ID,
        "families_discovered": families,
        "stats": {
            "raw_roles_generated": len(all_roles),
            "after_deduplication": len(unique_roles),
            "already_in_firestore": len(unique_roles) - len(new_roles),
            "newly_uploaded": count,
        },
        "roles": [
            {
                "job_title": normalizer.normalize_job_title(r["normalized_title"]),
                "skills": r["skills"],
                "certifications": r["certifications"],
                "family": r.get("family", ""),
            }
            for r in unique_roles
        ],
    }
    backup_path = Path(__file__).parent / "generated_dataset.json"
    with open(backup_path, "w") as f:
        json.dump(backup_data, f, indent=2)

    logger.info("\n" + "=" * 65)
    logger.success("Dataset generation complete!")
    logger.info(f"  Model used              : {LLM_MODEL_ID}")
    logger.info(f"  Families discovered     : {len(families)}")
    logger.info(f"  Raw roles generated     : {len(all_roles)}")
    logger.info(f"  After semantic dedup    : {len(unique_roles)}")
    logger.info(f"  Newly uploaded          : {count}")
    logger.info(f"  Local backup            : {backup_path}")
    logger.info("=" * 65)
    logger.info("\nNext step: restart the backend to rebuild the FAISS index.")


if __name__ == "__main__":
    main()