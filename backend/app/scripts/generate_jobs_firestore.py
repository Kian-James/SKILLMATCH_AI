import os
import re
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from app.repositories.firestore.client import db

# -------------------------
# ENV
# -------------------------
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY missing")

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

hf = InferenceClient(model=MODEL_ID, token=HF_API_KEY)

CATEGORIES = ["Computing", "Engineering"]

# -------------------------
# PARSER
# -------------------------
def parse_jobs_text(text: str):
    jobs = []
    blocks = text.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or len(lines) < 2:
            continue

        title_match = re.match(r"Job:\s*(.+)", lines[0])
        if not title_match:
            continue
        title = title_match.group(1).strip()

        # Skills
        skills_line = next((l for l in lines if l.lower().startswith("skills:")), "")
        skills_match = re.search(r"\[(.*)\]", skills_line)
        if skills_match:
            skills = [s.strip() for s in skills_match.group(1).split(",") if s.strip()]
        else:
            skills_text = skills_line.split(":", 1)[-1].strip()
            skills = [s.strip() for s in skills_text.split(",") if s.strip()]

        # Certifications
        certs_line = next((l for l in lines if l.lower().startswith("certifications:")), "")
        certs_match = re.search(r"\[(.*)\]", certs_line)
        if certs_match:
            certifications = [c.strip() for c in certs_match.group(1).split(",") if c.strip()]
        else:
            certs_text = certs_line.split(":", 1)[-1].strip()
            certifications = [] if certs_text.lower() in ("none", "") else [c.strip() for c in certs_text.split(",") if c.strip()]

        jobs.append({
            "title": title,
            "skills": skills,
            "certifications": certifications
        })

    return jobs

# -------------------------
# CATEGORY HINTS
# -------------------------
CATEGORY_HINTS = {
    "Computing": (
        "Generate software/computing roles with diverse specializations, e.g., Flutter Developer, React Frontend Developer, "
        "Python Backend Developer, Full Stack Developer, DevOps Engineer, Cloud Architect, Cybersecurity Analyst, "
        "Data Scientist, AI/ML Engineer, Network Engineer, Database Administrator. "
        "IMPORTANT: Skills must be ONLY programming languages (Python, Java, C++, Dart, JavaScript)."
    ),
    "Engineering": (
        "Generate specific engineering roles like Electrical Engineer, Mechanical Engineer, Civil Engineer, "
        "Mechanical Design Engineer, Embedded Systems Engineer, Process Engineer."
    )
}


# -------------------------
# FETCH EXISTING JOBS
# -------------------------
def get_existing_jobs():
    col = db.collection("jobs")
    jobs = {}
    for doc in col.stream():
        data = doc.to_dict()
        jobs[data.get("title")] = {"doc_id": doc.id, "skills": set(data.get("skills", [])), "certifications": set(data.get("certifications", []))}
    return jobs

# -------------------------
# JOB GENERATION
# -------------------------
def generate_jobs(category: str, existing_titles: set, max_jobs: int = 30, retries: int = 5):
    jobs = []
    seen_titles = set(existing_titles)

    for attempt in range(retries):
        if len(jobs) >= max_jobs:
            break

        remaining = max_jobs - len(jobs)
        prompt = f"""
STRICT RULES:
- OUTPUT ONLY IN THIS FORMAT:
Job: <job title>
Skills: [skill1, skill2, skill3, ...]
Certifications: [cert1, cert2, ...]
- GENERATE UP TO {remaining} NEW JOBS for category "{category}"
- Category-specific hint: {CATEGORY_HINTS[category]}
- NO extra text
- DO NOT OMIT skills or certifications
- Separate jobs by a blank line
"""

        response = hf.chat.completions.create(
            messages=[
                {"role": "system", "content": "You generate structured job lists only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.6
        )

        raw_text = response.choices[0].message.content.strip()
        new_jobs = parse_jobs_text(raw_text)

        for job in new_jobs:
            if job["title"] in seen_titles:
                continue  # We'll merge later in Firestore update
            seen_titles.add(job["title"])
            jobs.append(job)
            if len(jobs) >= max_jobs:
                break

    return jobs

# -------------------------
# FIRESTORE UPSERT (MERGE SKILLS & CERTS, lowercase)
# -------------------------
def store_or_update_job(category, job, existing_jobs):
    col = db.collection("jobs")

    # Normalize to lowercase
    title_lower = job["title"].lower()
    skills_lower = [s.lower() for s in job["skills"]]
    certs_lower = [c.lower() for c in job["certifications"]]

    if title_lower in existing_jobs:
        # Merge skills & certifications
        existing = existing_jobs[title_lower]
        new_skills = existing["skills"].union(set(skills_lower))
        new_certs = existing["certifications"].union(set(certs_lower))
        col.document(existing["doc_id"]).update({
            "skills": list(new_skills),
            "certifications": list(new_certs)
        })
        print(f"🔄 Updated {title_lower}")
    else:
        # New job
        job_lower = {
            "title": title_lower,
            "skills": skills_lower,
            "certifications": certs_lower,
            "category": category.lower()
        }
        col.add(job_lower)
        print(f"✅ Stored {title_lower}")


# -------------------------
# MAIN LOOP (24/7)
# -------------------------
def main_loop(interval_seconds: int = 600):
    while True:
        print(f"\n🕒 Running job generator at {time.ctime()}")
        existing_jobs = get_existing_jobs()
        existing_titles = set(existing_jobs.keys())

        for category in CATEGORIES:
            print(f"\n=== Generating jobs for {category} ===")
            new_jobs = generate_jobs(category, existing_titles, max_jobs=30)
            if not new_jobs:
                print("⚠️ No jobs generated")
                continue

            for job in new_jobs:
                store_or_update_job(category, job, existing_jobs)

        print(f"\n💤 Sleeping for {interval_seconds} seconds before next run...\n")
        time.sleep(interval_seconds)  # wait before next generation

# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    main_loop(interval_seconds=600)  # run every 10 minutes
