import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import fitz
import easyocr
import torch
import re
import numpy as np
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# ── Hardware ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── SBERT: load from Hugging Face Hub (online, cached after first run) ────────
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# ── EasyOCR reader ────────────────────────────────────────────────────────────
reader = easyocr.Reader(["en"], gpu=(device == "cuda"))

# ── Poppler: resolved from system PATH ───────────────────────────────────────
POPPLER_PATH = None


def extract_text(file_path: str):
    """Stage 1: OCR Extraction with Quality Metric."""
    text = ""
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        text = " ".join([page.get_text("text") for page in doc])
        doc.close()

    if len(text.strip()) < 50:
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        text = " ".join(
            [" ".join(reader.readtext(np.array(img), detail=0)) for img in images]
        )

    clean_text = re.sub(r"\s+", " ", text)
    alnum_ratio = (
        len([c for c in clean_text if c.isalnum()]) / len(clean_text)
        if clean_text else 0
    )
    return clean_text, round(alnum_ratio * 100, 2)


def extract_skills_from_corpus(resume_text: str, jobs_corpus: list) -> list[str]:
    """
    Stage 2: Embedding-based Skill Extraction.

    First pass  — exact/substring string match (case-insensitive, normalized).
    Second pass — semantic embedding match for OCR variations (high threshold).
    """
    # ── Collect all unique skills from Firestore corpus ───────────────────────
    all_known_skills = list({
        skill.strip()
        for job in jobs_corpus
        for skill in job.get("required_skills", [])
    })

    if not all_known_skills:
        return []

    resume_lower = resume_text.lower()

    # ── Normalize helper: strip punctuation and spaces for loose comparison ───
    def normalize(s: str) -> str:
        return re.sub(r'[\s.\-_/]', '', s).lower()

    resume_normalized = normalize(resume_text)

    # ── First pass: exact substring + normalized match ────────────────────────
    exact_matches = set()
    remaining_skills = []

    for skill in all_known_skills:
        skill_lower = skill.lower()
        skill_norm  = normalize(skill)

        # Skip single-character skills — too broad, always false positives
        if len(skill_norm) <= 1:
            continue

        # Short skills (<=3 chars) must match as whole words only
        if len(skill_norm) <= 3:
            pattern = rf'\b{re.escape(skill_lower)}\b'
            if re.search(pattern, resume_lower):
                exact_matches.add(skill)
            else:
                remaining_skills.append(skill)
            continue

        # Normal skills: check exact substring or normalized match
        if skill_lower in resume_lower or skill_norm in resume_normalized:
            exact_matches.add(skill)
        else:
            remaining_skills.append(skill)

    # ── Second pass: embedding match for OCR/formatting variations only ───────
    semantic_matches = set()
    if remaining_skills:
        resume_vec = model.encode(resume_text, convert_to_tensor=True)
        skill_vecs = model.encode(remaining_skills, convert_to_tensor=True)
        scores     = util.cos_sim(resume_vec, skill_vecs)[0]

        for i, score in enumerate(scores):
            if float(score) >= 0.60:
                semantic_matches.add(remaining_skills[i])

    all_matches = exact_matches | semantic_matches

    # ── Post-process: remove shorter skill if longer version also matched ─────
    to_remove = set()
    matches_list = list(all_matches)
    for i, skill_a in enumerate(matches_list):
        for j, skill_b in enumerate(matches_list):
            if i != j and skill_a.lower() in skill_b.lower():
                to_remove.add(skill_a)

    # ── Post-process: deduplicate case variants (keep title case version) ─────
    seen_normalized = {}
    for skill in sorted(all_matches - to_remove):
        key = normalize(skill)
        if key not in seen_normalized:
            seen_normalized[key] = skill

    return sorted(seen_normalized.values())


def fuzzy_skill_verify(discovered_list: list, ground_truth_list: list, threshold: int = 80):
    """Stage 2: Validation against expected skills."""
    matches = []
    for truth in ground_truth_list:
        for discovered in discovered_list:
            if fuzz.partial_ratio(truth.lower(), discovered.lower()) >= threshold:
                matches.append(truth)
                break
    return matches


def calculate_cosine_similarity(resume_text: str, resume_skills: list, jobs_corpus: list):
    """
    Stage 3: Hybrid Scoring with Comprehensive Skill & Gap Analysis.

    Uses pre-computed SBERT embeddings stored in each job document (added by
    generate_jobs.py) when available — falls back to live encoding if a job
    has no stored embedding (e.g. manually inserted documents).
    """
    resume_profile = f"Skills: {' '.join(resume_skills)}. Resume: {resume_text}"
    resume_vec = model.encode(resume_profile, convert_to_tensor=True)

    # ── Build job vector matrix ───────────────────────────────────────────────
    job_vecs_list = []
    jobs_to_encode_idx = []

    for i, job in enumerate(jobs_corpus):
        if "embedding" in job and job["embedding"]:
            vec = torch.tensor(job["embedding"], dtype=torch.float32).to(device)
            job_vecs_list.append(vec)
        else:
            job_vecs_list.append(None)
            jobs_to_encode_idx.append(i)

    # Live-encode any jobs that are missing embeddings
    if jobs_to_encode_idx:
        profiles = [
            f"Job: {jobs_corpus[i]['title']}. Core Skills: {' '.join(jobs_corpus[i]['required_skills'] * 3)}. Desc: {jobs_corpus[i]['description']}"
            for i in jobs_to_encode_idx
        ]
        live_vecs = model.encode(profiles, convert_to_tensor=True)
        for idx, vec in zip(jobs_to_encode_idx, live_vecs):
            job_vecs_list[idx] = vec

    job_vecs = torch.stack(job_vecs_list).to(device)
    cosine_scores = util.cos_sim(resume_vec, job_vecs)[0]

    # ── Score & rank ──────────────────────────────────────────────────────────
    results = []
    for i, score in enumerate(cosine_scores):
        raw_semantic_score = float(score)
        job_skills = jobs_corpus[i]["required_skills"]

        resume_set = {s.lower() for s in resume_skills}
        matched = [s for s in job_skills if s.lower() in resume_set]
        missing = [s for s in job_skills if s.lower() not in resume_set]
        gap_count = len(missing)

        match_ratio = len(matched) / len(job_skills) if job_skills else 0
        final_score = (raw_semantic_score * 0.6) + (match_ratio * 0.4)

        if gap_count == 0:
            final_score = min(0.985, final_score * 1.25)

        results.append(
            {
                "job_id":          jobs_corpus[i]["id"],
                "title":           jobs_corpus[i]["title"],
                "score":           f"{round(final_score * 100, 2)}%",
                "gap_count":       gap_count,
                "required_skills": job_skills,
                "matched_skills":  matched,
                "missing_skills":  missing,
            }
        )

    return sorted(results, key=lambda x: float(x["score"].replace("%", "")), reverse=True)[:5]


def identify_skill_gap(resume_skills: list, job_skills: list):
    """Stage 4: Gap Analysis."""
    resume_set = {s.strip().lower() for s in resume_skills}
    return [s for s in job_skills if s.lower() not in resume_set]