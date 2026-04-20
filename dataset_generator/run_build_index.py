"""
dataset_generator/build_faiss_index.py

Pre-build the FAISS indexes locally and save them to faiss_index/.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# --- PATH FIXING SECTION ---
# This ensures that the project root is in the python path,
# allowing "from backend.services..." to work correctly.
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent # Moves up from dataset_generator to SKILLMATCH_AI
backend_path = project_root / "backend"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))
# ---------------------------

from loguru import logger

OUTPUT_DIR = Path("faiss_index")
BATCH = 50       # jobs per embedding batch — keeps RAM low
CTX_BATCH = 500  # ctx texts per embedding batch


def build():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── 1. Firebase ───────────────────────────────────────────────────────────
    logger.info("Initializing Firebase...")
    # Import inside the function to ensure sys.path is already updated
    from backend.services.firebase_service import init_firebase, fetch_all_jobs
    init_firebase()

    jobs = fetch_all_jobs()
    if not jobs:
        logger.error("No jobs in Firestore. Run upload_dataset.py first.")
        sys.exit(1)

    N = len(jobs)
    logger.info(f"Loaded {N} jobs from Firestore.")

    # ── 2. Embedding model ────────────────────────────────────────────────────
    logger.info("Loading embedding model...")
    from backend.ai_modules.embedding_engine import EmbeddingEngine
    engine = EmbeddingEngine()
    dim = engine.model.get_sentence_embedding_dimension()
    logger.info(f"Model ready. Embedding dim: {dim}")

    # ── 3. IDF weights ─────────────────────────────────────────────────────────
    logger.info("Computing IDF weights...")
    freq: Counter = Counter()
    for job in jobs:
        freq.update(set(s.lower() for s in job.get("skills", [])))
    idf_weights = {s: math.log(N / f) for s, f in freq.items() if f > 0}
    logger.info(f"IDF weights: {len(idf_weights)} unique skills.")

    # ── 4. Job FAISS index (batched) ───────────────────────────────────────────
    import faiss
    job_index = faiss.IndexFlatIP(dim)
    metadata  = []

    logger.info(f"Embedding {N} jobs in batches of {BATCH}...")
    for start in range(0, N, BATCH):
        chunk = jobs[start : start + BATCH]
        texts = [
            job["job_title"] + " " + " ".join(job.get("skills", []))
            for job in chunk
        ]
        embs = engine.embed_batch(texts)
        job_index.add(embs)
        for job in chunk:
            metadata.append({
                "job_title":      job["job_title"],
                "skills":         job.get("skills", []),
                "certifications": job.get("certifications", []),
                "doc_id":         job.get("id", ""),
            })
        del embs, texts
        gc.collect()
        logger.info(f"  Job embeddings: {min(start + BATCH, N)}/{N}")

    # ── 5. Skill FAISS index + collect ctx texts ───────────────────────────────
    skill_index = faiss.IndexFlatIP(dim)
    skill_meta  = []
    all_ctx_texts: list[str] = []

    logger.info(f"Embedding skills for {N} jobs...")
    for job_idx, job in enumerate(jobs):
        skills = job.get("skills", [])
        if not skills:
            continue
        bare_embs = engine.embed_batch(skills)
        skill_index.add(bare_embs)
        for skill_name in skills:
            skill_meta.append({"job_idx": job_idx, "skill_name": skill_name})
        all_ctx_texts.extend(f"{job['job_title']}: {s}" for s in skills)
        del bare_embs
        if (job_idx + 1) % BATCH == 0:
            gc.collect()
            logger.info(f"  Skill embeddings: {job_idx + 1}/{N}")

    gc.collect()
    logger.info(f"Skill index: {skill_index.ntotal} vectors.")

    # ── 6. Contextual embeddings (float16) ─────────────────────────────────────
    logger.info(f"Embedding {len(all_ctx_texts)} contextual skill texts...")
    ctx_chunks: list[np.ndarray] = []
    for start in range(0, len(all_ctx_texts), CTX_BATCH):
        chunk_embs = engine.embed_batch(
            all_ctx_texts[start : start + CTX_BATCH]
        ).astype(np.float16)
        ctx_chunks.append(chunk_embs)
        del chunk_embs
        gc.collect()
        logger.info(f"  Ctx embeddings: {min(start + CTX_BATCH, len(all_ctx_texts))}/{len(all_ctx_texts)}")

    ctx_embs = np.concatenate(ctx_chunks) if ctx_chunks else None
    del ctx_chunks, all_ctx_texts
    gc.collect()

    # ── 7. Save all files ──────────────────────────────────────────────────────
    logger.info("Saving indexes to disk...")

    faiss.write_index(job_index,   str(OUTPUT_DIR / "job_embeddings.index"))
    faiss.write_index(skill_index, str(OUTPUT_DIR / "skill_embeddings.index"))

    with open(OUTPUT_DIR / "job_metadata.json", "w") as f:
        json.dump(metadata, f)

    with open(OUTPUT_DIR / "skill_metadata.json", "w") as f:
        json.dump(skill_meta, f)

    if ctx_embs is not None:
        np.save(str(OUTPUT_DIR / "ctx_embeddings.npy"), ctx_embs)

    with open(OUTPUT_DIR / "idf_weights.json", "w") as f:
        json.dump(idf_weights, f)

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Build complete. Files written to faiss_index/:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size_mb = p.stat().st_size / 1_000_000
        logger.info(f"  {p.name:35s} {size_mb:.1f} MB")
    logger.info("=" * 50)


if __name__ == "__main__":
    build()