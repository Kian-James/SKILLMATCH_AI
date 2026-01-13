import faiss
import numpy as np
from app.services.nlp.embeddings import embed
from app.database.firestore import db

DIM = 384
index = faiss.IndexFlatIP(DIM)
job_ids = []

def build_index():
    index.reset()
    job_ids.clear()

    texts = []
    for job in db.collection("jobs").stream():
        data = job.to_dict()
        texts.append(data["description"])
        job_ids.append(job.id)

    if texts:
        vectors = embed(texts).astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)

def search(query, k=10):
    vec = embed([query]).astype("float32")
    faiss.normalize_L2(vec)
    scores, ids = index.search(vec, k)

    results = []
    for i, idx in enumerate(ids[0]):
        if idx >= 0:
            job = db.collection("jobs").document(job_ids[idx]).get().to_dict()
            job["similarity"] = float(scores[0][i])
            results.append(job)

    return results
