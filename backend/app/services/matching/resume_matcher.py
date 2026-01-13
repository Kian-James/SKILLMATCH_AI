from app.services.vector_store.faiss_index import search

def match_jobs(resume_skills):
    query = " ".join(s["skill"] for s in resume_skills if s["confidence"] >= 0.5)
    jobs = search(query)

    resume_clusters = {
        s["cluster_id"] for s in resume_skills if s["confidence"] >= 0.4
    }

    matches = []
    for job in jobs:
        overlap = resume_clusters.intersection(job["skill_clusters"])
        if len(overlap) >= 3:
            matches.append({
                "title": job["title"],
                "similarity": job["similarity"],
                "matching_clusters": list(overlap)
            })

    return sorted(matches, key=lambda x: x["similarity"], reverse=True)
