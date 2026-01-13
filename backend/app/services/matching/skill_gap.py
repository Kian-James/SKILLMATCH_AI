def skill_gap(resume_clusters, job_clusters):
    return list(set(job_clusters) - set(resume_clusters))
