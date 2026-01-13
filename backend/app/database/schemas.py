def resume_schema(skills):
    return {
        "skills": skills
    }

def job_schema(title, description, skills, clusters):
    return {
        "title": title,
        "description": description,
        "skills": skills,
        "skill_clusters": clusters
    }
