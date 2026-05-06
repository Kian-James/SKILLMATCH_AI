# db/__init__.py — re-export facade so existing import paths keep working
from db.client  import get_db
from db.jobs    import fetch_jobs_corpus, derive_skills_from_corpus, derive_certs_from_corpus, cache_job_embeddings
from db.results import save_resume_result


def load_analyzer_data() -> tuple[list, list, list]:
    """
    Fetch the jobs corpus and derive skills + certs allowlists in one call.

    Returns
    -------
    (firestore_skills, jobs_corpus, firestore_certs)
    """
    jobs_corpus      = fetch_jobs_corpus()
    firestore_skills = derive_skills_from_corpus(jobs_corpus)
    firestore_certs  = derive_certs_from_corpus(jobs_corpus)
    return firestore_skills, jobs_corpus, firestore_certs