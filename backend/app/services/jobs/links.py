# backend/app/services/jobs/links.py
from typing import List, Dict

class JobLinkHelper:
    """
    Standardizes and formats job opportunity links.
    Supports Indeed, JobStreet, LinkedIn, company URLs.
    """

    @staticmethod
    def normalize_links(job: Dict) -> List[str]:
        """
        Returns a list of valid job links for a job entry
        """
        links = []
        if "indeed_url" in job and job["indeed_url"]:
            links.append(job["indeed_url"])
        if "jobstreet_url" in job and job["jobstreet_url"]:
            links.append(job["jobstreet_url"])
        if "linkedin_url" in job and job["linkedin_url"]:
            links.append(job["linkedin_url"])
        if "company_url" in job and job["company_url"]:
            links.append(job["company_url"])
        return links
