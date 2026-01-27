from typing import List, Dict
import re
from app.repositories.firestore.skills import SkillRepository
from app.repositories.firestore.certifications import CertificationRepository

# Initialize repositories
skill_repo = SkillRepository()
cert_repo = CertificationRepository()

def fetch_reference_data():
    """
    Fetch known skills, tools, and certifications from Firestore
    """
    skills_docs = skill_repo.get_all_skills()
    certifications_docs = cert_repo.get_all_certifications()

    # Flatten the documents to simple lists
    skills = [s.get("name") for s in skills_docs if s.get("name")]
    tools = [s.get("tools") for s in skills_docs if s.get("tools")]
    # Flatten nested tool lists if needed
    flat_tools = []
    for t in tools:
        if isinstance(t, list):
            flat_tools.extend(t)
        elif isinstance(t, str):
            flat_tools.append(t)

    certifications = [c.get("name") for c in certifications_docs if c.get("name")]

    return skills, flat_tools, certifications


def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    return [skill for skill in skills_list if skill.lower() in text.lower()]


def extract_tools(text: str, tools_list: List[str]) -> List[str]:
    return [tool for tool in tools_list if tool.lower() in text.lower()]


def extract_certifications(text: str, cert_list: List[str]) -> List[str]:
    return [cert for cert in cert_list if cert.lower() in text.lower()]


def extract_education(text: str) -> List[str]:
    education_keywords = ["Bachelor", "Master", "PhD", "BS", "MS", "B.Sc", "M.Sc"]
    lines = text.splitlines()
    edu_lines = [line.strip() for line in lines if any(k.lower() in line.lower() for k in education_keywords)]
    return edu_lines


def extract_experience(text: str) -> List[str]:
    experience_patterns = [
        r"(\d{4}-\d{4})",  # e.g. 2018-2021
        r"(intern|developer|engineer|manager|lead)",  # positions
    ]
    lines = text.splitlines()
    exp_lines = []
    for line in lines:
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in experience_patterns):
            exp_lines.append(line.strip())
    return exp_lines


def parse_resume(text: str) -> Dict[str, List[str]]:
    """
    Parses a resume text into structured entities using Firestore reference data
    """
    # Fetch reference lists from Firestore
    skills_list, tools_list, cert_list = fetch_reference_data()

    return {
        "skills": extract_skills(text, skills_list),
        "tools": extract_tools(text, tools_list),
        "certifications": extract_certifications(text, cert_list),
        "education": extract_education(text),
        "experience": extract_experience(text),
    }
