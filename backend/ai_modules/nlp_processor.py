"""
ai_modules/nlp_processor.py
Extracts technical skills and certifications from resume text.

Strategy:
1. Extract certifications with broad patterns, then deduplicate by keeping longest match
2. Extract skills from curated ontology only — no noun chunk heuristic
3. Exclude soft skills and cert fragments from skill results
"""

from __future__ import annotations

import re
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher
from loguru import logger

from utils.skill_normalizer import SkillNormalizer
from config import SPACY_MODEL


# ─── Certification patterns ───────────────────────────────────────────────────
# Each pattern tries to capture the full certification name in one match.
# Patterns are checked and then deduplicated — longest match wins.
CERT_PATTERNS = [
    # Format: "Provider – Certification Name" (dash/em-dash variants)
    r"\bUdemy\s*[–\-—]\s*[\w\s&()]+",
    r"\bCoursera\s*[–\-—]\s*[\w\s&()]+",
    r"\bedX\s*[–\-—]\s*[\w\s&()]+",
    r"\bLinkedIn\s+Learning\s*[–\-—]\s*[\w\s&()]+",
    r"\bFreeCodeCamp\s*[–\-—]\s*[\w\s&()]+",
    # Format: "Provider – Name" where name bleeds into next cert (common in PDFs)
    # Cisco specific — captures "Cisco – Certified Information Technology (ITF+)"
    r"\bCisco\s*[–\-—]\s*[\w\s&()+]+",
    r"\bCisco\s+Certified[\w\s&()]+",
    r"\bCisco\s+(?:CCNA|CCNP|CCIE|CyberOps)[\w\s]*",
    # AWS
    r"\bAWS\s+Certified\s+[\w\s]+",
    # Google Cloud
    r"\bGoogle\s+Cloud\s+[\w\s]+(?:Associate|Professional|Engineer|Developer)",
    r"\bProfessional\s+(?:Data\s+Engineer|Machine\s+Learning\s+Engineer|Cloud\s+Architect)",
    # Microsoft
    r"\bMicrosoft\s+Certified[\w\s:]+",
    r"\bAZ-\d{3}\b",
    # CompTIA
    r"\bCompTIA\s+[\w+]+",
    r"\bA\+\b",
    r"\bNetwork\+\b",
    r"\bSecurity\+\b",
    # Kubernetes
    r"\bCertified\s+Kubernetes\s+(?:Administrator|Developer|Security\s+Specialist)",
    r"\bCKA\b", r"\bCKAD\b", r"\bCKS\b",
    # Security
    r"\bCertified\s+Ethical\s+Hacker",
    r"\bCEH\b", r"\bCISSP\b", r"\bCISM\b", r"\bOSCP\b",
    # Project Mgmt
    r"\bPMP\b", r"\bPRINCE2[\w\s]*", r"\bScrumMaster\b", r"\bCSM\b", r"\bPSM\b",
    # Oracle / Red Hat
    r"\bOracle\s+Certified[\w\s]+",
    r"\bRed\s+Hat\s+Certified[\w\s]+",
    r"\bRHCE\b", r"\bRHCSA\b",
    # TensorFlow
    r"\bTensorFlow\s+Developer\s+Certificate",
    # Generic "Certified Information Technology" style
    r"\bCertified\s+Information\s+Technology[\w\s()]+",
    # FreeCodeCamp certs listed without dash: "Responsive Web Design FreeCodeCamp"
    r"(?:Responsive\s+Web\s+Design|Relational\s+Database|JavaScript\s+Algorithms"
    r"|Front\s+End\s+Development|Back\s+End\s+Development|Data\s+Visualization"
    r"|Scientific\s+Computing|Machine\s+Learning|Quality\s+Assurance"
    r"|College\s+Algebra)\s+(?:FreeCodeCamp|Certificate)?",
    # "Flutter Program Certified" (Udemy cert often appears without provider)
    r"\bFlutter\s+Program\s+Certif\w+",
    # ITF+ standalone
    r"\bITF\+",
    # ISO
    r"\bISO\s+\d+[\w\s]*(?:Lead\s+Auditor|Lead\s+Implementer)?",
]

# Soft skills — explicitly excluded from technical skill extraction
SOFT_SKILLS_EXCLUDE: set[str] = {
    "adaptability", "time management", "problem solving", "problem-solving",
    "analytical thinking", "attention to detail", "communication",
    "written & verbal", "written and verbal", "teamwork", "leadership",
    "critical thinking", "creativity", "interpersonal skills", "work ethic",
    "multitasking", "organization", "collaboration", "flexibility",
    "decision making", "emotional intelligence", "conflict resolution",
    "presentation skills", "negotiation", "detail oriented", "detail-oriented",
    "fast learner", "self-motivated", "verbal communication", "written communication",
    "strong work ethic", "hard skills", "soft skills",
}

# Phrases that signal a cert context — used to filter skill matches
CERT_PROVIDER_WORDS: set[str] = {
    "freecodecamp", "udemy", "coursera", "edx", "cisco", "microsoft",
    "google", "amazon", "oracle", "linkedin", "certified", "certification",
    "certificate",
}

# Curated technical skills
KNOWN_SKILLS: list[str] = [
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Kotlin", "Swift", "R", "Scala", "PHP", "Ruby", "MATLAB", "Dart",
    "Bash", "Shell", "PowerShell", "Groovy", "Lua",
    # Web Frameworks
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Express.js", "Next.js", "Nuxt.js", "Svelte", "jQuery",
    "Laravel", "Ruby on Rails", "ASP.NET",
    # Mobile
    "Flutter", "React Native", "Android", "iOS", "SwiftUI", "Jetpack Compose",
    # Data & ML
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Pandas", "NumPy",
    "FAISS", "HuggingFace", "spaCy", "NLTK", "OpenCV", "Spark", "Hadoop",
    "Airflow", "dbt", "Kafka", "Flink", "Matplotlib", "Seaborn",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Helm",
    "Linux", "Ubuntu", "Nginx", "Apache",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "DynamoDB", "Firebase", "Firestore", "BigQuery", "Snowflake", "SQLite",
    # Networking & Security
    "TCP/IP", "GraphQL", "gRPC", "OAuth", "JWT", "REST API", "RESTful APIs",
    "Penetration Testing", "SIEM", "Firewalls",
    # Tools
    "Git", "GitHub", "GitLab", "Jira", "Postman", "Swagger", "Figma",
    # Concepts & stacks
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Structures", "Algorithms", "Microservices", "CI/CD",
    "Agile", "Scrum", "System Design", "OOP", "Functional Programming",
    "Data Engineering", "MLOps", "DevOps", "Cloud Architecture",
    "CRUD", "MERN Stack", "MEAN Stack",
    "Full Stack Development", "Backend Development", "Frontend Development",
    "Mobile Development", "Web Development", "API Development",
    "Unit Testing", "TDD", "Responsive Design", "Version Control",
    "Authentication", "Authorization", "Role-Based Access Control",
    "Real-Time Data", "WebSockets",
    "Flutter & Firebase Development", "Flutter & Firebase",
    "Git Version Control",
    "REST APIs & Backend Development",
    "CRUD Application Development",
    "Blockchain", "Smart Contracts", "Solidity",
    "Embedded Systems", "IoT", "Cybersecurity",
    "Data Analysis", "Data Visualization", "Business Intelligence",
]


@lru_cache(maxsize=1)
def _load_nlp_model():
    try:
        nlp = spacy.load(SPACY_MODEL)
        logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
    except OSError:
        fallback = "en_core_web_sm"
        logger.warning(f"{SPACY_MODEL} not found. Falling back to {fallback}.")
        nlp = spacy.load(fallback)
    return nlp


def _build_phrase_matcher(nlp, skills: list[str]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill.lower()) for skill in skills]
    matcher.add("SKILL", patterns)
    return matcher


def _deduplicate_certs(raw_certs: list[str]) -> list[str]:
    """
    Remove cert duplicates by keeping the longest match when one string
    is a substring of another.
    e.g. "ITF+" is a substring of "Certified Information Technology (ITF+)"
    so we keep the longer one only.
    Also normalises spacing and trims trailing junk.
    """
    # Clean each cert
    cleaned: list[str] = []
    for c in raw_certs:
        c = " ".join(c.split())  # collapse whitespace
        c = c.strip(" –-—.,;:")  # strip leading/trailing punctuation
        if len(c) > 2:
            cleaned.append(c)

    # Sort by length descending — longer = more specific
    cleaned.sort(key=len, reverse=True)

    kept: list[str] = []
    for cert in cleaned:
        cert_lower = cert.lower()
        # Only keep if no already-kept cert contains this one as a substring
        if not any(cert_lower in kept_cert.lower() for kept_cert in kept):
            kept.append(cert)

    return sorted(kept)


class NLPProcessor:
    """
    Extracts technical skills and certifications from resume text.
    Certifications are extracted first, then used to filter skill extraction.
    """

    def __init__(self, extra_skills: list[str] | None = None):
        self.nlp = _load_nlp_model()
        self.normalizer = SkillNormalizer()
        all_skills = KNOWN_SKILLS + (extra_skills or [])
        self.matcher = _build_phrase_matcher(self.nlp, all_skills)

    def extract_certifications(self, text: str) -> list[str]:
        """
        Extract and deduplicate certification names.
        Keeps the longest match when overlapping patterns fire.
        """
        raw: list[str] = []
        for pattern in CERT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                match_text = m.group().strip()
                if len(match_text) > 3:
                    raw.append(match_text)

        return _deduplicate_certs(raw)

    def extract_skills(self, text: str, cert_strings: list[str]) -> list[str]:
        """
        Extract skills from the curated ontology.
        Filters out matches that appear inside a certification string.
        """
        # Build a set of all lowercase words that appear in cert names
        cert_context_words: set[str] = set()
        for cert in cert_strings:
            for word in cert.lower().split():
                if len(word) > 3:
                    cert_context_words.add(word)
        cert_context_words.update(CERT_PROVIDER_WORDS)

        doc = self.nlp(text)
        matches = self.matcher(doc)

        raw_skills: set[str] = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            span_text = span.text.strip()

            # Skip soft skills
            if span_text.lower() in SOFT_SKILLS_EXCLUDE:
                continue

            # Skip if the span itself is entirely cert provider words
            span_words = span_text.lower().split()
            if all(w in cert_context_words for w in span_words):
                continue

            # Skip if this exact text appears verbatim in a cert name
            span_lower = span_text.lower()
            if any(span_lower in cert.lower() and
                   any(w in cert.lower() for w in CERT_PROVIDER_WORDS)
                   for cert in cert_strings):
                continue

            raw_skills.add(span_text)

        normalized = [self.normalizer.normalize(s) for s in raw_skills]
        return sorted(set(filter(None, normalized)))

    def process(self, raw_text: str) -> dict:
        """Full pipeline: clean → extract certs → extract skills."""
        cleaned = self._clean_text(raw_text)

        # Certs first
        certifications = self.extract_certifications(cleaned)

        # Skills with cert exclusion context
        skills = self.extract_skills(cleaned, certifications)

        logger.info(
            f"NLP extracted {len(skills)} skills, {len(certifications)} certifications."
        )
        return {
            "skills": skills,
            "certifications": certifications,
            "word_count": len(cleaned.split()),
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", " ", text)
        text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()