"""
ai_modules/nlp_processor.py
Uses spaCy to extract skills and certifications from resume text.

Key improvements:
- Certifications extracted first, then excluded from skill extraction
- Broader cert patterns covering Cisco ITF+, Udemy, FreeCodeCamp, etc.
- Noun chunk heuristic removed (was grabbing soft skills, cert names, summaries)
- Soft skills explicitly excluded from the skill list
- Skills matched only against curated technical ontology
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
# Ordered from most specific to least specific.
# Each pattern captures the full certification name as it appears on a resume.
CERT_PATTERNS = [
    # AWS
    r"\bAWS\s+Certified\s+[\w\s]+",
    # Google Cloud
    r"\bGoogle\s+Cloud\s+[\w\s]+(?:Associate|Professional|Engineer|Developer)",
    r"\bProfessional\s+(?:Data\s+Engineer|Machine\s+Learning\s+Engineer|Cloud\s+Architect)",
    # Microsoft
    r"\bMicrosoft\s+Certified[\w\s:]+",
    r"\bAZ-\d{3}\b",
    # Cisco
    r"\bCisco\s+(?:Certified\s+)?[\w\s]+(?:ITF\+|CCNA|CCNP|CCIE|CyberOps)[\w\s]*",
    r"\bCisco\s+[\w\s]*ITF\+",
    r"\b(?:CCNA|CCNP|CCIE)\b",
    # CompTIA
    r"\bCompTIA\s+[\w+]+",
    r"\bA\+\b",
    r"\bNetwork\+\b",
    r"\bSecurity\+\b",
    # Kubernetes
    r"\bCertified\s+Kubernetes\s+(?:Administrator|Developer|Security\s+Specialist)",
    r"\bCKA\b",
    r"\bCKAD\b",
    r"\bCKS\b",
    # Security
    r"\bCertified\s+Ethical\s+Hacker",
    r"\bCEH\b",
    r"\bCISSP\b",
    r"\bCISM\b",
    r"\bOSCP\b",
    # Project Management
    r"\bPMP\b",
    r"\bPRINCE2[\w\s]*",
    r"\bScrumMaster\b",
    r"\bCSM\b",
    r"\bPSM\b",
    # Oracle / Red Hat
    r"\bOracle\s+Certified[\w\s]+",
    r"\bRed\s+Hat\s+Certified[\w\s]+",
    r"\bRHCE\b",
    r"\bRHCSA\b",
    # TensorFlow / ML
    r"\bTensorFlow\s+Developer\s+Certificate",
    # FreeCodeCamp — matches "FreeCodeCamp – Responsive Web Design" etc.
    r"\bFreeCodeCamp\s*[–\-—]\s*[\w\s]+",
    r"\bFreeCodeCamp\s+[\w\s]+(?:Certificate|Certification|Design|Database|Development)",
    r"(?:Responsive\s+Web\s+Design|Relational\s+Database|JavaScript\s+Algorithms)\s+(?:FreeCodeCamp|Certificate)?",
    # Udemy — matches "Udemy – Flutter Program Certified" etc.
    r"\bUdemy\s*[–\-—]\s*[\w\s]+",
    r"\bUdemy\s+[\w\s]+(?:Certificate|Certified|Course)?",
    r"Flutter\s+Program\s+Certified(?:\s*FreeCodeCamp)?",
    # Coursera / edX / LinkedIn Learning
    r"\bCoursera\s*[–\-—]\s*[\w\s]+",
    r"\bedX\s*[–\-—]\s*[\w\s]+",
    r"\bLinkedIn\s+Learning\s*[–\-—]\s*[\w\s]+",
    # Generic patterns — "Certified Information Technology (ITF+)"
    r"\bCertified\s+Information\s+Technology[\w\s()]+",
    r"\bITF\+",
    # ISO
    r"\bISO\s+\d+[\w\s]*(?:Lead\s+Auditor|Lead\s+Implementer)?",
]

# Soft skills to explicitly exclude — these are NOT technical skills
SOFT_SKILLS_EXCLUDE = {
    "adaptability", "time management", "problem solving", "problem-solving",
    "analytical thinking", "attention to detail", "communication",
    "written & verbal", "written and verbal", "teamwork", "leadership",
    "critical thinking", "creativity", "interpersonal skills", "work ethic",
    "multitasking", "organization", "collaboration", "flexibility",
    "decision making", "emotional intelligence", "conflict resolution",
    "project management", "presentation skills", "negotiation",
    "detail oriented", "detail-oriented", "fast learner", "self-motivated",
    "strong work ethic", "verbal communication", "written communication",
}

# Words/phrases that appear in cert names but should NOT be extracted as skills
CERT_KEYWORD_FRAGMENTS = {
    "freecodecamp", "udemy", "coursera", "edx", "certified", "certification",
    "certificate", "program certified",
}

# Curated technical skills list — only real technical skills, no soft skills
KNOWN_SKILLS: list[str] = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Kotlin", "Swift", "R", "Scala", "PHP", "Ruby", "MATLAB", "Dart",
    "Bash", "Shell", "PowerShell", "Groovy", "Lua", "Haskell", "Elixir",
    # Web Frameworks & Libraries
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Express.js", "Next.js", "Nuxt.js", "Svelte", "jQuery",
    "Laravel", "Ruby on Rails", "ASP.NET", "Blazor",
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
    "MariaDB", "Oracle Database", "MS SQL Server",
    # Networking & Security
    "TCP/IP", "GraphQL", "gRPC", "OAuth", "JWT", "REST API", "RESTful APIs",
    "Penetration Testing", "SIEM", "Firewalls", "VPN", "DNS", "HTTP", "HTTPS",
    # Tools
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
    "Postman", "Swagger", "VS Code", "IntelliJ", "Eclipse", "Xcode",
    "Figma", "Adobe XD", "Photoshop",
    # Concepts & Methodologies
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Structures", "Algorithms", "Microservices", "CI/CD",
    "Agile", "Scrum", "System Design", "OOP", "Functional Programming",
    "Data Engineering", "MLOps", "DevOps", "Cloud Architecture",
    "CRUD", "MVC", "MERN Stack", "MEAN Stack", "Full Stack Development",
    "Backend Development", "Frontend Development", "Web Development",
    "Mobile Development", "API Development", "Unit Testing", "TDD",
    "Responsive Design", "Version Control", "Authentication", "Authorization",
    "Role-Based Access Control", "Real-Time Data", "WebSockets",
    "Blockchain", "Smart Contracts", "Solidity",
    "Embedded Systems", "IoT", "RTOS", "Assembly",
    "Cybersecurity", "Ethical Hacking", "Network Security",
    "Data Analysis", "Data Visualization", "Business Intelligence",
]


@lru_cache(maxsize=1)
def _load_nlp_model():
    """Load spaCy model (cached singleton)."""
    try:
        nlp = spacy.load(SPACY_MODEL)
        logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
    except OSError:
        fallback = "en_core_web_sm"
        logger.warning(f"{SPACY_MODEL} not found. Falling back to {fallback}.")
        nlp = spacy.load(fallback)
    return nlp


def _build_phrase_matcher(nlp, skills: list[str]) -> PhraseMatcher:
    """Build a spaCy PhraseMatcher from skill list."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill.lower()) for skill in skills]
    matcher.add("SKILL", patterns)
    return matcher


def _is_cert_fragment(text: str) -> bool:
    """Return True if the text looks like it came from a certification name."""
    lower = text.lower()
    return any(frag in lower for frag in CERT_KEYWORD_FRAGMENTS)


def _is_soft_skill(text: str) -> bool:
    """Return True if the text is a soft skill that should be excluded."""
    return text.lower().strip() in SOFT_SKILLS_EXCLUDE


class NLPProcessor:
    """
    Extracts technical skills and certifications from resume text.

    Pipeline:
    1. Extract certifications first using regex
    2. Build a set of cert-related text to exclude from skills
    3. Extract skills using PhraseMatcher against curated ontology only
    4. Filter out anything that matched a certification or is a soft skill
    """

    def __init__(self, extra_skills: list[str] | None = None):
        self.nlp = _load_nlp_model()
        self.normalizer = SkillNormalizer()
        all_skills = KNOWN_SKILLS + (extra_skills or [])
        self.matcher = _build_phrase_matcher(self.nlp, all_skills)

    def extract_certifications(self, text: str) -> list[str]:
        """
        Extract certification names using regex patterns.
        Runs BEFORE skill extraction so certs can be excluded from skills.
        """
        found: set[str] = set()
        for pattern in CERT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                cleaned = " ".join(m.split()).strip(" –-—")
                if len(cleaned) > 3:
                    found.add(cleaned)
        return sorted(found)

    def extract_skills(self, text: str, cert_text_lower: set[str]) -> list[str]:
        """
        Extract skills using PhraseMatcher against the curated skills ontology.
        Excludes anything that:
        - Is part of a certification name
        - Is a known soft skill
        - Is a cert keyword fragment (FreeCodeCamp, Udemy, etc.)
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        raw_skills: set[str] = set()
        for match_id, start, end in matches:
            span_text = doc[start:end].text.strip()

            # Skip if this skill text appears inside a certification string
            if span_text.lower() in cert_text_lower:
                continue

            # Skip soft skills
            if _is_soft_skill(span_text):
                continue

            # Skip cert keyword fragments
            if _is_cert_fragment(span_text):
                continue

            raw_skills.add(span_text)

        # Normalize and deduplicate
        normalized = [self.normalizer.normalize(s) for s in raw_skills]
        return sorted(set(filter(None, normalized)))

    def process(self, raw_text: str) -> dict:
        """
        Full pipeline: text → certifications first, then skills.
        Certifications are used to build an exclusion set for skill extraction.
        """
        cleaned = self._clean_text(raw_text)

        # Step 1: Extract certifications
        certifications = self.extract_certifications(cleaned)

        # Step 2: Build exclusion set from cert names
        # Any word or phrase appearing in a cert name should not be a skill
        cert_text_lower: set[str] = set()
        for cert in certifications:
            cert_text_lower.add(cert.lower())
            # Also add individual words from cert names to catch fragments
            for word in cert.lower().split():
                if len(word) > 3:
                    cert_text_lower.add(word)

        # Step 3: Extract skills with cert exclusion
        skills = self.extract_skills(cleaned, cert_text_lower)

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
        """Remove noise from resume text."""
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", " ", text)
        text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()