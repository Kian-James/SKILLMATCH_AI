"""
ai_modules/nlp_processor.py
Uses spaCy to extract skills, certifications, and other resume entities.
Employs rule-based matching combined with a curated skill ontology.
"""

from __future__ import annotations

import re
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher
from loguru import logger

from utils.skill_normalizer import SkillNormalizer

# ─── Certification Keywords ────────────────────────────────────────────────────
# Common certification patterns — extended via dataset over time
CERT_PATTERNS = [
    r"\bAWS\s+Certified\s+[\w\s]+",
    r"\bGoogle\s+Cloud\s+[\w\s]+(?:Associate|Professional|Engineer)",
    r"\bMicrosoft\s+Certified[\w\s:]+",
    r"\bCisco\s+(?:CCNA|CCNP|CCIE|CyberOps)[\w\s]*",
    r"\bCertified\s+Kubernetes\s+(?:Administrator|Developer)",
    r"\bCKA\b",
    r"\bCKAD\b",
    r"\bPMP\b",
    r"\bScrumMaster\b",
    r"\bCSM\b",
    r"\bCompTIA\s+[\w+]+",
    r"\bOracle\s+Certified[\w\s]+",
    r"\bRed\s+Hat\s+Certified[\w\s]+",
    r"\bCertified\s+Ethical\s+Hacker",
    r"\bCEH\b",
    r"\bCISSP\b",
    r"\bTensorFlow\s+Developer\s+Certificate",
    r"\bProfessional\s+Data\s+Engineer",
    r"\bProfessional\s+Machine\s+Learning\s+Engineer",
]

# Known technical skills curated list (seed; grows via Firestore dataset)
KNOWN_SKILLS: list[str] = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Kotlin", "Swift", "R", "Scala", "PHP", "Ruby", "MATLAB", "Dart",
    # Web Frameworks
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Express.js", "Next.js", "Nuxt.js",
    # Mobile
    "Flutter", "React Native", "Android", "iOS", "SwiftUI", "Jetpack Compose",
    # Data & ML
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Pandas", "NumPy",
    "FAISS", "HuggingFace", "spaCy", "NLTK", "OpenCV", "Spark", "Hadoop",
    "Airflow", "dbt", "Kafka", "Flink",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Helm",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "DynamoDB", "Firebase", "Firestore", "BigQuery", "Snowflake",
    # Networking & Security
    "TCP/IP", "REST API", "GraphQL", "gRPC", "OAuth", "JWT",
    "Penetration Testing", "SIEM", "Firewalls",
    # Concepts
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Structures", "Algorithms", "Microservices", "CI/CD",
    "Agile", "Scrum", "System Design", "OOP", "Functional Programming",
    "Data Engineering", "MLOps", "DevOps", "Cloud Architecture",
]


@lru_cache(maxsize=1)
def _load_nlp_model():
    """Load spaCy model (cached singleton)."""
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        logger.warning("en_core_web_lg not found. Falling back to en_core_web_sm.")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def _build_phrase_matcher(nlp, skills: list[str]) -> PhraseMatcher:
    """Build a spaCy PhraseMatcher from skill list."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill.lower()) for skill in skills]
    matcher.add("SKILL", patterns)
    return matcher


class NLPProcessor:
    """
    Processes raw resume text to extract:
    - Skills (matched against known ontology + custom skills)
    - Certifications (regex pattern matching)
    """

    def __init__(self, extra_skills: list[str] | None = None):
        self.nlp = _load_nlp_model()
        self.normalizer = SkillNormalizer()
        all_skills = KNOWN_SKILLS + (extra_skills or [])
        self.matcher = _build_phrase_matcher(self.nlp, all_skills)

    def extract_skills(self, text: str) -> list[str]:
        """
        Extract skills from text using spaCy PhraseMatcher.
        Returns normalized, deduplicated skill list.
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        raw_skills: set[str] = set()
        for match_id, start, end in matches:
            span_text = doc[start:end].text
            raw_skills.add(span_text)

        # Also look for noun chunks that could be skills not in our list
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if 2 <= len(chunk_text.split()) <= 4:
                # Heuristic: short proper noun chunks in tech context
                if chunk_text[0].isupper():
                    raw_skills.add(chunk_text)

        # Normalize and deduplicate via skill normalizer
        normalized = [self.normalizer.normalize(s) for s in raw_skills]
        return sorted(set(filter(None, normalized)))

    def extract_certifications(self, text: str) -> list[str]:
        """Extract certification names using regex patterns."""
        found: set[str] = set()
        for pattern in CERT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                cleaned = " ".join(m.split())  # collapse whitespace
                found.add(cleaned)
        return sorted(found)

    def process(self, raw_text: str) -> dict:
        """
        Full pipeline: text → skills + certifications.
        Returns dict with extracted entities.
        """
        cleaned = self._clean_text(raw_text)
        skills = self.extract_skills(cleaned)
        certifications = self.extract_certifications(cleaned)

        logger.info(f"NLP extracted {len(skills)} skills, {len(certifications)} certifications.")
        return {
            "skills": skills,
            "certifications": certifications,
            "word_count": len(cleaned.split()),
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove noise (URLs, emails, phone numbers) from resume text."""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove email addresses
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", " ", text)
        # Remove phone numbers
        text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
