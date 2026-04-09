"""
ai_modules/nlp_processor.py

Section-aware resume parser.

FIXES (v2):
  BUG 1 - CERTIFICATIONS added to NEXT_SECTION_HEADERS (skills ran past cert block)
  BUG 2 - CAREER OBJECTIVE / OBJECTIVE added to NEXT_SECTION_HEADERS
  BUG 3 - TECHNOLOGIES/TECH STACK/TOOLS headers now require line-start anchor
           to avoid false positives inside project bullet points like "Technologies: React"
  BUG 5 - Expanded SOFT_SKILLS_EXCLUDE with generic phrases from experience text
           + minimum skill length enforced in extract_skills
"""

from __future__ import annotations

import re
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher
from loguru import logger

from utils.skill_normalizer import SkillNormalizer
from config import SPACY_MODEL


# ─── Section header patterns ──────────────────────────────────────────────────

SKILLS_SECTION_HEADERS = [
    # General skill headers — keep loose match (rarely appear inline)
    r"(?:^|\n)(?:SKILLS?|TECHNICAL\s+SKILLS?|CORE\s+SKILLS?|KEY\s+SKILLS?"
    r"|COMPETENCIES|HARD\s+SKILLS?)\s*(?:\n|:|\s{2,})",

    # FIX BUG 3: Use end-of-line anchor ($) so "Technologies: React, Docker"
    # (inline content after colon) is NOT matched, but "Technologies" or
    # "Technologies:" on their own line ARE matched as section headers.
    r"^(?:TECHNOLOGIES|TECH\s+STACK|TOOLS?\s+(?:&|AND)\s+TECHNOLOGIES?|EXPERTISE)\s*:?\s*$",
]

CERT_SECTION_HEADERS = [
    r"(?:^|\n|\s{2,})(?:CERTIFICATIONS?|CERTIFICATES?|LICENSES?\s+&\s+CERTIFICATIONS?"
    r"|PROFESSIONAL\s+CERTIFICATIONS?|CREDENTIALS?|ACCREDITATIONS?)\s*(?:\n|:|\s{2,})",
]

# FIX BUG 1 + BUG 2: Added CERTIFICATIONS? and CAREER OBJECTIVE / OBJECTIVE
NEXT_SECTION_HEADERS = [
    r"(?:^|\n|\s{2,})(?:EXPERIENCE|WORK\s+EXPERIENCE|EMPLOYMENT|PROJECTS?"
    r"|EDUCATION|ACHIEVEMENTS?|AWARDS?|INTERESTS?|LANGUAGES?|REFERENCES?"
    r"|PUBLICATIONS?|VOLUNTEER|EXTRACURRICULAR|ACTIVITIES|HOBBIES|SUMMARY"
    r"|CAREER\s+OBJECTIVE|OBJECTIVE|PROFILE|ABOUT|CONTACT|SOFT\s+SKILLS?"
    r"|CERTIFICATIONS?|CERTIFICATES?|LICENSES?\s+&\s+CERTIFICATIONS?"
    r"|PROFESSIONAL\s+CERTIFICATIONS?|CREDENTIALS?|ACCREDITATIONS?"
    r")\s*(?:\n|:|\s{2,})",
]

# ─── Certification patterns ───────────────────────────────────────────────────
CERT_PATTERNS = [
    r"\bUdemy\s*[–\-—]\s*[\w\s&()]+",
    r"\bCoursera\s*[–\-—]\s*[\w\s&()]+",
    r"\bedX\s*[–\-—]\s*[\w\s&()]+",
    r"\bLinkedIn\s+Learning\s*[–\-—]\s*[\w\s&()]+",
    r"\bFreeCodeCamp\s*[–\-—]\s*[\w\s&()]+",
    r"\bCisco\s*[–\-—]\s*[\w\s&()+]+",
    r"\bCisco\s+Certified[\w\s&()]+",
    r"\bCisco\s+(?:CCNA|CCNP|CCIE|CyberOps)[\w\s]*",
    r"\bAWS\s+Certified\s+[\w\s]+",
    r"\bGoogle\s+Cloud\s+[\w\s]+(?:Associate|Professional|Engineer|Developer)",
    r"\bProfessional\s+(?:Data\s+Engineer|Machine\s+Learning\s+Engineer|Cloud\s+Architect)",
    r"\bMicrosoft\s+Certified[\w\s:]+",
    r"\bAZ-\d{3}\b",
    r"\bCompTIA\s+[\w+]+",
    r"\bA\+\b", r"\bNetwork\+\b", r"\bSecurity\+\b",
    r"\bCertified\s+Kubernetes\s+(?:Administrator|Developer|Security\s+Specialist)",
    r"\bCKA\b", r"\bCKAD\b", r"\bCKS\b",
    r"\bCertified\s+Ethical\s+Hacker",
    r"\bCEH\b", r"\bCISSP\b", r"\bCISM\b", r"\bOSCP\b",
    r"\bPMP\b", r"\bPRINCE2[\w\s]*", r"\bScrumMaster\b", r"\bCSM\b", r"\bPSM\b",
    r"\bOracle\s+Certified[\w\s]+",
    r"\bRed\s+Hat\s+Certified[\w\s]+",
    r"\bRHCE\b", r"\bRHCSA\b",
    r"\bTensorFlow\s+Developer\s+Certificate",
    r"\bCertified\s+Information\s+Technology[\w\s()]+",
    r"\bCertified\s+Software\s+Development\s+Professional",
    r"\bCSDP\b",
    r"(?:Responsive\s+Web\s+Design|Relational\s+Database|JavaScript\s+Algorithms"
    r"|Front\s+End\s+Development|Back\s+End\s+Development|Data\s+Visualization"
    r"|Scientific\s+Computing|Machine\s+Learning|Quality\s+Assurance"
    r"|College\s+Algebra)\s+(?:FreeCodeCamp|Certificate)?",
    r"\bFlutter\s+Program\s+Certif\w+",
    r"\bITF\+",
    r"\bISO\s+\d+[\w\s]*(?:Lead\s+Auditor|Lead\s+Implementer)?",
]

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
    # FIX BUG 5: generic non-technical phrases from experience/support roles
    "customer service", "customer communication", "cross-team collaboration",
    "documentation", "knowledge sharing", "analytical problem solving",
    "it troubleshooting", "system support", "it support",
    "project coordination", "technical documentation",
}

CERT_PROVIDER_WORDS: set[str] = {
    "freecodecamp", "udemy", "coursera", "edx", "cisco", "microsoft",
    "google", "amazon", "oracle", "linkedin", "certified", "certification",
    "certificate",
}

KNOWN_SKILLS: list[str] = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Kotlin", "Swift", "R", "Scala", "PHP", "Ruby", "MATLAB", "Dart",
    "Bash", "Shell", "PowerShell",
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Express.js", "Next.js", "Nuxt.js", "Svelte", "jQuery",
    "Laravel", "Ruby on Rails", "ASP.NET",
    "Flutter", "React Native", "Android", "iOS", "SwiftUI", "Jetpack Compose",
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Pandas", "NumPy",
    "FAISS", "HuggingFace", "spaCy", "NLTK", "OpenCV", "Spark", "Hadoop",
    "Airflow", "dbt", "Kafka", "Flink", "Matplotlib", "Seaborn",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Helm",
    "Linux", "Ubuntu", "Nginx", "Apache",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "DynamoDB", "Firebase", "Firestore", "BigQuery", "Snowflake", "SQLite",
    "TCP/IP", "GraphQL", "gRPC", "OAuth", "JWT", "REST API", "RESTful APIs",
    "Penetration Testing", "SIEM", "Firewalls",
    "Git", "GitHub", "GitLab", "Jira", "Postman", "Swagger", "Figma",
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
    # Engineering / non-CS skills
    "AutoCAD", "SolidWorks", "AutoCAD Plant 3D", "HVAC Design",
    "Preventive Maintenance", "Root Cause Analysis", "GD&T", "FEA Analysis",
    "Piping Design", "Engineering Calculations", "Cost Reduction",
    "Project Management", "CAD", "Process Flow Diagrams",
    # Networking / distributed systems
    "Paxos", "Raft", "Microservices Architecture",
    # Misc
    "Microsoft Azure", "Version Control",
]


# ─── Section extraction ───────────────────────────────────────────────────────

def _extract_section(text: str, header_patterns: list[str], label: str = "section") -> str | None:
    combined_header = "|".join(f"(?:{p})" for p in header_patterns)
    combined_next   = "|".join(f"(?:{p})" for p in NEXT_SECTION_HEADERS)

    header_match = re.search(combined_header, text, re.IGNORECASE | re.MULTILINE)
    if not header_match:
        return None

    section_start = header_match.end()
    section_text  = text[section_start:]

    next_match = re.search(combined_next, section_text, re.IGNORECASE | re.MULTILINE)
    if next_match:
        stopped_header = section_text[next_match.start():next_match.start()+30].strip()
        logger.debug(f"{label} section ended at: '{stopped_header}'")
        section_text = section_text[:next_match.start()]

    return section_text.strip() or None


def _extract_skills_section(text: str) -> str | None:
    return _extract_section(text, SKILLS_SECTION_HEADERS, "skills")


def _extract_certs_section(text: str) -> str | None:
    return _extract_section(text, CERT_SECTION_HEADERS, "certs")


# ─── NLP helpers ─────────────────────────────────────────────────────────────

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
    cleaned: list[str] = []
    for c in raw_certs:
        c = " ".join(c.split()).strip(" \u2013-\u2014.,;:")
        if len(c) > 2:
            cleaned.append(c)

    cleaned.sort(key=len, reverse=True)
    kept: list[str] = []
    for cert in cleaned:
        cert_lower = cert.lower()
        if not any(cert_lower in kept_cert.lower() for kept_cert in kept):
            kept.append(cert)
    return sorted(kept)


# ─── Main processor ───────────────────────────────────────────────────────────

class NLPProcessor:

    def __init__(self, extra_skills: list[str] | None = None):
        self.nlp = _load_nlp_model()
        self.normalizer = SkillNormalizer()
        all_skills = KNOWN_SKILLS + (extra_skills or [])
        self.matcher = _build_phrase_matcher(self.nlp, all_skills)

    def extract_certifications(self, text: str) -> list[str]:
        raw: list[str] = []
        for pattern in CERT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                match_text = m.group().strip()
                if len(match_text) > 3:
                    raw.append(match_text)
        return _deduplicate_certs(raw)

    def extract_skills(self, text: str, cert_strings: list[str]) -> list[str]:
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
            span_text = doc[start:end].text.strip()

            # FIX BUG 5: enforce minimum meaningful length
            if len(span_text) < 2:
                continue

            if span_text.lower() in SOFT_SKILLS_EXCLUDE:
                continue

            span_words = span_text.lower().split()
            if all(w in cert_context_words for w in span_words):
                continue

            span_lower = span_text.lower()
            if any(span_lower in cert.lower() and
                   any(w in cert.lower() for w in CERT_PROVIDER_WORDS)
                   for cert in cert_strings):
                continue

            raw_skills.add(span_text)

        normalized = [self.normalizer.normalize(s) for s in raw_skills]
        return sorted(set(filter(None, normalized)))

    def process(self, raw_text: str) -> dict:
        cleaned = self._clean_text(raw_text)

        # ── Certifications ────────────────────────────────────────────────────
        cert_section = _extract_certs_section(cleaned)
        cert_text    = cert_section if cert_section else cleaned
        certifications = self.extract_certifications(cert_text)
        logger.info(
            f"Cert source: {'section' if cert_section else 'full text'} "
            f"→ {len(certifications)} certifications"
        )

        # ── Skills ────────────────────────────────────────────────────────────
        skill_section = _extract_skills_section(cleaned)
        if skill_section:
            logger.info(
                f"Skills section found ({len(skill_section)} chars) — "
                "extracting only from Skills section"
            )
            skills = self.extract_skills(skill_section, certifications)
        else:
            logger.warning(
                "No Skills section detected — falling back to full-text extraction"
            )
            skills = self.extract_skills(cleaned, certifications)

        logger.info(f"NLP extracted {len(skills)} skills, {len(certifications)} certifications.")
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