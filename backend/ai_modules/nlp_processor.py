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

FIXES (v3):
  BUG 6 - Projects section skills were bleeding into extraction when no Skills section
           was found. Fallback now extracts from EXPERIENCE + SUMMARY only, explicitly
           stripping the Projects block. This was the root cause of false 7-12% match
           scores — project tech terms created a plausible embedding that FAISS matched
           to loosely related jobs, and keyword overlap then gave matched_count=1,
           bypassing ZERO_SKILL_SCORE_CAP.
  BUG 7 - Added PROJECT_SECTION_HEADERS and _strip_projects_from_text() to surgically
           remove the Projects section before any fallback full-text extraction.
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

# FIX BUG 6/7: Projects section patterns — used to strip project blocks before
# fallback extraction so project-mentioned tech terms don't pollute skill lists.
PROJECT_SECTION_HEADERS = [
    r"(?:^|\n|\s{2,})(?:PROJECTS?|PERSONAL\s+PROJECTS?|ACADEMIC\s+PROJECTS?"
    r"|SIDE\s+PROJECTS?|KEY\s+PROJECTS?|NOTABLE\s+PROJECTS?"
    r"|PROJECT\s+EXPERIENCE|PORTFOLIO)\s*(?:\n|:|\s{2,})",
]

# When no Skills section is found, we extract only from Summary + Experience,
# never from Projects (too narrative) or Education (too broad).
SAFE_FALLBACK_SECTION_HEADERS = [
    r"(?:^|\n|\s{2,})(?:SUMMARY|PROFESSIONAL\s+SUMMARY|CAREER\s+SUMMARY"
    r"|PROFILE|ABOUT\s+ME?|OVERVIEW)\s*(?:\n|:|\s{2,})",
    r"(?:^|\n|\s{2,})(?:EXPERIENCE|WORK\s+EXPERIENCE|EMPLOYMENT\s+HISTORY"
    r"|PROFESSIONAL\s+EXPERIENCE|WORK\s+HISTORY)\s*(?:\n|:|\s{2,})",
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
    r"\bUdemy\s*[–\-—]\s*[^\n]+",
    r"\bCoursera\s*[–\-—]\s*[^\n]+",
    r"\bedX\s*[–\-—]\s*[^\n]+",
    r"\bLinkedIn\s+Learning\s*[–\-—]\s*[^\n]+",
    r"\bFreeCodeCamp\s*[–\-—]\s*[^\n]+",
    r"\bCisco\s*[–\-—]\s*[^\n]+",
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
    # Generic non-technical phrases from experience/support roles
    "customer service", "customer communication", "cross-team collaboration",
    "documentation", "knowledge sharing", "analytical problem solving",
    "it troubleshooting", "system support", "it support",
    "project coordination", "technical documentation",
    # Security/auth terms that are too generic when appearing in project bullets
    # (a candidate who "designed authentication" in a project does not necessarily
    #  have Authentication as a standalone skill — it's a feature they implemented)
    "authentication", "authorization", "role-based access control",
    "role based access control",
}

CERT_PROVIDER_WORDS: set[str] = {
    "freecodecamp", "udemy", "coursera", "edx", "cisco", "microsoft",
    "google", "amazon", "oracle", "linkedin", "certified", "certification",
    "certificate",
}

KNOWN_SKILLS: list[str] = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Kotlin", "Swift", "R", "Scala", "PHP", "Ruby", "MATLAB", "Dart",
    "Bash", "Shell", "PowerShell", "Groovy", "Perl", "Haskell", "Elixir",
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Express.js", "Next.js", "Nuxt.js", "Svelte", "jQuery",
    "Laravel", "Ruby on Rails", "ASP.NET", "Gatsby", "Remix", "Astro",
    "Flutter", "React Native", "Android", "iOS", "SwiftUI", "Jetpack Compose",
    "Xamarin", "Ionic", "Capacitor",
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Pandas", "NumPy",
    "FAISS", "HuggingFace", "spaCy", "NLTK", "OpenCV", "Spark", "Hadoop",
    "Airflow", "dbt", "Kafka", "Flink", "Matplotlib", "Seaborn", "Plotly",
    "LangChain", "LlamaIndex", "Transformers", "XGBoost", "LightGBM",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Helm", "ArgoCD",
    "Linux", "Ubuntu", "Nginx", "Apache", "Pulumi", "CDK",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "DynamoDB", "Firebase", "Firestore", "BigQuery", "Snowflake", "SQLite",
    "MariaDB", "Oracle DB", "CockroachDB", "Supabase", "PlanetScale",
    "TCP/IP", "GraphQL", "gRPC", "OAuth", "JWT", "REST API", "RESTful APIs",
    "WebSockets", "MQTT", "RabbitMQ", "Celery", "Nginx", "Load Balancing",
    "Penetration Testing", "SIEM", "Firewalls", "Zero Trust", "OWASP",
    "Git", "GitHub", "GitLab", "Jira", "Postman", "Swagger", "Figma",
    "Bitbucket", "Confluence", "Notion", "Linear", "Trello",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Structures", "Algorithms", "Microservices", "CI/CD",
    "Agile", "Scrum", "Kanban", "System Design", "OOP", "Functional Programming",
    "Data Engineering", "MLOps", "DevOps", "Cloud Architecture", "SRE",
    "CRUD", "MERN Stack", "MEAN Stack", "LAMP Stack",
    "Full Stack Development", "Backend Development", "Frontend Development",
    "Mobile Development", "Web Development", "API Development",
    "Unit Testing", "Integration Testing", "TDD", "BDD", "Responsive Design",
    "Version Control", "Authentication", "Authorization",
    "Role-Based Access Control", "Real-Time Data", "WebSockets",
    "Flutter & Firebase Development", "Flutter & Firebase",
    "Git Version Control", "REST APIs & Backend Development",
    "CRUD Application Development", "Object-Oriented Programming",
    "Blockchain", "Smart Contracts", "Solidity", "Web3",
    "Embedded Systems", "IoT", "Cybersecurity", "Network Security",
    "Data Analysis", "Data Visualization", "Business Intelligence",
    "SQL", "NoSQL", "Data Modeling", "ETL", "Data Pipelines",
    "AutoCAD", "SolidWorks", "AutoCAD Plant 3D", "HVAC Design",
    "Preventive Maintenance", "Root Cause Analysis", "GD&T", "FEA Analysis",
    "Piping Design", "Engineering Calculations", "Cost Reduction",
    "Project Management", "CAD", "Process Flow Diagrams",
    "Paxos", "Raft", "Microservices Architecture",
    "Microsoft Azure", "Amazon Web Services", "Google Cloud Platform",
    "Serverless", "Lambda Functions", "Cloud Functions",
    "HTML", "CSS", "SASS", "SCSS", "Tailwind CSS", "Bootstrap",
    "Material UI", "Chakra UI", "Styled Components",
    "Webpack", "Vite", "Babel", "ESLint", "Prettier",
    "Testing", "Jest", "Pytest", "Mocha", "Cypress", "Selenium",
    "OpenAI API", "Prompt Engineering", "Vector Databases",
    "Pinecone", "Weaviate", "ChromaDB", "Qdrant",
    "Distributed Systems", "System Architecture", "Design Patterns",
    "Clean Code", "SOLID Principles", "Domain-Driven Design",
    "Event-Driven Architecture", "Message Queues",
    "Monitoring", "Observability", "Prometheus", "Grafana", "Datadog",
    "ELK Stack", "Splunk", "New Relic",
]


# ─── Section extraction ───────────────────────────────────────────────────────

def _expand_parentheticals(text: str) -> str:
    """
    Expand parenthetical skill lists so individual components are extracted.

    Example:
      'MERN Stack (MongoDB, Express.js, React, Node.js)'
      → 'MERN Stack (MongoDB, Express.js, React, Node.js)\nMongoDB\nExpress.js\nReact\nNode.js'

    This ensures that PhraseMatcher sees each parenthetical component as a
    standalone token sequence, so MongoDB is matched even though it only
    appears inside a parenthetical on the resume.

    Only expands parentheticals that contain commas (i.e. lists), not
    single-word clarifications like "Python (3.x)" or dates like "(2024)".
    """
    def _expand(match: re.Match) -> str:
        original = match.group(0)
        inner    = match.group(1).strip()
        if "," not in inner:
            return original  # not a list — leave as-is
        # Split on commas, strip each component
        parts = [p.strip().strip("()") for p in inner.split(",")]
        parts = [p for p in parts if len(p) > 1]
        if not parts:
            return original
        return original + "\n" + "\n".join(parts)

    return re.sub(r"\(([^)]{4,})\)", _expand, text)


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


def _strip_projects_from_text(text: str) -> str:
    """
    FIX BUG 6/7: Remove the Projects section from resume text entirely.
    Projects describe what the candidate *built*, not what they *know* —
    tech terms mentioned there are unreliable signals of proficiency.
    Stripping prevents project tech terms from inflating the skill list
    and creating false FAISS embedding similarity scores.
    """
    combined_project = "|".join(f"(?:{p})" for p in PROJECT_SECTION_HEADERS)
    combined_next    = "|".join(f"(?:{p})" for p in NEXT_SECTION_HEADERS)

    result = text
    for match in re.finditer(combined_project, text, re.IGNORECASE | re.MULTILINE):
        section_text = text[match.end():]
        end_match = re.search(combined_next, section_text, re.IGNORECASE | re.MULTILINE)
        if end_match:
            section_end = match.end() + end_match.start()
        else:
            section_end = len(text)
        # Replace the projects block with whitespace to preserve positions
        result = result[:match.start()] + " " * (section_end - match.start()) + result[section_end:]
        logger.debug(f"Stripped Projects section ({section_end - match.start()} chars)")
        break  # only strip the first Projects block

    return result


def _extract_safe_fallback_text(text: str) -> str:
    """
    FIX BUG 6/7: When no Skills section is found, extract ONLY from
    Summary and Experience sections — not from Projects or Education.
    Falls back to full-text-minus-projects if those sections aren't found.
    """
    text_no_projects = _strip_projects_from_text(text)

    combined_safe = "|".join(f"(?:{p})" for p in SAFE_FALLBACK_SECTION_HEADERS)
    combined_next = "|".join(f"(?:{p})" for p in NEXT_SECTION_HEADERS)

    # Collect all safe section matches
    safe_chunks: list[str] = []
    for match in re.finditer(combined_safe, text_no_projects, re.IGNORECASE | re.MULTILINE):
        section_body = text_no_projects[match.end():]
        end_match = re.search(combined_next, section_body, re.IGNORECASE | re.MULTILINE)
        chunk = section_body[:end_match.start()] if end_match else section_body
        if chunk.strip():
            safe_chunks.append(chunk.strip())

    if safe_chunks:
        logger.info(f"Fallback: extracting skills from {len(safe_chunks)} safe section(s) (Summary/Experience only)")
        return "\n\n".join(safe_chunks)

    # Last resort: full text minus projects
    logger.warning("No Summary/Experience sections found — using full text minus Projects for fallback")
    return text_no_projects



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
    Deduplicates cert strings.
    1. Substring containment: keeps the longer version.
    2. Word-overlap: deduplicates same cert in different word order
       e.g. "Responsive Web Design FreeCodeCamp" vs
            "FreeCodeCamp – Responsive Web Design"
    """
    cleaned: list[str] = []
    for c in raw_certs:
        c = " ".join(c.split()).strip(" \u2013-\u2014.,;:")
        if len(c) > 2:
            cleaned.append(c)

    cleaned.sort(key=len, reverse=True)
    kept: list[str] = []
    for cert in cleaned:
        cert_lower = cert.lower()
        cert_words = set(w for w in re.split(r"\W+", cert_lower) if len(w) > 2)

        if any(cert_lower in kept_cert.lower() for kept_cert in kept):
            continue

        is_dup = False
        for kept_cert in kept:
            kept_words = set(w for w in re.split(r"\W+", kept_cert.lower()) if len(w) > 2)
            if cert_words and kept_words:
                overlap = len(cert_words & kept_words) / min(len(cert_words), len(kept_words))
                if overlap >= 0.80:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(cert)

    return sorted(kept)


# ─── Main processor ───────────────────────────────────────────────────────────

class NLPProcessor:

    # Class-level dynamic matcher injected by main.py after Firestore job corpus
    # is loaded. Falls back to static KNOWN_SKILLS matcher if not yet injected.
    _dynamic_matcher = None

    def __init__(self, extra_skills: list[str] | None = None):
        self.nlp = _load_nlp_model()
        self.normalizer = SkillNormalizer()
        all_skills = KNOWN_SKILLS + (extra_skills or [])
        self._static_matcher = _build_phrase_matcher(self.nlp, all_skills)

    @property
    def matcher(self):
        # Prefer the dynamic matcher (expanded with Firestore job skills)
        # if main.py has injected it; otherwise use the static one.
        return NLPProcessor._dynamic_matcher or self._static_matcher

    @matcher.setter
    def matcher(self, value):
        self._static_matcher = value

    def extract_certifications(self, text: str) -> list[str]:
        raw: list[str] = []
        for pattern in CERT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                match_text = m.group().strip()
                if len(match_text) > 3:
                    raw.append(match_text)
        return _deduplicate_certs(raw)

    def extract_skills(self, text: str, cert_strings: list[str]) -> list[str]:
        # Expand parenthetical skill lists so individual components are visible
        # e.g. "MERN Stack (MongoDB, Express.js, React, Node.js)"
        #   → also presents "MongoDB", "Express.js", "React", "Node.js" as tokens
        text = _expand_parentheticals(text)
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
        # Certifications are extracted from the full text (including Projects)
        # since certs are typically listed by name, not buried in narrative.
        cert_section = _extract_certs_section(cleaned)
        cert_text    = cert_section if cert_section else cleaned
        certifications = self.extract_certifications(cert_text)
        logger.info(
            f"Cert source: {'section' if cert_section else 'full text'} "
            f"→ {len(certifications)} certifications"
        )

        # ── Skills ────────────────────────────────────────────────────────────
        # Priority 1: Dedicated Skills / Technical Skills section (most reliable)
        skill_section = _extract_skills_section(cleaned)
        if skill_section:
            logger.info(
                f"Skills section found ({len(skill_section)} chars) — "
                "extracting only from Skills section"
            )
            skills = self.extract_skills(skill_section, certifications)

            # If Skills section exists but yields very few results (< 3),
            # augment with Summary/Experience fallback to avoid under-extraction
            # on sparse or unusually formatted skills sections.
            if len(skills) < 3:
                logger.warning(
                    f"Skills section yielded only {len(skills)} skills — "
                    "augmenting with Summary/Experience fallback"
                )
                safe_text = _extract_safe_fallback_text(cleaned)
                fallback_skills = self.extract_skills(safe_text, certifications)
                # Merge without duplicates, preserving order
                existing_lower = {s.lower() for s in skills}
                for s in fallback_skills:
                    if s.lower() not in existing_lower:
                        skills.append(s)
                        existing_lower.add(s.lower())
                skills = sorted(skills)
        else:
            # FIX BUG 6/7: DO NOT fall back to full-text extraction.
            # Full-text includes the Projects section, which contains tech terms
            # used in a narrative context ("built X using React") — these create
            # false skill signals that produce inaccurate FAISS embeddings and
            # bypass the ZERO_SKILL_SCORE_CAP via matched_count=1 keyword hits.
            #
            # Instead: extract from Summary + Experience sections only.
            # These are where candidates intentionally list what they know.
            logger.warning(
                "No Skills section detected — "
                "falling back to Summary/Experience extraction (Projects excluded)"
            )
            safe_text = _extract_safe_fallback_text(cleaned)
            skills = self.extract_skills(safe_text, certifications)

        logger.info(f"NLP extracted {len(skills)} skills, {len(certifications)} certifications.")
        return {
            "skills": skills,
            "certifications": certifications,
            "word_count": len(cleaned.split()),
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        # Remove noise
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", " ", text)
        text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)

        # FIX: collapse horizontal whitespace ONLY — preserve newlines.
        # The old re.sub(r'\s+', ' ') collapsed newlines into spaces, turning
        # 'SKILLS\nMERN Stack' into 'SKILLS MERN Stack', which broke every
        # section-header regex that relied on \n or \s{2,} after the keyword.
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3+ newlines to 2 to keep section breaks intact
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix two-column PDF cert merging: insert \n before known providers
        # when they appear mid-line (no preceding newline).
        # e.g. "Udemy – Flutter CertifiedFreeCodeCamp – Web Design"
        #   -> "Udemy – Flutter Certified\nFreeCodeCamp – Web Design"
        _PROVIDERS = (
            r"(?<=[^\n])(FreeCodeCamp|Udemy|Coursera|edX|LinkedIn\s+Learning"
            r"|Cisco|Microsoft|Google|Amazon|Oracle|CompTIA|IBM|Meta|Apple)"
        )
        text = re.sub(_PROVIDERS, r"\n\1", text)

        # Strip trailing spaces per line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()