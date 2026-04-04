"""
utils/skill_normalizer.py
Normalizes skill names and removes duplicates including:
- Case duplicates: "MERN Stack" vs "MERN stack"
- Substring duplicates: "CRUD" inside "CRUD Application Development" → keep longer
- Synonym duplicates: "JS" → "JavaScript"
"""

from __future__ import annotations

import re


SKILL_SYNONYMS: dict[str, list[str]] = {
    "JavaScript": ["js", "javascript", "java script", "ecmascript", "es6", "es2015"],
    "TypeScript": ["ts", "typescript"],
    "Python": ["python3", "python 3", "py"],
    "Machine Learning": ["ml", "machine learning", "ml/ai", "ai/ml"],
    "Deep Learning": ["dl", "deep learning"],
    "Natural Language Processing": ["nlp", "natural language processing"],
    "Computer Vision": ["cv", "computer vision"],
    "Artificial Intelligence": ["ai", "artificial intelligence"],
    "C++": ["cpp", "c plus plus", "c/c++"],
    "C#": ["csharp", "c sharp", "dotnet c#"],
    ".NET": ["dotnet", ".net", "dot net"],
    "Node.js": ["nodejs", "node js", "node"],
    "React": ["reactjs", "react.js", "react js"],
    "Vue.js": ["vuejs", "vue js", "vue"],
    "Angular": ["angularjs", "angular js"],
    "Kubernetes": ["k8s", "kubernetes"],
    "Docker": ["docker", "dockerfile"],
    "PostgreSQL": ["postgres", "postgresql", "pg"],
    "MongoDB": ["mongo", "mongodb"],
    "Amazon Web Services": ["aws", "amazon web services"],
    "Google Cloud Platform": ["gcp", "google cloud", "google cloud platform"],
    "Microsoft Azure": ["azure", "ms azure", "microsoft azure"],
    "CI/CD": ["cicd", "ci cd", "ci/cd"],
    "GraphQL": ["graphql"],
    "SQL": ["sql", "structured query language"],
    "NoSQL": ["nosql", "no-sql"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "scikit-learn": ["sklearn", "scikit learn", "scikit-learn"],
    "HuggingFace": ["huggingface", "hugging face", "hf transformers"],
    "Agile": ["agile methodology", "agile development"],
    "Scrum": ["scrum methodology"],
    # Git aliases — but NOT "Git Version Control" (that's a valid standalone skill)
    "Git": ["git"],
    "Linux": ["linux", "ubuntu", "debian", "centos", "unix"],
    "Data Structures": ["data structures", "dsa", "data structures and algorithms"],
    "Object-Oriented Programming": ["oop", "object oriented", "object-oriented programming"],
    "Microservices": ["microservice", "microservices architecture"],
    # MERN Stack aliases
    "MERN Stack": ["mern stack", "mern"],
    # Flutter & Firebase aliases
    "Flutter & Firebase Development": [
        "flutter & firebase development",
        "flutter firebase development",
        "flutter & firebase",
        "flutter firebase",
    ],
    # CRUD aliases
    "CRUD Application Development": [
        "crud application development",
        "crud based systems",
        "crud operations",
        "crud functionality",
        "crud",
    ],
    # REST API aliases
    "RESTful APIs": [
        "restful apis", "rest apis", "rest api", "restful api",
        "rest apis & backend development",
        "application programming interface",
    ],
}

# Reverse map: alias → canonical
_REVERSE_MAP: dict[str, str] = {}
for canonical, aliases in SKILL_SYNONYMS.items():
    for alias in aliases:
        _REVERSE_MAP[alias.lower()] = canonical


class SkillNormalizer:

    def normalize(self, raw_skill: str) -> str | None:
        cleaned = self._clean(raw_skill)
        if not cleaned or len(cleaned) < 2:
            return None
        lower = cleaned.lower()
        if lower in _REVERSE_MAP:
            return _REVERSE_MAP[lower]
        return cleaned.strip()

    def normalize_job_title(self, title: str) -> str:
        title = re.sub(r"\s*\(.*?\)", "", title).strip()
        title = title.strip().title()
        return title

    def deduplicate_skills(self, skills: list[str]) -> list[str]:
        """
        Full deduplication pipeline:
        1. Normalize synonyms (MERN stack → MERN Stack, CRUD → CRUD Application Development)
        2. Remove case duplicates (case-insensitive seen set)
        3. Remove substring duplicates — if "CRUD" is already covered by
           "CRUD Application Development", drop the shorter one
        """
        # Step 1: normalize
        normalized: list[str] = []
        seen_lower: set[str] = set()
        for s in skills:
            norm = self.normalize(s)
            if norm and norm.lower() not in seen_lower:
                seen_lower.add(norm.lower())
                normalized.append(norm)

        # Step 2: remove substring duplicates
        # Sort longest first so the more descriptive form is kept
        normalized.sort(key=len, reverse=True)
        final: list[str] = []
        for skill in normalized:
            skill_lower = skill.lower()
            # Keep this skill only if no already-kept skill contains it as a substring
            # e.g. drop "CRUD" if "CRUD Application Development" is already kept
            if not any(skill_lower in kept.lower() for kept in final):
                final.append(skill)

        return sorted(final)

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[^\w\s.#+/&]", " ", text)
        return re.sub(r"\s+", " ", text).strip()