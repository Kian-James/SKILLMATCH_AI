"""
utils/skill_normalizer.py
Normalizes skill names using synonym mapping and string cleanup.
Prevents duplicates like "JS" vs "JavaScript", "ML" vs "Machine Learning".
"""

from __future__ import annotations

import re


# Synonym map: canonical skill name → list of aliases (all lowercase for matching)
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
    "Continuous Integration": ["ci", "continuous integration"],
    "Continuous Deployment": ["cd", "continuous deployment", "continuous delivery"],
    "CI/CD": ["cicd", "ci cd", "ci/cd"],
    "Application Programming Interface": ["api", "apis", "rest api", "restful api"],
    "GraphQL": ["graphql"],
    "SQL": ["sql", "structured query language"],
    "NoSQL": ["nosql", "no-sql"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "scikit-learn": ["sklearn", "scikit learn", "scikit-learn"],
    "HuggingFace": ["huggingface", "hugging face", "hf transformers"],
    "Agile": ["agile methodology", "agile development"],
    "Scrum": ["scrum methodology"],
    "Git": ["git", "version control", "github", "gitlab", "bitbucket"],
    "Linux": ["linux", "ubuntu", "debian", "centos", "unix"],
    "Data Structures": ["data structures", "dsa", "data structures and algorithms"],
    "Object-Oriented Programming": ["oop", "object oriented", "object-oriented programming"],
    "Microservices": ["microservice", "microservices architecture"],
}

# Reverse map: alias → canonical
_REVERSE_MAP: dict[str, str] = {}
for canonical, aliases in SKILL_SYNONYMS.items():
    for alias in aliases:
        _REVERSE_MAP[alias.lower()] = canonical


class SkillNormalizer:
    """
    Normalizes raw skill strings to canonical form.
    Also handles basic cleaning (punctuation stripping, whitespace).
    """

    def normalize(self, raw_skill: str) -> str | None:
        """
        Normalize a raw skill string to its canonical name.
        Returns None if the skill is noise (too short, numeric-only, etc.).
        """
        cleaned = self._clean(raw_skill)
        if not cleaned or len(cleaned) < 2:
            return None

        # Check synonym map
        lower = cleaned.lower()
        if lower in _REVERSE_MAP:
            return _REVERSE_MAP[lower]

        # Return title-cased version if not in map
        return cleaned.strip()

    def normalize_job_title(self, title: str) -> str:
        """
        Normalize job titles to avoid duplicates.
        e.g. "Software Engineer (Python)" → "Software Engineer"
             "Python Software Engineer" → "Software Engineer"
        """
        # Remove parenthetical qualifiers
        title = re.sub(r"\s*\(.*?\)", "", title).strip()
        # Remove leading language prefix ("Python Software Engineer" → "Software Engineer")
        # This is a heuristic — the LLM also helps with this
        title = title.strip().title()
        return title

    def deduplicate_skills(self, skills: list[str]) -> list[str]:
        """
        Normalize all skills in a list and remove duplicates.
        """
        seen: set[str] = set()
        result: list[str] = []
        for s in skills:
            normalized = self.normalize(s)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return sorted(result)

    @staticmethod
    def _clean(text: str) -> str:
        """Remove extra punctuation and whitespace."""
        text = re.sub(r"[^\w\s.#+/]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
