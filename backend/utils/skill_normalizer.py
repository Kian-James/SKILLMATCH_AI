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
    "JavaScript": ["js", "javascript", "java script", "ecmascript", "es6", "es2015", "es6+", "vanilla js"],
    "TypeScript": ["ts", "typescript", "ts/js", "js/ts"],
    "Python": ["python3", "python 3", "py", "python programming"],
    "Machine Learning": ["ml", "machine learning", "ml/ai", "ai/ml", "ml engineering"],
    "Deep Learning": ["dl", "deep learning", "neural networks", "neural network"],
    "Natural Language Processing": ["nlp", "natural language processing", "text processing"],
    "Computer Vision": ["cv", "computer vision", "image processing", "image recognition"],
    "Artificial Intelligence": ["ai", "artificial intelligence", "generative ai", "gen ai"],
    "C++": ["cpp", "c plus plus", "c/c++", "c and c++"],
    "C#": ["csharp", "c sharp", "dotnet c#", "c# .net"],
    ".NET": ["dotnet", ".net", "dot net", "asp.net core"],
    "Node.js": ["nodejs", "node js", "node", "node.js runtime"],
    "React": ["reactjs", "react.js", "react js", "react framework"],
    "Vue.js": ["vuejs", "vue js", "vue", "vue 3"],
    "Angular": ["angularjs", "angular js", "angular 2+", "angular framework"],
    "Kubernetes": ["k8s", "kubernetes", "kube"],
    "Docker": ["docker", "dockerfile", "docker compose", "containerization", "containers"],
    "PostgreSQL": ["postgres", "postgresql", "pg", "postgres db"],
    "MongoDB": ["mongo", "mongodb", "mongo db"],
    "Amazon Web Services": ["aws", "amazon web services", "amazon aws"],
    "Google Cloud Platform": ["gcp", "google cloud", "google cloud platform"],
    "Microsoft Azure": ["azure", "ms azure", "microsoft azure", "azure cloud"],
    "CI/CD": ["cicd", "ci cd", "ci/cd", "continuous integration", "continuous deployment",
              "continuous integration/continuous deployment"],
    "GraphQL": ["graphql", "graph ql"],
    "SQL": ["sql", "structured query language", "sql queries", "sql programming"],
    "NoSQL": ["nosql", "no-sql", "no sql"],
    "TensorFlow": ["tensorflow", "tf", "tensorflow 2"],
    "PyTorch": ["pytorch", "torch", "py torch"],
    "scikit-learn": ["sklearn", "scikit learn", "scikit-learn"],
    "HuggingFace": ["huggingface", "hugging face", "hf transformers", "hugging face transformers"],
    "Agile": ["agile methodology", "agile development", "agile practices", "agile/scrum"],
    "Scrum": ["scrum methodology", "scrum framework"],
    "Git": ["git", "git scm"],
    "Linux": ["linux", "ubuntu", "debian", "centos", "unix", "linux/unix"],
    "Data Structures": ["data structures", "dsa", "data structures and algorithms",
                        "data structures & algorithms"],
    "Object-Oriented Programming": ["oop", "object oriented", "object-oriented programming",
                                    "object oriented programming", "oops"],
    "Microservices": ["microservice", "microservices architecture", "micro services",
                      "microservices-based"],
    "MERN Stack": ["mern stack", "mern"],
    "Flutter & Firebase Development": [
        "flutter & firebase development", "flutter firebase development",
        "flutter & firebase", "flutter firebase",
    ],
    "CRUD Application Development": [
        "crud application development", "crud based systems",
        "crud operations", "crud functionality", "crud",
    ],
    "RESTful APIs": [
        "restful apis", "rest apis", "rest api", "restful api",
        "rest apis & backend development", "restful web services",
        "application programming interface", "api development",
        "rest services", "http apis",
    ],
    "HTML": ["html", "html5", "hypertext markup language"],
    "CSS": ["css", "css3", "cascading style sheets"],
    "Tailwind CSS": ["tailwind", "tailwindcss", "tailwind css"],
    "Bootstrap": ["bootstrap", "bootstrap css", "bootstrap framework"],
    "SQL": ["sql", "mysql queries", "sql queries", "relational databases"],
    "Unit Testing": ["unit testing", "unit tests", "automated testing", "test automation"],
    "Version Control": ["version control", "source control", "vcs"],
    "Authentication": ["authentication", "auth", "user authentication"],
    "Authorization": ["authorization", "authz", "access control"],
    "DevOps": ["devops", "dev ops", "development operations"],
    "Data Analysis": ["data analysis", "data analytics", "analytics"],
    "Data Visualization": ["data visualization", "data viz", "dataviz"],
    "API Development": ["api development", "api design", "backend api", "rest api development"],
    "Backend Development": ["backend development", "back-end development", "server-side development",
                            "backend engineering"],
    "Frontend Development": ["frontend development", "front-end development", "client-side development",
                             "frontend engineering", "ui development"],
    "Full Stack Development": ["full stack development", "full-stack development",
                               "fullstack development", "full stack engineering"],
    "Mobile Development": ["mobile development", "mobile app development",
                           "android development", "ios development"],
    "Cloud Architecture": ["cloud architecture", "cloud computing", "cloud infrastructure",
                           "cloud engineering"],
    "System Design": ["system design", "software architecture", "software design",
                      "architectural design"],
    "Cybersecurity": ["cybersecurity", "cyber security", "information security",
                      "infosec", "network security"],
    "Project Management": ["project management", "pm", "technical project management"],
    "Serverless": ["serverless", "serverless architecture", "faas", "function as a service"],
    "Distributed Systems": ["distributed systems", "distributed computing",
                            "distributed architecture"],
    "Design Patterns": ["design patterns", "software design patterns", "architectural patterns"],
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