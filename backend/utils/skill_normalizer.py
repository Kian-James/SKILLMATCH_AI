"""
utils/skill_normalizer.py

Normalizes skill names using a two-tier approach:

  Tier 1 — Static + LLM-expanded synonym map (zero-cost exact lookup)
    At startup, load_dynamic_synonyms() fetches LLM-generated aliases stored
    in Firestore (collection: skill_synonyms) and merges them into _REVERSE_MAP.
    generate_and_persist_synonyms() calls Zephyr 7B once per skill (offline job),
    persists results to Firestore, so subsequent startups are free dict lookups.

  Tier 2 — Embedding fallback (handles unknown/novel aliases at runtime)
    If Tier 1 misses, embed the raw skill and find the nearest canonical skill
    by cosine similarity. If sim >= EMBEDDING_FALLBACK_THRESHOLD, return that
    canonical. Catches aliases never seen before with no manual work required.

  Tier 3 — Return as-is
    Genuinely new/unknown skills pass through unchanged — correct behaviour
    since they may be valid skills used directly in the job dataset.

Performance notes:
  - Tier 1 is a pure dict lookup — microseconds.
  - Tier 2 uses diskcache so repeated inputs skip model inference.
  - Canonical embeddings computed once and held in memory per process.
  - EmbeddingEngine is lazy-loaded on first Tier 2 miss to avoid
    triggering model download before the app is ready.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ai_modules.embedding_engine import EmbeddingEngine

EMBEDDING_FALLBACK_THRESHOLD: float = 0.82

SKILL_SYNONYMS: dict[str, list[str]] = {
    "JavaScript": ["js", "javascript", "java script", "ecmascript", "es6", "es2015",
                   "es6+", "vanilla js", "es2016", "es2017", "es2018", "es2019",
                   "es2020", "es next", "esnext"],
    "TypeScript": ["ts", "typescript", "ts/js", "js/ts", "typed javascript"],
    "Python": ["python3", "python 3", "py", "python programming", "python scripting",
               "python language"],
    "Machine Learning": ["ml", "machine learning", "ml/ai", "ai/ml", "ml engineering",
                         "predictive modeling", "predictive modelling"],
    "Deep Learning": ["dl", "deep learning", "neural networks", "neural network",
                      "ann", "artificial neural network"],
    "Natural Language Processing": ["nlp", "natural language processing", "text processing",
                                    "text analytics", "computational linguistics"],
    "Computer Vision": ["cv", "computer vision", "image processing", "image recognition",
                        "image classification"],
    "Artificial Intelligence": ["ai", "artificial intelligence", "generative ai",
                                "gen ai", "genai"],
    "C++": ["cpp", "c plus plus", "c/c++", "c and c++", "cplusplus"],
    "C#": ["csharp", "c sharp", "dotnet c#", "c# .net", "c# dotnet"],
    ".NET": ["dotnet", ".net", "dot net", "asp.net core", "dotnet core",
             ".net core", "aspnet"],
    "Node.js": ["nodejs", "node js", "node", "node.js runtime", "node backend"],
    "React": ["reactjs", "react.js", "react js", "react framework", "react library",
              "react/redux", "react hooks", "react ui"],
    "Vue.js": ["vuejs", "vue js", "vue", "vue 3", "vue2", "vue3", "vuex"],
    "Angular": ["angularjs", "angular js", "angular 2+", "angular framework",
                "angular cli", "ng"],
    "Kubernetes": ["k8s", "kubernetes", "kube", "k8", "container orchestration"],
    "Docker": ["docker", "dockerfile", "docker compose", "containerization",
               "containers", "docker swarm"],
    "PostgreSQL": ["postgres", "postgresql", "pg", "postgres db", "psql"],
    "MongoDB": ["mongo", "mongodb", "mongo db", "mongoose"],
    "Amazon Web Services": ["aws", "amazon web services", "amazon aws", "aws cloud"],
    "Google Cloud Platform": ["gcp", "google cloud", "google cloud platform", "gcloud"],
    "Microsoft Azure": ["azure", "ms azure", "microsoft azure", "azure cloud",
                        "azure devops"],
    "CI/CD": ["cicd", "ci cd", "ci/cd", "continuous integration",
              "continuous deployment", "continuous delivery",
              "continuous integration/continuous deployment",
              "continuous integration/continuous delivery"],
    "GraphQL": ["graphql", "graph ql", "graphql api"],
    "SQL": ["sql", "structured query language", "sql queries", "sql programming",
            "relational database", "relational databases"],
    "NoSQL": ["nosql", "no-sql", "no sql", "non-relational database"],
    "TensorFlow": ["tensorflow", "tf", "tensorflow 2", "tensorflow2", "tf2"],
    "PyTorch": ["pytorch", "torch", "py torch", "pytorch lightning"],
    "scikit-learn": ["sklearn", "scikit learn", "scikit-learn", "scikitlearn"],
    "HuggingFace": ["huggingface", "hugging face", "hf transformers",
                    "hugging face transformers", "hf"],
    "Agile": ["agile methodology", "agile development", "agile practices",
              "agile/scrum", "agile framework"],
    "Scrum": ["scrum methodology", "scrum framework", "scrum master"],
    "Git": ["git", "git scm", "git version control", "version control",
            "source control", "vcs", "scm"],
    "Linux": ["linux", "ubuntu", "debian", "centos", "unix", "linux/unix",
              "rhel", "red hat"],
    "Data Structures": ["data structures", "dsa", "data structures and algorithms",
                        "data structures & algorithms"],
    "Object-Oriented Programming": ["oop", "object oriented", "object-oriented programming",
                                    "object oriented programming", "oops", "ooad"],
    "Microservices": ["microservice", "microservices architecture", "micro services",
                      "microservices-based", "service oriented"],
    "MERN Stack": ["mern stack", "mern"],
    "Flutter & Firebase Development": ["flutter & firebase development",
                                       "flutter firebase development",
                                       "flutter & firebase", "flutter firebase"],
    "CRUD Application Development": ["crud application development", "crud based systems",
                                     "crud operations", "crud functionality", "crud"],
    "RESTful APIs": ["restful apis", "rest apis", "rest api", "restful api",
                     "rest apis & backend development", "restful web services",
                     "application programming interface", "api development",
                     "rest services", "http apis", "http api", "web api"],
    "HTML": ["html", "html5", "hypertext markup language"],
    "CSS": ["css", "css3", "cascading style sheets"],
    "Tailwind CSS": ["tailwind", "tailwindcss", "tailwind css"],
    "Bootstrap": ["bootstrap", "bootstrap css", "bootstrap framework"],
    "Unit Testing": ["unit testing", "unit tests", "automated testing",
                     "test automation", "software testing"],
    "Authentication": ["oauth", "user authentication", "auth flow", "jwt auth",
                       "firebase authentication", "firebase auth"],
    "Authorization": ["authz", "access control", "rbac",
                      "role based access control"],
    "DevOps": ["devops", "dev ops", "development operations"],
    "Data Analysis": ["data analysis", "data analytics", "analytics",
                      "exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "data viz", "dataviz", "reporting"],
    "API Development": ["api development", "api design", "backend api",
                        "rest api development", "api engineering"],
    "Backend Development": ["backend development", "back-end development",
                            "server-side development", "backend engineering",
                            "server side development"],
    "Frontend Development": ["frontend development", "front-end development",
                             "client-side development", "frontend engineering",
                             "ui development", "client side development"],
    "Full Stack Development": ["full stack development", "full-stack development",
                               "fullstack development", "full stack engineering",
                               "fullstack engineering"],
    "Mobile Development": ["mobile development", "mobile app development",
                           "android development", "ios development",
                           "cross platform development"],
    "Cloud Architecture": ["cloud architecture", "cloud computing",
                           "cloud infrastructure", "cloud engineering",
                           "cloud solutions"],
    "System Design": ["system design", "software architecture", "software design",
                      "architectural design", "solution architecture"],
    "Cybersecurity": ["cybersecurity", "cyber security", "information security",
                      "infosec", "network security", "it security"],
    "Project Management": ["project management", "pm", "technical project management",
                           "program management"],
    "Serverless": ["serverless", "serverless architecture", "faas",
                   "function as a service", "lambda", "cloud functions"],
    "Distributed Systems": ["distributed systems", "distributed computing",
                            "distributed architecture", "distributed applications"],
    "Design Patterns": ["design patterns", "software design patterns",
                        "architectural patterns", "gang of four", "gof patterns"],
    "Flutter": ["flutter", "flutter sdk", "flutter framework",
                "flutter app", "flutter mobile"],
    "Dart": ["dart", "dart language", "dart programming"],
    "Firebase": ["firebase", "firebase realtime database",
                 "firebase auth", "firebase cloud"],
    "Firestore": ["firestore", "cloud firestore", "firebase firestore"],
    "React Native": ["react native", "rn", "react-native", "reactnative"],
    "Swift": ["swift", "swift language", "swift programming", "swiftui"],
    "Kotlin": ["kotlin", "kotlin android", "kotlin language"],
    "Django": ["django", "django framework", "django rest framework",
               "drf", "django web"],
    "Flask": ["flask", "flask framework", "flask api", "flask python"],
    "FastAPI": ["fastapi", "fast api", "fastapi python"],
    "Spring Boot": ["spring boot", "spring", "spring framework", "spring mvc"],
    "Laravel": ["laravel", "laravel php", "laravel framework"],
    "Redis": ["redis", "redis cache", "redis db", "redis server"],
    "Elasticsearch": ["elasticsearch", "elastic search", "elk", "opensearch"],
    "Kafka": ["kafka", "apache kafka", "kafka streaming", "event streaming"],
    "Terraform": ["terraform", "terraform iac", "infrastructure as code",
                  "iac", "hashicorp terraform"],
    "Jenkins": ["jenkins", "jenkins ci", "jenkins pipeline"],
    "GitHub Actions": ["github actions", "gh actions", "github ci", "github workflow"],
    "Nginx": ["nginx", "nginx server", "nginx proxy"],
    "Pandas": ["pandas", "pandas python", "pd", "data manipulation"],
    "NumPy": ["numpy", "np", "numerical python", "numerical computing"],
    "Matplotlib": ["matplotlib", "matplotlib python", "plt"],
    "Webpack": ["webpack", "webpack bundler", "module bundler"],
    "Jest": ["jest", "jest testing", "jest js"],
    "Pytest": ["pytest", "pytest python", "py.test"],
    "Selenium": ["selenium", "selenium webdriver", "browser automation"],
    "Microservices Architecture": ["microservices architecture",
                                   "microservice architecture", "soa",
                                   "service oriented architecture"],
}

# Build reverse map: alias.lower() → canonical
_REVERSE_MAP: dict[str, str] = {}
for _canonical, _aliases in SKILL_SYNONYMS.items():
    for _alias in _aliases:
        _REVERSE_MAP[_alias.lower()] = _canonical
    _REVERSE_MAP[_canonical.lower()] = _canonical


class SkillNormalizer:
    """
    Two-tier skill normalizer.
      Tier 1: Static + LLM-expanded reverse map (exact, zero-cost).
      Tier 2: Embedding cosine similarity against canonical list (fallback).
    """

    def __init__(self):
        self._engine: "EmbeddingEngine | None" = None
        self._canonical_list: list[str] = list(SKILL_SYNONYMS.keys())
        self._canonical_embs: np.ndarray | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def normalize(self, raw_skill: str) -> str | None:
        cleaned = self._clean(raw_skill)
        if not cleaned or len(cleaned) < 2:
            return None

        lower = cleaned.lower()

        # Tier 1: exact reverse-map lookup (microseconds)
        if lower in _REVERSE_MAP:
            return _REVERSE_MAP[lower]

        # Tier 2: embedding cosine similarity
        try:
            canonical = self._embedding_normalize(cleaned)
            if canonical:
                _REVERSE_MAP[lower] = canonical  # cache for rest of this run
                logger.debug(f"Embedding normalized: '{cleaned}' → '{canonical}'")
                return canonical
        except Exception as e:
            logger.warning(f"Embedding fallback failed for '{cleaned}': {e}")

        # Tier 3: genuinely unknown — return as-is
        return cleaned.strip()

    def normalize_job_title(self, title: str) -> str:
        title = re.sub(r"\s*\(.*?\)", "", title).strip()
        return title.strip().title()

    def deduplicate_skills(self, skills: list[str]) -> list[str]:
        """
        1. Normalize synonyms
        2. Remove case duplicates
        3. Remove substring duplicates (keep the longer, more descriptive form)
        """
        normalized: list[str] = []
        seen_lower: set[str] = set()
        for s in skills:
            norm = self.normalize(s)
            if norm and norm.lower() not in seen_lower:
                seen_lower.add(norm.lower())
                normalized.append(norm)

        normalized.sort(key=len, reverse=True)
        final: list[str] = []
        for skill in normalized:
            skill_lower = skill.lower()
            if not any(skill_lower in kept.lower() for kept in final):
                final.append(skill)

        return sorted(final)

    # ── LLM + Firestore synonym expansion (Option B) ──────────────────────────

    @staticmethod
    def load_dynamic_synonyms() -> int:
        """
        Load LLM-generated synonyms from Firestore and merge into _REVERSE_MAP.
        Call once at startup after Firebase is initialized.
        Returns number of new alias entries added.
        """
        try:
            from services.firebase_service import get_db
            db = get_db()
            docs = db.collection("skill_synonyms").stream()
            added = 0
            for doc in docs:
                data = doc.to_dict()
                canonical = data.get("canonical", doc.id)
                aliases: list[str] = data.get("aliases", [])
                for alias in aliases:
                    key = alias.lower().strip()
                    if key and key not in _REVERSE_MAP:
                        _REVERSE_MAP[key] = canonical
                        added += 1
            logger.info(f"Dynamic synonyms loaded from Firestore: {added} new aliases.")
            return added
        except Exception as e:
            logger.warning(f"Could not load dynamic synonyms from Firestore: {e}")
            return 0

    @staticmethod
    def generate_and_persist_synonyms(canonical_skills: list[str] | None = None) -> int:
        """
        One-time offline job: call Zephyr 7B to generate aliases for each
        canonical skill, persist to Firestore (collection: skill_synonyms),
        and add to the live _REVERSE_MAP.

        Only processes skills that don't already have a Firestore entry.
        Safe to re-run — skips existing entries.
        Returns number of skills successfully processed.
        """
        import json
        import re as _re
        from services.firebase_service import get_db
        from ai_modules.llm_resources import _call_llm

        if canonical_skills is None:
            canonical_skills = list(SKILL_SYNONYMS.keys())

        db = get_db()
        coll = db.collection("skill_synonyms")

        existing = {doc.id for doc in coll.stream()}
        to_generate = [s for s in canonical_skills if s not in existing]

        if not to_generate:
            logger.info("All canonical skills already have Firestore synonym entries.")
            return 0

        logger.info(f"Generating synonyms for {len(to_generate)} skills via LLM...")
        processed = 0

        system = (
            "You are a technical recruiter and software engineer. "
            "You always respond with valid JSON only — no markdown, no explanation."
        )

        for skill in to_generate:
            user = (
                f'List all common aliases, abbreviations, alternate spellings, '
                f'and package/library names for the technical skill "{skill}" '
                f'as they appear in resumes and job postings.\n'
                f'Return ONLY a JSON array of lowercase strings. Max 12 items. '
                f'Do not include the canonical skill name itself. No markdown.'
            )
            try:
                raw = _call_llm(system, user, max_tokens=200)
                match = _re.search(r"\[.*?\]", raw, _re.DOTALL)
                if not match:
                    logger.warning(f"  No JSON array in LLM response for '{skill}'")
                    continue

                aliases: list[str] = json.loads(match.group())
                aliases = [
                    a.lower().strip()
                    for a in aliases
                    if isinstance(a, str) and len(a.strip()) > 1
                ]

                coll.document(skill).set({
                    "canonical": skill,
                    "aliases": aliases,
                    "source": "llm_zephyr",
                })

                for alias in aliases:
                    if alias not in _REVERSE_MAP:
                        _REVERSE_MAP[alias] = skill

                logger.info(f"  '{skill}' → {len(aliases)} aliases persisted.")
                processed += 1

            except Exception as e:
                logger.warning(f"  Failed to generate synonyms for '{skill}': {e}")
                continue

        logger.info(
            f"Synonym generation complete: {processed}/{len(to_generate)} skills processed."
        )
        return processed

    # ── Embedding fallback (Option A) ─────────────────────────────────────────

    def _get_engine(self) -> "EmbeddingEngine":
        if self._engine is None:
            from ai_modules.embedding_engine import EmbeddingEngine
            self._engine = EmbeddingEngine()
        return self._engine

    def _get_canonical_embs(self) -> np.ndarray:
        if self._canonical_embs is None:
            logger.debug(
                f"Computing canonical embedding matrix ({len(self._canonical_list)} skills)..."
            )
            self._canonical_embs = self._get_engine().embed_batch(self._canonical_list)
            logger.debug("Canonical embedding matrix ready.")
        return self._canonical_embs

    def _embedding_normalize(self, raw: str) -> str | None:
        emb = self._get_engine().embed(raw)          # (dim,) L2-normalized
        canon_embs = self._get_canonical_embs()      # (n, dim)
        sims = canon_embs @ emb                      # (n,) cosine similarities
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= EMBEDDING_FALLBACK_THRESHOLD:
            return self._canonical_list[best_idx]
        return None

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[^\w\s.#+/&]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
