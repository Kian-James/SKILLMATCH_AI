"""
ai_modules/skill_implications.py

Option D: LLM-generated skill implication map.

At FAISS rebuild time, Zephyr 7B is asked once per unique job skill:
  "What skills/technologies demonstrate competency in {skill}?"

Results stored in Firestore (collection: skill_implications) and loaded into
_IMPLICATION_MAP at warmup. Gate 4 in resume_service is then an O(1) dict lookup:
  if any(u.lower() in _IMPLICATION_MAP.get(req_skill.lower(), set())
         for u in user_skills_lower) → matched_by = "llm_implied"

This captures React→HTML/CSS (sim 0.61), Java→OOP (sim 0.58), Flutter→Dart,
MongoDB→NoSQL — all cases RoBERTa misses because cosine < threshold.
"""

from __future__ import annotations

import json
import re

from loguru import logger

# Runtime implication map: req_skill.lower() → frozenset of implying skills (lowercase)
# Populated from Firestore at startup + seed map below.
_IMPLICATION_MAP: dict[str, frozenset] = {}

# Seed map — always available even before LLM runs.
# Covers highest-impact implication gaps RoBERTa consistently misses.
_SEED_IMPLICATIONS: dict[str, list[str]] = {
    "javascript": [
        "react", "reactjs", "node.js", "nodejs", "typescript", "vue.js", "vuejs",
        "angular", "express.js", "next.js", "jquery", "svelte", "nuxt.js",
        "react native", "mern stack", "mean stack", "javascript es6+", "es6",
        "ember.js", "backbone.js",
    ],
    "html/css": [
        "react", "vue.js", "angular", "svelte", "next.js", "html5", "css3",
        "bootstrap", "tailwind css", "sass", "scss", "frontend development",
        "web development", "html", "css",
    ],
    "html5": ["react", "vue.js", "angular", "html/css", "frontend development", "html"],
    "css3": ["react", "vue.js", "angular", "html/css", "tailwind css", "sass", "scss", "css"],
    "frontend development": [
        "react", "vue.js", "angular", "next.js", "svelte", "javascript",
        "typescript", "html/css", "html5", "css3",
    ],
    "front end development": [
        "react", "vue.js", "angular", "next.js", "svelte", "javascript",
        "typescript", "html/css", "html5", "css3",
    ],
    "backend development": [
        "node.js", "express.js", "django", "flask", "fastapi", "spring boot",
        "ruby on rails", "laravel", "asp.net", "restful apis",
        "rest apis & backend development",
    ],
    "back end development": [
        "node.js", "express.js", "django", "flask", "fastapi", "spring boot",
        "ruby on rails", "laravel", "asp.net", "restful apis",
    ],
    "web development": [
        "react", "node.js", "express.js", "javascript", "html/css",
        "mern stack", "full stack development", "html5", "css3",
    ],
    "full stack development": [
        "mern stack", "mean stack", "react", "node.js", "express.js",
        "mongodb", "angular", "vue.js",
    ],
    "database management": [
        "mongodb", "mysql", "postgresql", "redis", "firestore", "sqlite",
        "sql", "nosql", "dynamodb",
    ],
    "nosql": [
        "mongodb", "firestore", "dynamodb", "cassandra", "redis",
        "couchdb", "firebase",
    ],
    "sql": [
        "mysql", "postgresql", "sqlite", "sql server", "mariadb",
        "database management", "mysql certified developer (mcd)",
    ],
    "relational databases": ["mysql", "postgresql", "sqlite", "sql server", "mariadb", "sql"],
    "cloud computing": [
        "aws", "gcp", "google cloud platform", "azure", "docker",
        "kubernetes", "terraform", "heroku", "firebase",
    ],
    "devops": [
        "docker", "kubernetes", "ci/cd", "jenkins", "github actions",
        "gitlab ci", "terraform", "ansible", "linux",
    ],
    "mobile development": [
        "flutter", "react native", "swift", "kotlin", "android", "ios",
        "dart", "swiftui", "jetpack compose", "flutter & firebase development",
        "xamarin", "ionic",
    ],
    "android development": ["kotlin", "java", "android sdk", "android studio", "jetpack compose"],
    "ios development": ["swift", "swiftui", "objective c", "xcode", "cocoa touch", "uikit"],
    "object-oriented programming": [
        "java", "python", "c#", "c++", "kotlin", "typescript", "ruby",
        "php", "scala", "swift",
    ],
    "oop": [
        "java", "python", "c#", "c++", "kotlin", "typescript",
        "ruby", "php", "scala", "swift",
    ],
    "version control systems": ["git", "github", "gitlab", "bitbucket", "svn"],
    "version control": ["git", "github", "gitlab", "bitbucket"],
    "agile methodologies": ["scrum", "kanban", "agile", "jira", "confluence"],
    "agile": ["scrum", "kanban", "jira", "agile methodologies"],
    "testing frameworks": [
        "junit", "jest", "pytest", "mocha", "jasmine", "cypress",
        "selenium", "testng", "rspec",
    ],
    "unit testing": ["junit", "jest", "pytest", "mocha", "rspec", "testng", "nunit"],
    "api design": [
        "restful apis", "graphql", "grpc", "rest api", "openapi",
        "swagger", "rest apis & backend development",
    ],
    "restful apis": [
        "rest api", "api design", "rest apis & backend development",
        "graphql", "express.js", "fastapi", "django", "flask",
    ],
    "microservices": ["docker", "kubernetes", "spring boot", "node.js", "grpc", "kafka"],
    "data structures": ["algorithms", "data structures and algorithms", "leetcode"],
    "algorithms": ["data structures", "data structures and algorithms"],
    "machine learning": [
        "tensorflow", "pytorch", "scikit-learn", "keras", "pandas",
        "numpy", "deep learning", "nlp",
    ],
    "deep learning": ["tensorflow", "pytorch", "keras", "neural networks"],
    "dart": ["flutter", "flutter & firebase development", "flutter & firebase"],
    "flutter": ["dart", "flutter & firebase", "flutter & firebase development"],
    "kotlin": ["android", "android sdk", "android studio", "java"],
    "swift": ["ios", "swiftui", "xcode", "objective c", "ios sdk"],
    "typescript": ["javascript", "angular", "react", "node.js", "vue.js"],
    "spring boot": ["java", "spring framework", "hibernate", "maven", "gradle"],
    "django": ["python", "rest framework", "postgresql", "sqlite"],
    "react native": ["javascript", "react", "typescript", "expo", "mobile development"],
    "git version control": ["git", "github", "gitlab", "bitbucket", "version control"],
    "ci/cd": ["jenkins", "github actions", "gitlab ci", "circleci", "devops", "docker"],
    "containerization": ["docker", "kubernetes", "docker compose", "podman"],
    "security best practices": [
        "owasp", "penetration testing", "ssl/tls", "encryption",
        "authentication", "oauth", "jwt",
    ],
    "responsive web design": ["html/css", "css3", "bootstrap", "tailwind css", "css media queries"],
    "state management": ["redux", "mobx", "vuex", "zustand", "context api"],
    "cross browser compatibility": ["html/css", "css3", "javascript", "responsive web design"],
    "crud application development": [
        "mongodb", "mysql", "postgresql", "express.js", "django",
        "rest api", "node.js",
    ],
    "flutter & firebase development": ["flutter", "dart", "firebase", "firestore"],
    "flutter & firebase": ["flutter", "dart", "firebase", "firestore"],
    "mern stack": ["mongodb", "express.js", "react", "node.js"],
    "mean stack": ["mongodb", "express.js", "angular", "node.js"],
}


def load_implications_from_firestore() -> int:
    """
    Load LLM-generated implications from Firestore into _IMPLICATION_MAP.
    Merges with seed map. Call once at warmup after Firebase is ready.
    Returns number of skills with implication entries loaded.
    """
    global _IMPLICATION_MAP

    # Start with seed
    merged: dict[str, set] = {
        k.lower(): set(v) for k, v in _SEED_IMPLICATIONS.items()
    }

    try:
        from services.firebase_service import get_db
        db   = get_db()
        docs = db.collection("skill_implications").stream()
        loaded = 0
        for doc in docs:
            data    = doc.to_dict()
            skill   = data.get("skill", doc.id).lower()
            implied = [s.lower() for s in data.get("implied_by", [])]
            if skill and implied:
                merged.setdefault(skill, set()).update(implied)
                loaded += 1
        logger.info(
            f"Skill implications loaded: {loaded} from Firestore + "
            f"{len(_SEED_IMPLICATIONS)} seed entries = "
            f"{len(merged)} total skills covered."
        )
    except Exception as e:
        logger.warning(f"Could not load skill implications from Firestore: {e}")

    _IMPLICATION_MAP = {k: frozenset(v) for k, v in merged.items()}
    return len(_IMPLICATION_MAP)


def check_implied(req_skill: str, user_skills_lower: set[str]) -> bool:
    """
    Return True if any user skill implies competency in req_skill.
    O(1) dict lookup — no LLM call at request time.
    """
    implied_by = _IMPLICATION_MAP.get(req_skill.lower())
    if not implied_by:
        return False
    return bool(user_skills_lower & implied_by)


def build_implications_with_llm(
        skills: list[str] | None = None,
        batch_size: int = 20,
) -> int:
    """
    One-time offline job: call Zephyr 7B to generate implication lists for each
    unique job skill and persist to Firestore (collection: skill_implications).

    For each skill, the prompt asks: "What technologies/skills demonstrate
    competency in {skill}? A candidate who knows X can be considered proficient
    in {skill} even if they haven't listed it explicitly."

    Safe to re-run — skips skills already in Firestore.
    Returns number of skills processed.

    Call via POST /admin/build-implications after deploy.
    Runs in ~25-35 minutes for 500 skills on HuggingFace free tier.
    """
    import time
    from services.firebase_service import get_db, fetch_all_jobs
    from ai_modules.llm_resources import _call_llm

    db   = get_db()
    coll = db.collection("skill_implications")

    # Collect all unique skills from the job corpus
    if skills is None:
        jobs = fetch_all_jobs()
        all_skills: set[str] = set()
        for job in jobs:
            for s in job.get("skills", []):
                s = s.strip()
                if s and len(s) >= 3:
                    all_skills.add(s)
        skills = sorted(all_skills)

    # Skip already processed
    existing = {doc.id for doc in coll.stream()}
    to_process = [s for s in skills if s not in existing]

    if not to_process:
        logger.info("All skills already have implication entries in Firestore.")
        return 0

    logger.info(f"Building implications for {len(to_process)} skills via LLM...")

    system = (
        "You are a senior technical recruiter with deep knowledge of software engineering. "
        "You always respond with valid JSON only — no markdown, no explanation."
    )

    processed = 0
    errors    = 0

    for i, skill in enumerate(to_process):
        user = (
            f'For the technical skill/requirement "{skill}", list all skills, '
            f'technologies, frameworks, or tools that IMPLY or DEMONSTRATE '
            f'competency in it. For example, knowing React implies JavaScript. '
            f'Knowing Flutter implies Dart. Knowing Spring Boot implies Java.\n\n'
            f'Return ONLY a JSON object with one key "implied_by" whose value is '
            f'an array of lowercase skill/technology strings (max 15). '
            f'Include framework→language, tool→concept, and specialization→general '
            f'relationships. No markdown. No explanation.'
        )

        try:
            raw   = _call_llm(system, user, max_tokens=300)
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if not match:
                logger.warning(f"  No JSON for '{skill}'")
                errors += 1
                continue

            data    = json.loads(match.group())
            implied = data.get("implied_by", [])
            implied = [
                s.lower().strip() for s in implied
                if isinstance(s, str) and len(s.strip()) >= 2
            ]

            coll.document(skill).set({
                "skill":      skill,
                "implied_by": implied,
                "source":     "llm_zephyr",
            })

            # Update live map immediately
            key = skill.lower()
            current = set(_IMPLICATION_MAP.get(key, frozenset()))
            current.update(implied)
            _IMPLICATION_MAP[key] = frozenset(current)

            processed += 1
            if processed % 10 == 0:
                logger.info(f"  Progress: {processed}/{len(to_process)} skills done.")

            # Rate limit: HuggingFace free tier allows ~20 req/min
            if (i + 1) % batch_size == 0:
                logger.info(f"  Batch of {batch_size} done. Brief pause...")
                time.sleep(3)

        except Exception as e:
            logger.warning(f"  Failed for '{skill}': {e}")
            errors += 1
            continue

    logger.info(
        f"Implication build complete: {processed} succeeded, "
        f"{errors} failed, {len(to_process) - processed - errors} skipped."
    )
    return processed