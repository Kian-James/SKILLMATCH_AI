# backend/app/utils/validators.py
from typing import List

def deduplicate(items: List[str]) -> List[str]:
    """
    Remove duplicates while preserving order
    """
    seen = set()
    result = []
    for item in items:
        normalized = item.strip().lower()
        if normalized not in seen:
            result.append(item)
            seen.add(normalized)
    return result

def sanitize_skills(skills: List[str]) -> List[str]:
    """
    Basic validation for AI-generated skill lists
    - Removes empty strings
    - Deduplicates
    - Strips whitespace
    """
    clean_skills = [s.strip() for s in skills if s.strip()]
    return deduplicate(clean_skills)

def sanitize_learning_paths(paths: List[dict]) -> List[dict]:
    """
    Validate learning paths output from AI
    - Remove duplicates by 'title'
    """
    seen = set()
    result = []
    for p in paths:
        title = p.get("title", "").strip()
        if title and title.lower() not in seen:
            result.append(p)
            seen.add(title.lower())
    return result
