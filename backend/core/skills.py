"""
core/skills.py
==============
Stage 2 — Skill extraction.

Thin wrapper around extractor_base.discover_items_dynamically().
Only skills-specific config lives here: the heading regex, section boundary,
and corpus key.
"""

from __future__ import annotations

import re

from core.extractor_base import SectionConfig, discover_items_dynamically, extract_section

_SKILL_HEADING_RE = re.compile(
    r"^[^\w]*"
    r"(?:"
    r"(?:key\s+|core\s+|technical\s+|professional\s+)?skills?"
    r"(?:\s*(?:&|and)\s*(?:tools?|technologies?))?"
    r"|(?:technical\s+|core\s+|key\s+)?competenc(?:ies|y)"
    r"|(?:areas?\s+of\s+)?technical\s+expertise"
    r"|tools?\s*(?:&|and)\s*technologies?"
    r"|(?:technical\s+)?proficienc(?:ies|y)"
    r")"
    r"\s*:?\s*$",
    re.I | re.M,
    )

_SKILLS_END_RE = re.compile(
    r"^\s*[^\w]*(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?|"
    r"work\s+experience|experience|education|employment|projects?|awards?|"
    r"publications?|references?|volunteer|languages?|interests?|activities|"
    r"summary|objective(?![\w\-]))",
    re.I | re.M,
    )

SKILL_CONFIG = SectionConfig(
    heading_re=_SKILL_HEADING_RE,
    end_re=_SKILLS_END_RE,
    corpus_key="required_skills",
    label="Skills",
)


def extract_skills_section(text: str) -> str:
    """Return raw text of the skills section."""
    return extract_section(text, SKILL_CONFIG)


def discover_skills_dynamically(
        resume_text:  str,
        jobs_corpus:  list[dict],
        extra_skills: list[str] | None = None,
        is_native_pdf: bool = False,
) -> list[str]:
    """Stage 2 — return deduplicated, canonicalised skill names."""
    section_text = extract_skills_section(resume_text)
    return discover_items_dynamically(
        resume_text=resume_text,
        jobs_corpus=jobs_corpus,
        config=SKILL_CONFIG,
        extra_items=extra_skills,
        is_native_pdf=is_native_pdf,
        section_text=section_text,
    )