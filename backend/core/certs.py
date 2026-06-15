"""
core/certs.py
=============
Stage 2B — Certification extraction.

Thin wrapper around extractor_base.discover_items_dynamically().
Only certs-specific config lives here: the heading regex, section boundary,
and corpus key.
"""

from __future__ import annotations

import re

from core.extractor_base import SectionConfig, discover_items_dynamically, extract_section

_CERT_HEADING_RE = re.compile(
    r"^(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?)",
    re.I | re.M,
    )

_CERT_END_RE = re.compile(
    r"^\s*[^\w]*(work\s+experience|experience|education|employment|projects?|"
    r"awards?|publications?|references?|volunteer|languages?|interests?|"
    r"activities|summary|objective(?![\w\-]))",
    re.I | re.M,
    )

CERT_CONFIG = SectionConfig(
    heading_re=_CERT_HEADING_RE,
    end_re=_CERT_END_RE,
    corpus_key="required_certifications",
    label="Certs",
)


def extract_cert_section(text: str) -> str:
    """Return raw text of the certifications section."""
    return extract_section(text, CERT_CONFIG)


def extract_certificates(
        resume_text:  str,
        jobs_corpus:  list[dict],
        extra_certs:  list[str] | None = None,
        is_native_pdf: bool = False,
) -> list[str]:
    """Stage 2B — return deduplicated, canonicalised certification names."""
    return discover_items_dynamically(
        resume_text=resume_text,
        jobs_corpus=jobs_corpus,
        config=CERT_CONFIG,
        extra_items=extra_certs,
        is_native_pdf=is_native_pdf,
    )