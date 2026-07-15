"""
core/certs.py
=============
Stage 2 — Certification extraction (alongside skills.py's skill extraction).

Thin wrapper around extractor_base.discover_items_dynamically().
Only certs-specific config lives here: the heading regex, section boundary,
corpus key, and a pre-processing hook that keeps "Provider - Cert Name"
(and the split-line equivalent) intact as a single entry.
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

# ---------------------------------------------------------------------------
# Cert-specific pre-processing: keep "Provider - Cert Name" as one unit.
#
# Two forms show up in resumes:
#   1. Inline dash: "Udemy - Flutter Certified Programmer" — already one
#      line, but the " - " could later be mistaken for a sub-header split.
#   2. Split lines: "Udemy" / "Flutter Certified Programmer" on separate
#      lines — these don't get merged by _merge_split_lines since neither
#      line alone matches the allowlist well, so we join them here.
# ---------------------------------------------------------------------------

# Common cert/credential issuers, used to recognize provider lines.
_KNOWN_PROVIDERS = {
    "udemy", "coursera", "freecodecamp", "fcc", "linkedin", "linkedin learning",
    "cisco", "google", "microsoft", "amazon", "aws", "meta", "ibm", "oracle",
    "comptia", "ec-council", "isc2", "isaca", "pmi", "axelos", "peoplecert",
    "scrum alliance", "scrumorg", "scrum.org", "scaled agile", "safe",
    "edx", "pluralsight", "alison", "skillshare", "datacamp", "kaggle",
    "hubspot", "salesforce", "adobe", "red hat", "vmware", "juniper",
    "certiport", "testout", "lpi", "lpic", "python institute",
}

# Keywords that indicate a line is a cert title rather than a provider name.
_CERT_KEYWORD_RE = re.compile(
    r"\b(certified|certification|certificate|associate|professional|"
    r"engineer|developer|architect|foundation|practitioner|specialist|"
    r"expert|master|analyst|administrator|technician|designer|"
    r"responsive|relational|database|design|web|programming|development|"
    r"security|network|cloud|data|machine\s+learning|artificial\s+intelligence)\b",
    re.I,
)

# A provider line is title-cased / ALL-CAPS with no lowercase-leading words.
_PROVIDER_LINE_RE = re.compile(
    r"^(?:[A-ZÁÉÍÓÚÑÜ][A-Za-záéíóúñü0-9]*(?:[\s\-][A-ZÁÉÍÓÚÑÜ][A-Za-záéíóúñü0-9]*)*)$"
)

# Inline "Provider - Cert Name" — captures provider (group 1) and cert (group 2).
_INLINE_PROVIDER_DASH_RE = re.compile(
    r"^([A-ZÁÉÍÓÚÑÜ][A-Za-záéíóúñü0-9]*(?:[\s\-][A-ZÁÉÍÓÚÑÜ][A-Za-záéíóúñü0-9]*){0,3})"
    r"\s+-\s+"
    r"(.{5,})$"
)


def _protect_inline_dash(line: str) -> str:
    """
    Replace " - " with "  —  " in "Provider - Cert Name" lines so downstream
    logic never treats the dash as a list delimiter or splits off the provider
    as a sub-header. Visually identical; matching still works on the cert part.
    """
    m = _INLINE_PROVIDER_DASH_RE.match(line)
    if m:
        return f"{m.group(1)}  —  {m.group(2)}"
    return line


def _is_provider_line(line: str) -> bool:
    """
    True if `line` looks like a cert issuer/platform name rather than a cert
    title: 1–4 words, no cert-title keywords (unless it's a known provider),
    and title-cased or ALL-CAPS.
    """
    words = line.split()
    if not (1 <= len(words) <= 4):
        return False
    if line.lower() in _KNOWN_PROVIDERS:
        return True
    if _CERT_KEYWORD_RE.search(line):
        return False
    if not _PROVIDER_LINE_RE.match(line):
        return False
    return True


def _normalize_cert_lines(section_text: str) -> str:
    """
    Pre-process the raw certifications section so provider + cert name stay
    paired as a single entry, whether they're already inline or split across
    lines (e.g. "FreeCodeCamp" then "Certified Information Technology (ITF+)"
    on the next non-empty line).
    """
    lines = section_text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Provider line followed by a cert line → join them.
        if _is_provider_line(line):
            next_idx = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    next_idx = j
                    break

            if next_idx is not None:
                nxt = lines[next_idx].strip()
                # Don't pair two consecutive provider-only lines.
                if not _is_provider_line(nxt) and len(nxt.split()) >= 1:
                    combined = f"{line} - {nxt}"
                    i = next_idx + 1
                    out.append(_protect_inline_dash(combined))
                    continue

        out.append(_protect_inline_dash(line))
        i += 1

    return "\n".join(out)


# ---------------------------------------------------------------------------

CERT_CONFIG = SectionConfig(
    heading_re=_CERT_HEADING_RE,
    end_re=_CERT_END_RE,
    corpus_key="required_certifications",
    label="Certs",
    preprocess_section_text=_normalize_cert_lines,
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
    """Stage 2 — return deduplicated, canonicalised certification names."""
    section_text = extract_cert_section(resume_text)
    return discover_items_dynamically(
        resume_text=resume_text,
        jobs_corpus=jobs_corpus,
        config=CERT_CONFIG,
        extra_items=extra_certs,
        is_native_pdf=is_native_pdf,
        section_text=section_text,
    )