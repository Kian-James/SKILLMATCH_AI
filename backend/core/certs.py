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
# Cert-specific pre-processing: keep "Provider - Cert Name" as one unit
# ---------------------------------------------------------------------------

# A provider token is a short (1–4 word) title-cased or ALL-CAPS label that
# is followed by " - " (inline) or appears alone on a line directly before
# the cert name line.  We do NOT want to split or strip it.
#
# Step 1 – inline dash form:  "Udemy - Flutter Certified Programmer"
#   → already one line, but _strip_subheader might later mistake the part
#     before " - " as a sub-header label.  We replace " - " with a protected
#     separator so the colon-based strip logic is never triggered.
#
# Step 2 – split-line form:
#   Udemy
#   Flutter Certified Programmer
#   → _merge_split_lines won't join them when "Udemy" scores low alone AND
#     "Udemy Flutter Certified Programmer" also scores low (provider not in
#     allowlist).  We detect this pattern and pre-join them with " - ".

# Known cert/credential issuers — used to identify provider lines.  This list
# covers the most common platforms; the regex below is the authoritative check.
_KNOWN_PROVIDERS = {
    "udemy", "coursera", "freecodecamp", "fcc", "linkedin", "linkedin learning",
    "cisco", "google", "microsoft", "amazon", "aws", "meta", "ibm", "oracle",
    "comptia", "ec-council", "isc2", "isaca", "pmi", "axelos", "peoplecert",
    "scrum alliance", "scrumorg", "scrum.org", "scaled agile", "safe",
    "edx", "pluralsight", "alison", "skillshare", "datacamp", "kaggle",
    "hubspot", "salesforce", "adobe", "red hat", "vmware", "juniper",
    "certiport", "testout", "lpi", "lpic", "python institute",
}

# A line is a provider when it is a short (1–4 word) phrase AND either:
#   a) it is an exact known-provider match (case-insensitive), OR
#   b) it is purely title-cased / ALL-CAPS with no digits and no cert-like
#      keywords that would indicate it is itself a cert name.
_CERT_KEYWORD_RE = re.compile(
    r"\b(certified|certification|certificate|associate|professional|"
    r"engineer|developer|architect|foundation|practitioner|specialist|"
    r"expert|master|analyst|administrator|technician|designer|"
    r"responsive|relational|database|design|web|programming|development|"
    r"security|network|cloud|data|machine\s+learning|artificial\s+intelligence)\b",
    re.I,
)

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
    If the line matches "Provider - Cert Name", replace the separating " - "
    with "  —  " (em-dash + spaces).  This prevents any downstream logic from
    treating the dash as a list delimiter or the provider word as a sub-header.
    The em-dash form is visually identical when the cert name is ultimately
    returned, and SBERT/fuzzy matching works on the cert portion regardless.
    """
    m = _INLINE_PROVIDER_DASH_RE.match(line)
    if m:
        return f"{m.group(1)}  —  {m.group(2)}"
    return line


def _is_provider_line(line: str) -> bool:
    """
    Return True when line looks like a cert issuer / platform name rather than
    a cert title.

    Rules (all must pass):
      1. 1–4 words, no digits embedded in words (version numbers disqualify).
      2. Matches the known-provider list  OR  contains no cert-title keywords.
      3. Structurally title-cased / ALL-CAPS (no lowercase-leading words).
    """
    words = line.split()
    if not (1 <= len(words) <= 4):
        return False
    # Known provider → always treat as provider
    if line.lower() in _KNOWN_PROVIDERS:
        return True
    # Contains cert-title keywords → it IS a cert, not a provider
    if _CERT_KEYWORD_RE.search(line):
        return False
    # Must be title-cased or ALL-CAPS (no all-lowercase words)
    if not _PROVIDER_LINE_RE.match(line):
        return False
    return True


def _normalize_cert_lines(section_text: str) -> str:
    """
    Pre-process the raw certifications section text so that cert entries
    referencing an issuing provider are preserved as a single line.

    Two patterns are handled:

    1. Inline dash  — "Udemy - Flutter Certified Programmer"
       Already on one line. Replace the separating " - " with "  —  " so that
       downstream splitters and _strip_subheader never break the entry apart.

    2. Split lines (including \\n\\n-separated)  — provider on its own line /
       paragraph, cert name on the next non-empty line / paragraph.  We
       identify provider lines with _is_provider_line(), find the very next
       non-empty line (skipping any blank lines), and join them as
       "Provider - Cert Name", then apply pattern-1 normalisation.

       Example (\\n\\n layout from the PDF extractor):
           FreeCodeCamp\\n\\nCertified Information Technology (ITF+)\\n\\nCisco...
       After normalisation:
           FreeCodeCamp  —  Certified Information Technology (ITF+)
           Cisco  —  Responsive Web Design
           ...
    """
    lines = section_text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # ── Pattern 2: provider line followed by cert line ───────────────────
        if _is_provider_line(line):
            # Find the actual index of the next non-empty line (skips blanks).
            next_idx = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    next_idx = j
                    break

            if next_idx is not None:
                nxt = lines[next_idx].strip()
                # Only pair when the next line is NOT itself a bare provider
                # (guards against two consecutive provider-only lines).
                if not _is_provider_line(nxt) and len(nxt.split()) >= 1:
                    combined = f"{line} - {nxt}"
                    i = next_idx + 1          # advance past the consumed cert line
                    out.append(_protect_inline_dash(combined))
                    continue

        # ── Pattern 1: inline dash ───────────────────────────────────────────
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