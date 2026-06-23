"""
core/extractor_base.py
======================
Shared extraction pipeline for Stage 2 — both skills and certs discovery.
Both call into discover_items_dynamically(), which semantically matches
discovered text against each job's allowlist (SBERT cosine, threshold 0.80)
before anything is accepted as a discovered skill or cert.

Both extractors are identical in behaviour — they differ only in which section
heading they search for and which Firestore allowlist key they read from.
Everything else (section slicing, line filtering, OCR correction, compound
expansion, SBERT matching, fuzzy fallback, dedup) lives here and is shared.

Public API
----------
  SectionConfig              — dataclass injected by each caller
  discover_items_dynamically — full pipeline; called by skills.py and certs.py
  extract_section             — regex section slicer (exposed for unit tests)

Changes (v26 — OCR correction made lazy/fallback-only):
  FIXED    v25 applied OCR correction (apply_ocr_corrections) eagerly to every
           line/token before splitting, on the theory that this was needed to
           catch collapsed-separator errors like "HTMLICSS" -> "HTML/CSS"
           before _expand_slash ran. In practice this was too aggressive:
           _separator_candidates has no real floor on "is this actually an
           improvement" — it just takes whichever candidate scores marginally
           higher against the allowlist — so it fired on already-clean text
           too, e.g. "Responsive Design" -> "Responsiv/e D/esign" and
           "UI Design" -> "UI D/esign". And it still didn't even fix the
           target case ("HTMLICSS" stayed unsplit) because the spurious score
           bump from mangling clean phrases buried the real fix.

           Fix: OCR correction is no longer applied during extraction at all
           (_parse_section_to_entries and _section_text_to_entries no longer
           touch it). Instead, discover_items_dynamically now does:
             1. Extract entries from raw text (untouched).
             2. Run SBERT matching once on those raw entries (step 4).
             3. ONLY for entries that failed to match (step 4b), run
                apply_ocr_corrections on that specific entry, re-split it on
                '/' and '&', and retry SBERT matching on the resulting pieces.
           This means correction is only ever attempted on text that has
           already proven it doesn't match anything as-is — clean entries
           like "Responsive Design" match on the first pass and are never
           touched by the correction logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from config import BLOB_FUZZY_THRESHOLD
from core.utils import (
    normalize,
    build_allow_norm_sorted,
    canonicalize,
    fuzzy_scan,
)
from core.ocr_correction import build_ocr_corrector, get_sbert

log = logging.getLogger(__name__)

_SBERT_COSINE_THRESHOLD:  float = 0.80
_SINGLE_TOKEN_CONFIDENCE: int   = 85

# ---------------------------------------------------------------------------
# Shared regex constants
# ---------------------------------------------------------------------------

# Generic next-section boundary used as a fallback in all extractors.
_NEXT_SECTION_RE = re.compile(
    r"^\s*[^\w]*(work\s+experience|experience|education|employment|awards?|"
    r"publications?|references?|volunteer|languages?|interests?|activities|"
    r"summary|objective(?![\w\-]))",
    re.I | re.M,
    )

_CONTACT_LINE_RE = re.compile(
    r"@"
    r"|(?:\+?[\d\s\-\(\)]{7,})"
    r"|[\U00002709\U0000260E\U0001F4DE\U0001F4E7\U0001F4E9\U0001F4CC\U0001F4CD\U0001F4F1]"
    r"|\b(?:www\.|http)",
    re.I,
)

_EXPERIENCE_LINE_RE = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|present|current"
    r"|\b(developed?|managed?|led|built|designed?|implemented?|integrated?|"
    r"optimized?|maintained?|participated?|created?|delivered?|deployed?|"
    r"automated?|reduced?|improved?|increased?|collaborated?|coordinated?)\b",
    re.I,
)

_BANNER_STRUCT_RE = re.compile(
    r"^[A-Za-záéíóúñüÁÉÍÓÚÑÜ][A-Za-záéíóúñüÁÉÍÓÚÑÜ\s\-]{4,59}$"
)

_BULLET_RE = re.compile(r"^\s*[\u2022\-\*\u25e6\u25aa]|\s*\d+\.\s+", re.M)


# ---------------------------------------------------------------------------
# Configuration dataclass — the only thing callers need to supply
# ---------------------------------------------------------------------------

@dataclass
class SectionConfig:
    """
    Describes how to find a specific resume section.

    heading_re              : matches the section heading line (e.g. "SKILLS", "CERTIFICATIONS").
    end_re                  : matches the first line of the next section (stops extraction).
    corpus_key              : key inside each jobs_corpus dict that holds the allowlist
                              (e.g. "required_skills", "required_certifications").
    label                   : short name for log messages ("Skills", "Certs", …).
    preprocess_section_text : optional callable(str) -> str applied to the raw section
                              text *before* it is split into entries.  Use this for
                              section-specific normalisation that must not affect other
                              extractors (e.g. rejoining cert provider lines).
                              Defaults to None (no-op).
    """
    heading_re:              re.Pattern
    end_re:                  re.Pattern
    corpus_key:              str
    label:                   str
    preprocess_section_text: "callable[[str], str] | None" = None


# ---------------------------------------------------------------------------
# Parenthesis protection / restoration
# ---------------------------------------------------------------------------

def _protect_parens(text: str) -> tuple[str, dict[str, str]]:
    """Replace commas inside parentheses with placeholders to avoid splitting compound items."""
    pmap: dict[str, str] = {}

    def replacer(m: re.Match) -> str:
        key = f"__PAREN_{len(pmap)}__"
        pmap[key] = m.group(0)
        return key

    return re.sub(r"\([^)]*\)", replacer, text), pmap


def _restore_parens(text: str, pmap: dict[str, str]) -> str:
    for key, original in pmap.items():
        text = text.replace(key, original)
    return text


# ---------------------------------------------------------------------------
# Banner-line detection
# ---------------------------------------------------------------------------

def _is_banner_line(line: str, allowlist: list[str]) -> bool:
    """
    Return True if line looks like a personal name or job-title banner injected
    between section content and the next heading. Three layers must all pass:
      1. Structural: letters/spaces/hyphens only, 2+ words, length 5–60, no digits.
      2. Word shape: every word is pure Title Case or pure ALL CAPS.
      3. Allowlist guard: fuzzy score < 65 against all known items
         (token_set_ratio catches subset matches like "CRUD Application Development").
    """
    if not _BANNER_STRUCT_RE.match(line) or " " not in line:
        return False
    if any(ch.isdigit() for ch in line):
        return False
    for word in line.split():
        w = word.strip("-")
        if w and not (w[0].isupper() and w[1:].islower() or w.isupper()):
            return False
    if allowlist:
        line_norm = line.lower().strip()
        for item in allowlist:
            score = max(
                fuzz.ratio(line_norm, item.lower()),
                fuzz.token_sort_ratio(line_norm, item.lower()),
                fuzz.token_set_ratio(line_norm, item.lower()),
            )
            if score >= 65:
                log.debug(f"[BannerCheck] '{line}' score={score} vs '{item}' -> treated as item")
                return False
    return True


# ---------------------------------------------------------------------------
# Subsumed-item detection
# ---------------------------------------------------------------------------

def _is_subsumed(item: str, others: list[str]) -> bool:
    """
    Return True when item is a parenthetical fragment split out of a longer
    item in others (e.g. "SolidWorks" from "CAD Software (SolidWorks, AutoCAD)").
    Distinct items that merely share words are always kept.
    """
    s_lower   = item.lower()
    pattern   = re.compile(r'(?<![a-z0-9])' + re.escape(s_lower) + r'(?![a-z0-9])')

    for other in others:
        if other.lower() == s_lower:
            continue
        other_l = other.lower()
        if len(other_l) <= len(s_lower):
            continue
        m = pattern.search(other_l)
        if not m:
            continue
        following    = other_l[m.end()] if m.end() < len(other_l) else ''
        before_match = other_l[:m.start()]
        if following in (',', ')') and before_match.count('(') > before_match.count(')'):
            return True
    return False


# ---------------------------------------------------------------------------
# OCR blob segmentation
# ---------------------------------------------------------------------------

def _split_ocr_blob(line: str, allowlist: list[str]) -> list[str]:
    """
    Segment a collapsed OCR blob into individual item phrases.
    Checks single-token confidence first; falls back to longest-window search.
    """
    if not allowlist or not line.strip():
        return [line.strip()] if line.strip() else []

    sorted_allow = sorted(allowlist, key=lambda a: len(a.split()), reverse=True)
    norm_allow   = [(normalize(a), a) for a in sorted_allow]
    max_win      = max(len(a.split()) for a in sorted_allow)

    tokens = line.split()
    result: list[str] = []
    i = 0

    while i < len(tokens):
        w1_norm = normalize(tokens[i])
        thr1    = (BLOB_FUZZY_THRESHOLD - 10 if len(w1_norm) <= 5
                   else BLOB_FUZZY_THRESHOLD - 5 if len(w1_norm) <= 9
        else BLOB_FUZZY_THRESHOLD)

        w1_score, w1_val = 0, ""
        for akey, aval in norm_allow:
            s = max(fuzz.token_sort_ratio(w1_norm, akey), fuzz.ratio(w1_norm, akey))
            if s > w1_score:
                w1_score, w1_val = s, aval

        if w1_score >= _SINGLE_TOKEN_CONFIDENCE:
            result.append(w1_val)
            i += 1
            continue

        best_score = w1_score if w1_score >= thr1 else 0
        best_canon = w1_val   if w1_score >= thr1 else None
        best_len   = 1        if w1_score >= thr1 else 0

        for win_len in range(min(max_win, len(tokens) - i), 1, -1):
            wn   = normalize(" ".join(tokens[i : i + win_len]))
            wlen = len(wn)
            thr  = (BLOB_FUZZY_THRESHOLD - 10 if wlen <= 5
                    else BLOB_FUZZY_THRESHOLD - 5 if wlen <= 9
            else BLOB_FUZZY_THRESHOLD)
            ws, wv = 0, ""
            for akey, aval in norm_allow:
                s = max(fuzz.token_sort_ratio(wn, akey), fuzz.ratio(wn, akey))
                if s > ws:
                    ws, wv = s, aval
            if ws >= thr and ws > best_score:
                best_score, best_canon, best_len = ws, wv, win_len

        if best_canon:
            result.append(best_canon)
            i += best_len
        else:
            raw = tokens[i].strip(".,;:()[]")
            if raw:
                result.append(raw)
            i += 1

    return result


# ---------------------------------------------------------------------------
# Line-merging helpers
# ---------------------------------------------------------------------------

def _score(text_norm: str, allowlist: list[str]) -> int:
    return max(
        (max(fuzz.ratio(text_norm, normalize(a)), fuzz.token_sort_ratio(text_norm, normalize(a)))
         for a in allowlist),
        default=0,
    )


def _merge_split_lines(lines: list[str], allowlist: list[str]) -> list[str]:
    """
    Merge consecutive OCR-split lines that together form a single allowlist item.
    Tries both forward (A + B) and reversed (B + A) joins to handle column-layout
    word-order inversions. Guards prevent merging two valid standalone items.
    """
    if not allowlist or len(lines) < 2:
        return lines

    merged: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if len(cur.split()) == 1 and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            # Don't merge if either line is a mid-list OCR fragment:
            # trailing comma means it's part of a comma-delimited list,
            # unclosed paren means it's the start of a multi-line parenthetical.
            if (cur.endswith(",") or nxt.endswith(",")
                    or cur.count("(") > cur.count(")")
                    or nxt.count("(") > nxt.count(")")):
                merged.append(cur)
                i += 1
                continue
            if 1 <= len(nxt.split()) <= 2:
                cur_score = _score(normalize(cur), allowlist)
                nxt_score = _score(normalize(nxt), allowlist)
                if cur_score < 85 and nxt_score < 85:
                    fwd     = cur + " " + nxt
                    rev     = nxt + " " + cur
                    fs      = _score(normalize(fwd), allowlist)
                    rs      = _score(normalize(rev), allowlist)
                    best    = max(fs, rs)
                    best_s  = rev if rs > fs else fwd
                    if best >= 80:
                        log.debug(f"[MergeLines] {cur!r} + {nxt!r} -> {best_s!r}")
                        merged.append(best_s)
                        i += 2
                        continue
                elif nxt_score >= 85:
                    log.debug(f"[MergeLines] Skipping merge — {nxt!r} is standalone")
        merged.append(cur)
        i += 1
    return merged


def _rejoin_open_parens(lines: list[str]) -> list[str]:
    """
    Rejoin lines where OCR split an open parenthesis across multiple lines.
    Buffers until open paren count balances; emits unclosed parens as-is.
    """
    result: list[str] = []
    buf = ""

    for line in lines:
        stripped = line.strip()
        if buf:
            buf = buf.rstrip() + " " + stripped
            if buf.count("(") <= buf.count(")"):
                result.append(buf)
                buf = ""
        else:
            if stripped.count("(") > stripped.count(")"):
                buf = stripped
            else:
                result.append(stripped)

    if buf:
        log.debug(f"[ReJoinParens] Unclosed parenthesis at EOF, emitting: {buf!r}")
        result.append(buf)

    return result


# ---------------------------------------------------------------------------
# Compound expansion helpers
# ---------------------------------------------------------------------------

def _fuzzy_score(entry: str, allowlist: list[str]) -> int:
    en = normalize(entry)
    return max(
        (max(fuzz.ratio(en, normalize(a)), fuzz.token_sort_ratio(en, normalize(a)))
         for a in allowlist),
        default=0,
    )


def _expand_parentheticals(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    Expand "X (A, B, C)" into ["X", "A", "B", "C"] only when the full compound
    is not itself a known allowlist item but at least one inner part is.
    Without an allowlist, always expand.
    """
    expanded: list[str] = []
    for entry in entries:
        m = re.match(r'^(.*?)\s*\(([^)]+)\)\s*$', entry.strip())
        if not m:
            expanded.append(entry)
            continue
        base  = m.group(1).strip()
        parts = [p.strip() for p in re.split(r'[,;]\s*', m.group(2)) if p.strip()]
        if not allowlist:
            if base:
                expanded.append(base)
            expanded.extend(parts)
            continue
        if _fuzzy_score(entry, allowlist) >= 75:
            expanded.append(entry)
            continue
        if any(_fuzzy_score(p, allowlist) >= 75 for p in parts):
            if base:
                expanded.append(base)
            expanded.extend(parts)
        else:
            expanded.append(entry)
    return expanded


def _expand_ampersand(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    Split "A & B" / "A and B" into separate entries unless the compound scores
    >= 90 as a known item (e.g. "Jest & Enzyme" stays intact).
    """
    _AMP_RE = re.compile(r'\s+(?:&|and)\s+', re.I)
    expanded: list[str] = []
    for entry in entries:
        if not _AMP_RE.search(entry):
            expanded.append(entry)
            continue
        if allowlist and _fuzzy_score(entry, allowlist) >= 90:
            expanded.append(entry)
            continue
        expanded.extend(p.strip() for p in _AMP_RE.split(entry) if p.strip())
    return expanded


# Characters that OCR commonly substitutes for '/'
_SLASH_SURROGATES: frozenset[str] = frozenset("Il1|")


def _normalize_sep_local(s: str) -> str:
    """
    Separator-aware normalize (local copy to avoid circular import).
    Converts '/' and '&' to spaces instead of stripping them, so that
    "IDS/IPS" -> "ids ips" and "IDS/IIPS" -> "ids iips" are distinguishable.
    """
    s = s.lower()
    s = re.sub(r"[/&]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _restore_ocr_slash(entries: list[str], allowlist: list[str]) -> list[str]:
    if not allowlist:
        return entries

    allow_sep_keys = [_normalize_sep_local(a) for a in allowlist]

    def _score(candidate: str) -> tuple[float, float]:
        primary   = max(
            (fuzz.ratio(_normalize_sep_local(candidate), ak) for ak in allow_sep_keys),
            default=0,
        )
        secondary = max(
            (fuzz.ratio(candidate.lower(), a.lower()) for a in allowlist),
            default=0,
        )
        return (primary, secondary)

    restored: list[str] = []
    for entry in entries:
        # Skip entries that already contain a slash or are too short to split.
        if "/" in entry or len(entry) < 4:
            restored.append(entry)
            continue

        best       = _score(entry)
        best_entry = entry

        for i, ch in enumerate(entry):
            if ch not in _SLASH_SURROGATES:
                continue
            if i == 0 or i == len(entry) - 1:
                continue
            left, right = entry[:i], entry[i + 1:]
            # Require at least one uppercase letter on each side.
            if not (any(c.isupper() for c in left) and any(c.isupper() for c in right)):
                continue
            candidate = left + "/" + right
            s = _score(candidate)
            if s > best:
                best       = s
                best_entry = candidate

        if best_entry != entry:
            log.debug(
                f"[RestoreSlash] '{entry}' -> '{best_entry}' "
                f"(primary={best[0]:.0f}, secondary={best[1]:.0f})"
            )

        restored.append(best_entry)

    return restored

def _expand_slash(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    Split "QA/QC Procedures" into ["QA Procedures", "QC Procedures"] when the
    full compound is unknown but both halves score >= 75. Known compounds like
    "TCP/IP" (score >= 85) are always kept intact.
    """
    expanded: list[str] = []
    for entry in entries:
        if '/' not in entry:
            expanded.append(entry)
            continue
        if allowlist and _fuzzy_score(entry, allowlist) >= 85:
            expanded.append(entry)
            continue
        m = re.search(r'(\w+)/(\w+)(.*)', entry)
        if not m:
            expanded.append(entry)
            continue
        left  = (m.group(1) + m.group(3)).strip()
        right = (m.group(2) + m.group(3)).strip()
        if allowlist and _fuzzy_score(left, allowlist) >= 75 and _fuzzy_score(right, allowlist) >= 75:
            log.debug(f"[ExpandSlash] Splitting {entry!r} -> ['{left}', '{right}']")
            expanded.extend([left, right])
        else:
            expanded.append(entry)
    return expanded


def _strip_subheader(line: str, allowlist: list[str] | None = None) -> str:
    """Strip category labels like 'Frontend: React' -> 'React', guarded by allowlist scoring."""
    if ":" not in line:
        return line
    if allowlist and _fuzzy_score(line, allowlist) >= 75:
        return line
    colon_idx = line.index(":")
    before    = line[:colon_idx].strip()
    after     = line[colon_idx + 1:].strip()
    if "," in before or len(before.split()) > 5 or not after:
        return line
    if allowlist and _fuzzy_score(after, allowlist) < 50:
        return line
    log.debug(f"[StripSubheader] '{before}' -> '{after}'")
    return after


# ---------------------------------------------------------------------------
# Double-newline paren reunification
# ---------------------------------------------------------------------------

def _reunite_double_nl_parens(raw_block: str) -> str:
    """
    Rejoin chunks separated by \\n\\n when a chunk has an unclosed parenthesis.

    EasyOCR splits two-column layouts on \\n\\n, which can cut a parenthetical
    skill across multiple chunks, e.g.:
        "MERN Stack (MongoDB, Expressjs,"  <- opens paren
        "Node js)"                         <- closes paren

    This function merges such chunks before any further splitting so the
    parenthetical is treated as a single candidate entry.
    """
    chunks = re.split(r"\n{2,}", raw_block)
    print(f"\n[ReuniteParens] {len(chunks)} raw chunks from \\n\\n split:")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c.strip()!r}")

    result: list[str] = []
    buf = ""
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if buf:
            buf = buf.rstrip().rstrip(",") + ", " + chunk
            if buf.count("(") <= buf.count(")"):
                print(f"  [ReuniteParens] Paren closed -> emitting: {buf!r}")
                result.append(buf)
                buf = ""
            else:
                print(f"  [ReuniteParens] Still open  -> buffering: {buf!r}")
        elif chunk.count("(") > chunk.count(")"):
            print(f"  [ReuniteParens] Unclosed paren -> buffering: {chunk!r}")
            buf = chunk
        else:
            result.append(chunk)
    if buf:
        print(f"  [ReuniteParens] EOF unclosed -> emitting as-is: {buf!r}")
        result.append(buf)

    print(f"\n[ReuniteParens] {len(result)} chunks after reunification:")
    for i, r in enumerate(result):
        print(f"  [{i}] {r!r}")
    return "\n\n".join(result)


# ---------------------------------------------------------------------------
# Section text extraction
# ---------------------------------------------------------------------------

_NAME_RE  = re.compile(r'^[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ\s\-]{2,49}$')
_TITLE_RE = re.compile(r'^[A-Za-záéíóúñü][A-Za-záéíóúñü0-9\s\-\(\)\/]{4,79}$')


def _clip_at_name_banner(block: str) -> str:
    """
    Trim a raw section block at the first ALL-CAPS full name line that is
    immediately followed by a job-title line, e.g.:

        NERISSA TOLENTINO
        Technical Support Engineer (Tier 2)

    This prevents bleed-through from the next candidate's header when
    multiple resumes are concatenated or a layout places the next name
    inside the skills block.
    """
    lines = block.splitlines()
    clipped: list[str] = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            clipped.append(line)
            continue
        if (_NAME_RE.match(s) and s.isupper()
                and 2 <= len(s.split()) <= 4
                and not any(ch.isdigit() for ch in s)):
            # Look ahead for a job-title line
            next_stripped = [l.strip() for l in lines[idx + 1:] if l.strip()]
            if next_stripped and _TITLE_RE.match(next_stripped[0]):
                log.debug(
                    f"[ExtractSection] Stopped at name+title banner: "
                    f"{s!r} / {next_stripped[0]!r}"
                )
                break
        clipped.append(line)
    return "\n".join(clipped).strip()


def extract_section(text: str, config: SectionConfig) -> str:
    """Locate heading, slice to the next section boundary, return raw block text."""
    m = config.heading_re.search(text)
    if not m:
        return ""
    rest  = text[m.end():]
    end_m = config.end_re.search(rest)
    block = rest[: end_m.start()].strip() if end_m else rest.strip()
    return _clip_at_name_banner(block)


# ---------------------------------------------------------------------------
# Shared section parsing  (heading -> kept_lines -> entries)
# ---------------------------------------------------------------------------

def _parse_section_to_entries(
        text:          str,
        config:        SectionConfig,
        allowlist:     list[str],
        is_native_pdf: bool = False,
) -> list[str]:
    """
    Locate the section, filter lines, and split by delimiter.
    Returns a flat list of raw candidate strings ready for SBERT / fuzzy matching.

    Note (v26): OCR correction is intentionally NOT applied here anymore.
    Eagerly correcting every line/token before splitting was too aggressive —
    it fired on already-clean text (e.g. "Responsive Design" -> "Responsiv/e
    D/esign") because the separator-recovery heuristic only needs a marginal
    score improvement against the allowlist to commit, and that can trigger
    on phrases that never needed correction in the first place.

    OCR correction now happens centrally in discover_items_dynamically, and
    only as a fallback for entries that SBERT fails to match on the first
    pass — i.e. only when it's actually needed.
    """
    m = config.heading_re.search(text)
    if not m:
        return []

    rest  = text[m.end():]
    end_m = config.end_re.search(rest)
    block = rest[: end_m.start()].strip() if end_m else rest.strip()

    # Capture up to 5 lines above the heading (some layouts place items there).
    before_lines = [l.strip() for l in text[: m.start()].splitlines() if l.strip()]
    pre: list[str] = []
    for lb in reversed(before_lines[-5:]):
        if config.end_re.match(lb) or _CONTACT_LINE_RE.search(lb):
            break
        if _EXPERIENCE_LINE_RE.search(lb) or len(lb.split()) > 5:
            break
        pre.insert(0, lb)

    raw_lines = pre + block.splitlines()

    kept_lines: list[str] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if kept_lines and _CONTACT_LINE_RE.search(s):
            break
        if kept_lines and _is_banner_line(s, allowlist):
            break
        if not is_native_pdf and kept_lines and len(s.split()) >= 3 and _EXPERIENCE_LINE_RE.search(s):
            break
        kept_lines.append(s)

    if not kept_lines:
        return []

    if "\n\n" in block:
        print(f"\n[{config.label}] Double-newline layout detected — using two-column deinterleave path.")

        reunited = _reunite_double_nl_parens(block)
        paragraphs: list[str] = []
        for para in re.split(r"\n{2,}", reunited):
            para = para.strip().strip(",").strip()
            if not para:
                continue
            protected_para, pmap_para = _protect_parens(para)
            for sub in re.split(r"\s*,\s*", protected_para):
                sub = _restore_parens(sub.strip(), pmap_para).strip().strip(",").strip()
                if sub:
                    paragraphs.append(sub)

        print(f"\n[{config.label}] {len(paragraphs)} raw candidates after paragraph + comma split:")
        for i, p in enumerate(paragraphs):
            print(f"  [{i}] {p!r}")

        double_nl_lines: list[str] = []
        for p in paragraphs:
            if double_nl_lines and _CONTACT_LINE_RE.search(p):
                print(f"  [Filter] Stopped at contact line: {p!r}")
                break
            if double_nl_lines and _is_banner_line(p, allowlist):
                print(f"  [Filter] Stopped at banner line: {p!r}")
                break
            if not is_native_pdf and double_nl_lines and len(p.split()) >= 3 and _EXPERIENCE_LINE_RE.search(p):
                print(f"  [Filter] Stopped at experience line: {p!r}")
                break
            double_nl_lines.append(p)

        print(f"\n[{config.label}] {len(double_nl_lines)} candidates after guard filtering:")
        for i, p in enumerate(double_nl_lines):
            print(f"  [{i}] {p!r}")

        if double_nl_lines:
            double_nl_lines = _merge_split_lines(double_nl_lines, allowlist)
            double_nl_lines = _rejoin_open_parens(double_nl_lines)
            entries_out: list[str] = [_strip_subheader(ln, allowlist) for ln in double_nl_lines if ln.strip()]
            entries_out = _expand_ampersand(entries_out, allowlist)
            print(f"\n[{config.label}] After expand_ampersand (pre-paren):")
            for i, e in enumerate(entries_out):
                print(f"  [{i}] {e!r}")

            expanded: list[str] = []
            for ln in entries_out:
                for sub in _expand_parentheticals([ln], allowlist):
                    if sub.strip():
                        expanded.append(sub.strip())
            entries_out = expanded

            print(f"\n[{config.label}] After expand_parentheticals:")
            for i, e in enumerate(entries_out):
                print(f"  [{i}] {e!r}")

            entries_out = _restore_ocr_slash(entries_out, allowlist)
            entries_out = _expand_slash(entries_out, allowlist)

            print(f"\n[{config.label}] Final entries going into SBERT ({len(entries_out)} total):")
            for i, e in enumerate(entries_out):
                print(f"  [{i}] {e!r}")

            return entries_out

    kept_lines   = _merge_split_lines(kept_lines, allowlist)
    kept_lines   = _rejoin_open_parens(kept_lines)
    joined_block = "\n".join(kept_lines)

    entries: list[str] = []

    if "," in joined_block:
        protected, pmap = _protect_parens(joined_block)
        stopped = False
        for part in re.split(r"\s*,\s*|\s*\n\s*", protected):
            if stopped:
                break
            part = _restore_parens(part.strip(), pmap)
            part = _strip_subheader(part, allowlist)
            if not part:
                continue
            if len(re.sub(r'\([^)]*\)', '', part).strip().split()) > 6:
                stopped = True
                break
            entries.extend(_expand_parentheticals([part], allowlist))

    elif ";" in joined_block:
        for line in kept_lines:
            for part in re.split(r"\s*[;\n]\s*", _strip_subheader(line, allowlist)):
                if part.strip():
                    entries.append(part.strip())

    elif "|" in joined_block:
        for line in kept_lines:
            for part in re.split(r"\s*[|\n]\s*", _strip_subheader(line, allowlist)):
                if part.strip():
                    entries.append(part.strip())

    elif _BULLET_RE.search(joined_block):
        for line in kept_lines:
            clean = re.sub(r"^\s*[\u2022\-\*\u25e6\u25aa]\s*|\s*\d+\.\s+",
                           "", _strip_subheader(line, allowlist)).strip()
            if clean:
                entries.append(clean)

    elif len(kept_lines) > 1:
        entries = [_strip_subheader(l, allowlist).strip() for l in kept_lines if l.strip()]

    else:
        blob = kept_lines[0]
        entries = _split_ocr_blob(blob, allowlist) if allowlist else [blob]

    entries = _expand_ampersand(entries, allowlist)
    entries = _restore_ocr_slash(entries, allowlist)
    entries = _expand_slash(entries, allowlist)
    return entries


# ---------------------------------------------------------------------------
# SBERT matching
# ---------------------------------------------------------------------------

def _sbert_match(
        entries:       list[str],
        allowlist:     list[str],
        already_found: set[str],
        threshold:     float = _SBERT_COSINE_THRESHOLD,
        label:         str   = "SBERT",
) -> tuple[list[str], list[str]]:
    """
    Encode entries and allowlist with SBERT.

    Returns (discovered, unmatched):
      discovered : canonical allowlist names whose cosine similarity to a
                   candidate meets threshold.
      unmatched  : candidate strings (post parenthetical-expansion) that did
                   NOT reach a match. Callers can retry these through OCR
                   correction before giving up on them — see v26 fallback
                   in discover_items_dynamically.

    Expands parentheticals before encoding so each component is matched independently.
    """
    if not entries:
        log.info(f"[{label}] SBERT skipped — no candidate entries.")
        return [], []
    if not allowlist:
        log.info(f"[{label}] SBERT skipped — allowlist is empty (no {label.lower()} in jobs corpus).")
        return [], list(entries)

    sbert   = get_sbert()
    entries = _expand_parentheticals(entries)

    log.info(f"[{label}] Encoding {len(allowlist)} allowlist items...")
    allow_vecs = sbert.encode(allowlist, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    pending = [e for e in entries if e.lower().strip() not in already_found]
    if not pending:
        return [], []

    log.info(f"[{label}] Encoding {len(pending)} candidates...")
    entry_vecs = sbert.encode(pending, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    sim_matrix = entry_vecs @ allow_vecs.T   # cosine sim (both L2-normalised)

    discovered: list[str] = []
    unmatched:  list[str] = []
    seen = set(already_found)

    for i, entry in enumerate(pending):
        idx   = int(np.argmax(sim_matrix[i]))
        score = float(sim_matrix[i, idx])
        canon = allowlist[idx]
        if score >= threshold:
            key = canon.lower().strip()
            if key not in seen:
                seen.add(key)
                discovered.append(canon)
            log.debug(f"[{label}] '{entry}' -> '{canon}' (cosine={score:.4f})")
        else:
            log.debug(f"[{label}] '{entry}' -> no match (best='{canon}' cosine={score:.4f})")
            unmatched.append(entry)

    return discovered, unmatched


# ---------------------------------------------------------------------------
# Section text -> entries (used when pre-extracted section_text is provided)
# ---------------------------------------------------------------------------

def _section_text_to_entries(
        section_text: str,
        allowlist:    list[str],
) -> list[str]:
    """
    Convert an already-extracted section text string into a flat list of
    candidate entries for SBERT matching.

    This is the fast path used when skills.py / certs.py pass in their
    rule-based section text directly, bypassing _parse_section_to_entries
    and its re-extraction from raw resume text.

    Note (v26): OCR correction is intentionally NOT applied here. See the
    docstring on _parse_section_to_entries for why — eager correction over
    every token mangles clean text. Correction now happens centrally in
    discover_items_dynamically, only as a fallback for entries SBERT fails
    to match.

    Reuses the same delimiter logic and compound expansion as the main path.
    """
    lines = [l.strip() for l in section_text.splitlines() if l.strip()]
    if not lines:
        return []

    # ── Name + job-title banner guard ───────────────────────────────────────
    # Some resume layouts bleed into the next candidate's header block, e.g.:
    #   "NERISSA TOLENTINO"
    #   "Technical Support Engineer (Tier 2)"
    # Stop as soon as we see a line that looks like an ALL-CAPS full name
    # immediately followed by a job-title line.
    # ALL-CAPS words only, 2–4 tokens, no digits → name line.
    # Next non-empty line is a job title if it is mixed case, 2–6 words,
    # no digits, and scores low against the allowlist (not itself a skill).
    clipped: list[str] = []
    for idx, line in enumerate(lines):
        # ALL-CAPS full name (2–4 words, no digits)
        if (_NAME_RE.match(line) and line.isupper()
                and 2 <= len(line.split()) <= 4
                and not any(ch.isdigit() for ch in line)):
            next_lines = [l for l in lines[idx + 1:] if l.strip()]
            if next_lines and _TITLE_RE.match(next_lines[0]):
                next_score = _fuzzy_score(next_lines[0], allowlist) if allowlist else 0
                if next_score < 60:
                    log.debug(
                        f"[SectionText] Stopped at name+title banner: "
                        f"{line!r} / {next_lines[0]!r}"
                    )
                    break
        # Single-line banner guard (existing helper)
        if clipped and _is_banner_line(line, allowlist):
            log.debug(f"[SectionText] Stopped at banner line: {line!r}")
            break
        clipped.append(line)

    lines = clipped
    if not lines:
        return []
    # ────────────────────────────────────────────────────────────────────────

    lines = _merge_split_lines(lines, allowlist)
    lines = _rejoin_open_parens(lines)
    joined = "\n".join(lines)

    entries: list[str] = []

    if "," in joined:
        protected, pmap = _protect_parens(joined)
        for part in re.split(r"\s*,\s*|\s*\n\s*", protected):
            part = _restore_parens(part.strip(), pmap)
            part = _strip_subheader(part, allowlist)
            if not part:
                continue
            if len(re.sub(r'\([^)]*\)', '', part).strip().split()) > 6:
                break
            entries.extend(_expand_parentheticals([part], allowlist))

    elif ";" in joined:
        for line in lines:
            for part in re.split(r"\s*[;\n]\s*", _strip_subheader(line, allowlist)):
                if part.strip():
                    entries.append(part.strip())

    elif "|" in joined:
        for line in lines:
            for part in re.split(r"\s*[|\n]\s*", _strip_subheader(line, allowlist)):
                if part.strip():
                    entries.append(part.strip())

    elif _BULLET_RE.search(joined):
        for line in lines:
            clean = re.sub(r"^\s*[\u2022\-\*\u25e6\u25aa]\s*|\s*\d+\.\s+",
                           "", _strip_subheader(line, allowlist)).strip()
            if clean:
                entries.append(clean)

    elif len(lines) > 1:
        entries = [_strip_subheader(l, allowlist).strip() for l in lines if l.strip()]

    else:
        entries = _split_ocr_blob(lines[0], allowlist) if allowlist else [lines[0]]

    entries = _expand_ampersand(entries, allowlist)
    entries = _restore_ocr_slash(entries, allowlist)
    entries = _expand_slash(entries, allowlist)

    log.debug(f"[SectionText] {len(entries)} entries from pre-extracted section text.")
    return entries


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def discover_items_dynamically(
        resume_text:   str,
        jobs_corpus:   list[dict],
        config:        SectionConfig,
        extra_items:   list[str] | None = None,
        is_native_pdf: bool = False,
        section_text:  str | None = None,
) -> list[str]:
    """
    Full extraction pipeline shared by skills and certs.

    Steps:
      1. Build allowlist from jobs_corpus[config.corpus_key].
      2. Build OcrCorrector once from the allowlist — held in reserve, only
         used in step 4b below.
      3. Parse section into candidate entries (no OCR correction applied
         here — entries are extracted as-is from raw or pre-extracted text).
         - Fast path: section_text is provided by the caller.
         - Slow path: re-extract from raw resume_text.
      4. SBERT pass — semantic matching (cosine >= 0.80) on the raw entries.
      4b. OCR-correction retry — ONLY for entries that failed to match in
          step 4. Each unmatched entry is run through apply_ocr_corrections,
          re-split on '/' and '&', and retried through SBERT. This keeps
          correction lazy: clean entries that already matched are never
          touched, so there's no risk of correction mangling text that
          didn't need it.
      5. Fuzzy fallback — if SBERT found nothing at all.
      6. Drop parenthetical fragment duplicates.
    """
    # 1. Build allowlist
    allowlist: set[str] = set()
    for job in jobs_corpus:
        allowlist.update(job.get(config.corpus_key, []))
    if extra_items:
        allowlist.update(extra_items)
    allowlist_list = sorted(allowlist)

    # 2. Build OcrCorrector once (FAISS index over allowlist embeddings).
    #    Skipped only for native PDFs — their text is already clean.
    #    Used ONLY as a fallback below (step 4b), for entries SBERT fails
    #    to match on the first pass — not applied eagerly to every entry.
    corrector = None
    if not is_native_pdf and allowlist_list:
        log.info(f"[{config.label}] Building OcrCorrector from {len(allowlist_list)} allowlist entries...")
        corrector = build_ocr_corrector(allowlist_list)
        log.info(f"[{config.label}] OcrCorrector ready.")

    # 3. Parse section into candidate entries
    if section_text is not None:
        # Fast path: caller already extracted the section text.
        # If section_text is empty the section simply doesn't exist in this resume
        # (e.g. no cert section) — skip the emergency fallback entirely.
        if section_text:
            log.info(f"[{config.label}] Using pre-extracted section text ({len(section_text)} chars).")
            if config.preprocess_section_text is not None:
                section_text = config.preprocess_section_text(section_text)
                log.debug(f"[{config.label}] preprocess_section_text applied.")
            entries = _section_text_to_entries(section_text, allowlist_list)
        else:
            log.info(f"[{config.label}] No {config.label.lower()} section found — skipping.")
            entries = []
    else:
        # Slow path: re-extract from raw resume text.
        entries = _parse_section_to_entries(
            resume_text,
            config,
            allowlist_list,
            is_native_pdf,
        )
        # Emergency fallback only on the slow path — section heading may have been
        # missed entirely, so scan the first 3000 chars as a last resort.
        if not entries:
            log.info(f"[{config.label}] Section not found — emergency comma-split fallback.")
            entries = [e.strip() for e in resume_text[:3000].split(",") if e.strip()]

    if entries:
        log.info(f"[{config.label}] Final entries going into SBERT ({len(entries)} total):")
        for i, e in enumerate(entries):
            log.info(f"  [{i}] {e!r}")

    # 4. SBERT pass on the raw, uncorrected entries.
    seen:      set[str]  = set()
    canonical: list[str] = []

    matched, unmatched = _sbert_match(entries, allowlist_list, seen, label=config.label)
    for canon in matched:
        key = canon.lower().strip()
        if key not in seen:
            seen.add(key)
            canonical.append(canon)

    log.info(f"[{config.label}] After SBERT (raw pass): {len(canonical)} matched, {len(unmatched)} unmatched.")

    # 4b. OCR-correction retry — ONLY for entries SBERT failed to match.
    # This is deliberately a fallback rather than an eager pass over every
    # entry: correcting clean text up front (e.g. "Responsive Design") risks
    # mangling tokens that never needed it. Only entries that already failed
    # to find a home in the allowlist are worth a forced repair attempt.
    #
    # Two candidate sources are tried for each unmatched entry:
    #   1. apply_ocr_corrections() — handles letter/digit OCR confusions and,
    #      when the allowlist holds a COMBINED phrase (e.g. an "IDS/IPS"-style
    #      entry), its separator recovery can resolve that too.
    #   2. Direct surrogate-character separator insertion — apply_ocr_corrections'
    #      separator recovery scores the WHOLE joined candidate against
    #      individual allowlist entries, so it can't reliably recover a split
    #      like "HTMLICSS" -> "HTML/CSS" when "HTML" and "CSS" are SEPARATE
    #      allowlist entries rather than a combined phrase — splitting never
    #      scores higher than leaving the blob alone matching one half's
    #      prefix. So we also try inserting '/' at every common OCR-surrogate
    #      character position ('I', 'l', '1', '|') directly here.
    #
    # Every candidate from both sources is run through _expand_ampersand and
    # then _expand_slash, which (unlike the correction above) score the two
    # resulting halves INDEPENDENTLY against the allowlist — that's what
    # correctly turns "HTML/CSS" into ["HTML", "CSS"] rather than requiring
    # the allowlist to contain that exact compound phrase.
    if unmatched and corrector is not None:
        from core.ocr_correction import apply_ocr_corrections

        _SLASH_SURROGATES = frozenset("Il1|")

        retry_candidates: list[str] = []
        for entry in unmatched:
            entry_key = entry.strip().lower()
            split_candidates: set[str] = set()

            # Source 1: token-level OCR correction.
            corrected = apply_ocr_corrections(entry, corrector)
            if corrected.strip().lower() != entry_key:
                split_candidates.add(corrected)

            # Source 2: direct surrogate-character separator insertion.
            stripped = entry.strip(".,;:()[]")
            for i, ch in enumerate(stripped):
                if ch not in _SLASH_SURROGATES or i == 0 or i == len(stripped) - 1:
                    continue
                split_candidates.add(stripped[:i] + "/" + stripped[i + 1:])  # consume surrogate
                split_candidates.add(stripped[:i] + "/" + stripped[i:])      # keep surrogate

            for cand in split_candidates:
                if cand.strip().lower() == entry_key:
                    continue
                for piece in _expand_slash(
                        _expand_ampersand([cand], allowlist_list), allowlist_list
                ):
                    piece = piece.strip()
                    if piece and piece.lower() not in seen and piece.lower() != entry_key:
                        retry_candidates.append(piece)

        if retry_candidates:
            log.info(
                f"[{config.label}] Retrying {len(retry_candidates)} OCR-corrected "
                f"candidate(s) from {len(unmatched)} unmatched entries: {retry_candidates!r}"
            )
            retried_matched, _ = _sbert_match(
                retry_candidates, allowlist_list, seen, label=f"{config.label}-OCRRetry"
            )
            for canon in retried_matched:
                key = canon.lower().strip()
                if key not in seen:
                    seen.add(key)
                    canonical.append(canon)

    log.info(f"[{config.label}] After SBERT: {len(canonical)} matched.")

    # 5. Fuzzy fallback
    if not canonical:
        log.info(f"[{config.label}] SBERT found nothing — fuzzy fallback.")
        canonical = fuzzy_scan(", ".join(entries), allowlist_list)

    # 6. Drop parenthetical split fragments
    canonical = [s for s in canonical if not _is_subsumed(s, canonical)]

    log.info(f"[{config.label}] {len(canonical)} items discovered.")
    return canonical