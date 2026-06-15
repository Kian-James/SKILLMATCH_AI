"""
core/extractor_base.py
======================
Shared extraction pipeline for Stage 2 (skills) and Stage 2B (certs).

Both extractors are identical in behaviour — they differ only in which section
heading they search for and which Firestore allowlist key they read from.
Everything else (section slicing, line filtering, OCR correction, compound
expansion, SBERT matching, fuzzy fallback, dedup) lives here and is shared.

Public API
----------
  SectionConfig              — dataclass injected by each caller
  discover_items_dynamically — full pipeline; called by skills.py and certs.py
  extract_section            — regex section slicer (exposed for unit tests)
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

    heading_re    : matches the section heading line (e.g. "SKILLS", "CERTIFICATIONS").
    end_re        : matches the first line of the next section (stops extraction).
    corpus_key    : key inside each jobs_corpus dict that holds the allowlist
                    (e.g. "required_skills", "required_certifications").
    label         : short name for log messages ("Skills", "Certs", …).
    """
    heading_re: re.Pattern
    end_re:     re.Pattern
    corpus_key: str
    label:      str


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


def _restore_ocr_slash(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    Restore slashes corrupted by OCR into 'I' or 'l' between uppercase groups
    (e.g. "QAIQC" -> "QA/QC"). Only commits when the replacement scores higher.
    """
    _OCR_SLASH_RE = re.compile(r'([A-Z]{1,6})[Il]([A-Z]{1,6})')
    restored: list[str] = []
    for entry in entries:
        m = _OCR_SLASH_RE.search(entry)
        if not m:
            restored.append(entry)
            continue
        candidate = entry[:m.start()] + m.group(1) + '/' + m.group(2) + entry[m.end():]
        if not allowlist or _fuzzy_score(candidate, allowlist) > _fuzzy_score(entry, allowlist):
            log.debug(f"[RestoreSlash] '{entry}' -> '{candidate}'")
            restored.append(candidate)
        else:
            restored.append(entry)
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
        corrector=None,
) -> list[str]:
    """
    Locate the section, filter lines, apply OCR repairs, and split by delimiter.
    Returns a flat list of raw candidate strings ready for SBERT / fuzzy matching.

    corrector : OcrCorrector built from the allowlist (passed in from
                discover_items_dynamically so it is built only once per call).
                Skipped when is_native_pdf=True or corrector is None.
    """
    from core.ocr_correction import apply_ocr_corrections  # avoid circular at module level

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

    # ── OCR correction pass ──────────────────────────────────────────────────
    if not is_native_pdf and corrector is not None:
        kept_lines = [apply_ocr_corrections(line, corrector) for line in kept_lines]
        log.debug(f"[{config.label}] OCR correction applied to {len(kept_lines)} lines.")
    # ────────────────────────────────────────────────────────────────────────

    if "\n\n" in block:
        print(f"\n[{config.label}] Double-newline layout detected — using two-column deinterleave path.")

        corrected_block = block
        if not is_native_pdf and corrector is not None:
            corrected_block_lines = [
                apply_ocr_corrections(ln, corrector)
                for ln in block.splitlines()
            ]
            corrected_block = "\n".join(corrected_block_lines)
            log.debug(f"[{config.label}] OCR correction applied to raw block for \\n\\n path.")

        reunited = _reunite_double_nl_parens(corrected_block)
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
) -> list[str]:
    """
    Encode entries and allowlist with SBERT; return canonical allowlist names
    whose cosine similarity to a candidate meets threshold.
    Expands parentheticals before encoding so each component is matched independently.
    """
    if not entries:
        log.info(f"[{label}] SBERT skipped — no candidate entries.")
        return []
    if not allowlist:
        log.info(f"[{label}] SBERT skipped — allowlist is empty (no {label.lower()} in jobs corpus).")
        return []

    sbert   = get_sbert()
    entries = _expand_parentheticals(entries)

    log.info(f"[{label}] Encoding {len(allowlist)} allowlist items...")
    allow_vecs = sbert.encode(allowlist, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    pending = [e for e in entries if e.lower().strip() not in already_found]
    if not pending:
        return []

    log.info(f"[{label}] Encoding {len(pending)} candidates...")
    entry_vecs = sbert.encode(pending, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    sim_matrix = entry_vecs @ allow_vecs.T   # cosine sim (both L2-normalised)

    discovered: list[str] = []
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

    return discovered


# ---------------------------------------------------------------------------
# Section text -> entries (used when pre-extracted section_text is provided)
# ---------------------------------------------------------------------------

def _section_text_to_entries(section_text: str, allowlist: list[str]) -> list[str]:
    """
    Convert an already-extracted, already-OCR-corrected section text string
    into a flat list of candidate entries for SBERT matching.

    This is the fast path used when skills.py / certs.py pass in their
    rule-based section text directly, bypassing _parse_section_to_entries
    and its re-extraction from raw resume text (which would see garbled OCR).

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
      2. Build OcrCorrector once from the allowlist (reused across both paths).
      3. Parse section into candidate entries.
         - Fast path: section_text is provided (already extracted + OCR-corrected
           by the caller) — use _section_text_to_entries directly.
         - Slow path: re-extract from raw resume_text via _parse_section_to_entries
           (fallback when no pre-extracted text is available).
      4. SBERT pass — semantic matching (cosine >= 0.80).
      5. Fuzzy fallback — if SBERT found nothing.
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
    #    Skipped for native PDFs — their text is already clean.
    #    Also skipped for the section_text fast path since that text is
    #    already corrected upstream.
    corrector = None
    if not is_native_pdf and not section_text and allowlist_list:
        log.info(f"[{config.label}] Building OcrCorrector from {len(allowlist_list)} allowlist entries...")
        corrector = build_ocr_corrector(allowlist_list)
        log.info(f"[{config.label}] OcrCorrector ready.")

    # 3. Parse section into candidate entries
    if section_text is not None:
        # Fast path: caller already extracted + OCR-corrected the section text.
        # If section_text is empty the section simply doesn't exist in this resume
        # (e.g. no cert section) — skip the emergency fallback entirely.
        if section_text:
            log.info(f"[{config.label}] Using pre-extracted section text ({len(section_text)} chars).")
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
            corrector=corrector,
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

    # 4. SBERT pass
    seen:      set[str]  = set()
    canonical: list[str] = []

    for canon in _sbert_match(entries, allowlist_list, seen, label=config.label):
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