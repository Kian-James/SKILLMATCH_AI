"""
matcher.py
==========
Resume Skill & Certification Matcher

Pipeline:
  Stage 1   → Text acquisition   (PyMuPDF native text; EasyOCR fallback)
  Stage 2   → Skill extraction   (Llama-3.1-8B via HF Inference API → fuzzy fallback)
  Stage 2B  → Cert extraction    (Llama-3.1-8B via HF Inference API → regex fallback)
  Stage 3   → Hybrid scoring     (SBERT cosine + skill-match ratio via FAISS)
  Stage 4   → Gap analysis       (missing skills vs. best-matched job)

LLM calls mirror generate_jobs.py exactly:
  - POST https://router.huggingface.co/v1/chat/completions
  - model: meta-llama/Llama-3.1-8B-Instruct:cerebras
  - Authorization: Bearer {HF_TOKEN}
  - No local model loading for the LLM — API only.

Environment variables (loaded from .env by the caller):
    HF_TOKEN          — HuggingFace API token          (required)
    HF_MODEL          — override LLM model string
                        (default: meta-llama/Llama-3.1-8B-Instruct:cerebras)
    EASYOCR_LANGUAGES — comma-separated language codes (default: en)

Requirements:
    pip install requests easyocr pymupdf \
                sentence-transformers faiss-cpu \
                numpy Pillow tqdm python-dotenv rapidfuzz

Changes (v4):
  REMOVED  Hardcoded _OCR_CORRECTIONS dict — static token map replaced entirely.
  ADDED    Embedding-based OCR correction:
             build_ocr_corrector(allowlist) — encodes every allowlist entry with
               SBERT and builds a small FAISS IndexFlatIP index over them.
               Returns an OcrCorrector dataclass (index + canonical labels +
               a per-token LRU cache).
             _ocr_correct_token(token, corrector) — encodes one OCR token, finds
               its nearest allowlist neighbour by cosine similarity, and returns
               the canonical form when the similarity exceeds OCR_EMBED_THRESHOLD.
               Falls back to the original token when no neighbour is close enough,
               so the downstream fuzzy/exact matching still has a shot.
             _apply_ocr_corrections(s, corrector) — replaces the old hardcoded
               version; applies _ocr_correct_token token-by-token, identical
               call-signature for callers that pass corrector=None (no-op).
  UPDATED  discover_skills_dynamically — builds corrector from corpus allowlist
             once per call and passes it through canonicalization helpers.
  UPDATED  _canonicalize / _direct_scan_section — accept optional corrector arg.
  KEPT     All FIX-1 … FIX-5 logic intact; only the correction mechanism changed.
"""

from __future__ import annotations

# ── OpenMP / FAISS segfault guard — must be set before any torch/faiss import ─
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
import re
import string
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import easyocr
import faiss
import fitz          # PyMuPDF
import numpy as np
import requests
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — identical variable names to generate_jobs.py
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_API_URL  = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL    = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
OCR_LANGUAGES      = os.getenv("EASYOCR_LANGUAGES", "en").split(",")
MIN_NATIVE_CHARS   = 50
TOP_K              = 5

# Minimum cosine similarity (0–1) for the embedding-based OCR corrector to
# accept a nearest-neighbour as the "true" form of an OCR-garbled token.
# Tune upward (e.g. 0.82) to be more conservative; downward (e.g. 0.72) to
# be more aggressive.  0.78 is a good starting point for all-MiniLM-L6-v2.
OCR_EMBED_THRESHOLD: float = float(os.getenv("OCR_EMBED_THRESHOLD", "0.78"))

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────────────────────────────────────

_ocr_reader: Optional[easyocr.Reader]      = None
model:       Optional[SentenceTransformer] = None   # public — imported by main.py


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        log.info("[OCR] Loading EasyOCR reader (gpu=False) …")
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def _get_sbert() -> SentenceTransformer:
    global model
    if model is None:
        log.info(f"[SBERT] Loading {EMBEDDING_MODEL} …")
        model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("[SBERT] Ready.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# HF Inference API call — mirrors generate_jobs.py call_llama() exactly
# ─────────────────────────────────────────────────────────────────────────────

def _call_llama(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """
    POST to the HF Inference API exactly as generate_jobs.py does.
    Returns the raw assistant message string.
    Raises RuntimeError on non-200 responses.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,   # deterministic — we want exact extractions
        "stream": False,
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"HF API returned {response.status_code}: {response.text[:300]}"
        )

    return response.json()["choices"][0]["message"]["content"].strip()


def _clean_json_fence(text: str) -> str:
    """Strip markdown code fences the model may emit."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$",        "", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — normalize
# ─────────────────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Strip punctuation/accents, collapse whitespace, lowercase. Public."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-based OCR corrector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OcrCorrector:
    """
    Lightweight wrapper around a FAISS index built from allowlist embeddings.

    Attributes
    ----------
    index      : faiss.IndexFlatIP — L2-normalised embeddings, inner-product
                 search == cosine similarity.
    labels     : list[str] — canonical allowlist strings in index-row order.
    encode_fn  : callable(str) -> np.ndarray(float32, shape=(dim,)) — encodes
                 a single string to a unit-norm vector.
    cache      : dict[str, str] — per-request token → corrected form cache so
                 repeated tokens (e.g. "Microsoft") only hit the index once.
    """
    index:     faiss.IndexFlatIP
    labels:    list[str]
    encode_fn: Callable[[str], np.ndarray]
    cache:     dict[str, str] = field(default_factory=dict)


def build_ocr_corrector(allowlist: list[str]) -> OcrCorrector:
    """
    Build an OcrCorrector from a list of canonical skill/cert strings.

    Each allowlist entry is encoded with SBERT (all-MiniLM-L6-v2), L2-
    normalised, and added to a FAISS IndexFlatIP so that nearest-neighbour
    search returns the most semantically similar canonical form.

    This replaces the old hardcoded _OCR_CORRECTIONS dict entirely — the
    "correction table" is now the full Firestore skills/certs allowlist.

    Parameters
    ----------
    allowlist : list of canonical strings from Firestore (required_skills /
                required_certifications aggregated across all job documents).

    Returns
    -------
    OcrCorrector ready for use in _ocr_correct_token().
    """
    if not allowlist:
        # Return a no-op corrector — index has 0 vectors, lookup always misses.
        sbert = _get_sbert()
        dim   = sbert.get_sentence_embedding_dimension()
        return OcrCorrector(
            index     = faiss.IndexFlatIP(dim),
            labels    = [],
            encode_fn = lambda s: sbert.encode(s, normalize_embeddings=True).astype(np.float32),
        )

    sbert = _get_sbert()
    log.info(f"[OcrCorrector] Encoding {len(allowlist)} allowlist entries …")

    # Encode all allowlist strings in one batch for speed
    vecs = sbert.encode(
        allowlist,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)          # shape: (N, dim)

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)      # belt-and-suspenders — already unit from SBERT
    index.add(vecs)

    log.info(f"[OcrCorrector] Index built ({index.ntotal} entries, dim={dim}).")

    def _encode_one(s: str) -> np.ndarray:
        v = sbert.encode(s, normalize_embeddings=True).astype(np.float32)
        faiss.normalize_L2(v.reshape(1, -1))
        return v

    return OcrCorrector(index=index, labels=list(allowlist), encode_fn=_encode_one)


def _ocr_correct_token(token: str, corrector: OcrCorrector) -> str:
    """
    Return the canonical allowlist form nearest to *token* in embedding space,
    provided the cosine similarity exceeds OCR_EMBED_THRESHOLD.

    Design notes
    ------------
    • Single-token scope: we correct one whitespace-delimited token at a time
      (called from _apply_ocr_corrections).  Multi-word allowlist entries are
      candidates too — if "Microsoft Excel" is closest to the garbled token
      "Microson Exel" split as ["Microson", "Exel"], the per-token correction
      maps "Microson" → "Microsoft" and "Exel" → "Excel" independently, which
      re-assembles correctly downstream when the corrected tokens are rejoined.

    • Cache: the OcrCorrector.cache dict avoids re-encoding the same token
      multiple times within a single request (e.g. "microsoft" appears many
      times in a resume's skills section).

    • Threshold guard: if the best cosine similarity is below
      OCR_EMBED_THRESHOLD the original token is returned unchanged so the
      downstream fuzzy/exact matching still has a chance.

    Parameters
    ----------
    token     : single whitespace-free OCR token (may be garbled, e.g. "Microson").
    corrector : OcrCorrector built from the Firestore allowlist.

    Returns
    -------
    Canonical allowlist string if confident; otherwise the original token.
    """
    if not corrector.labels or not token.strip():
        return token

    # Strip trailing punctuation for the lookup key, but preserve for output
    stripped = token.strip(".,;:()[]")
    cache_key = stripped.lower()

    if cache_key in corrector.cache:
        corrected = corrector.cache[cache_key]
        # Re-attach any trailing punctuation that was stripped
        suffix = token[len(stripped):]
        return corrected + suffix

    vec = corrector.encode_fn(stripped).reshape(1, -1)
    D, I = corrector.index.search(vec, 1)  # top-1 nearest neighbour

    similarity = float(D[0][0])
    if similarity >= OCR_EMBED_THRESHOLD and I[0][0] >= 0:
        canonical = corrector.labels[I[0][0]]

        # For multi-word canonical entries (e.g. "Microsoft Excel") we only
        # want to return the single best-matching word from that entry when
        # correcting a single token, not the whole phrase.  Find the word
        # within the canonical entry whose normalized form is closest to the
        # stripped token's normalized form.
        canon_words = canonical.split()
        if len(canon_words) > 1:
            norm_stripped = normalize(stripped)
            best_word = max(
                canon_words,
                key=lambda w: fuzz.ratio(normalize(w), norm_stripped),
            )
            corrected = best_word
        else:
            corrected = canonical

        log.debug(
            f"[OcrCorrector] '{stripped}' → '{corrected}' "
            f"(sim={similarity:.3f}, nearest='{canonical}')"
        )
    else:
        corrected = stripped  # below threshold — leave unchanged

    corrector.cache[cache_key] = corrected
    suffix = token[len(stripped):]
    return corrected + suffix


def _apply_ocr_corrections(s: str, corrector: Optional[OcrCorrector] = None) -> str:
    """
    Token-level OCR correction using the embedding-based OcrCorrector.

    Splits the input into whitespace-separated tokens, corrects each token
    individually using _ocr_correct_token(), then rejoins.  This allows
    compound strings like "Microson Azuie" to be corrected to
    "Microsoft Azure" token-by-token.

    When corrector is None (e.g. in unit tests or fallback paths) the
    function is a no-op and returns the original string unchanged.

    Parameters
    ----------
    s         : raw OCR string, potentially containing garbled tokens.
    corrector : OcrCorrector built from the Firestore allowlist, or None.

    Returns
    -------
    String with OCR-garbled tokens replaced by their canonical allowlist forms.
    """
    if corrector is None:
        return s
    return " ".join(_ocr_correct_token(tok, corrector) for tok in s.split())


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Text acquisition
# ─────────────────────────────────────────────────────────────────────────────

def _native_pdf_text(file_path: str) -> str:
    doc   = fitz.open(file_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return " ".join(pages).strip()


def _ocr_fallback(file_path: str) -> str:
    reader = _get_ocr_reader()

    if file_path.lower().endswith(".pdf"):
        try:
            from pdf2image import convert_from_path
            with tempfile.TemporaryDirectory() as tmpdir:
                pages = convert_from_path(file_path, dpi=200, output_folder=tmpdir)
                lines: list[str] = []
                for img in tqdm(pages, desc="OCR pages", unit="page", leave=False):
                    lines.extend(
                        reader.readtext(np.array(img), detail=0, paragraph=True)
                    )
                return "\n".join(lines)
        except Exception as exc:
            log.warning(f"[OCR] pdf2image failed ({exc}); direct image fallback …")

    result = reader.readtext(file_path, detail=0, paragraph=True)
    return "\n".join(result)


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(c.isalnum() for c in text) / len(text) * 100, 2)


def extract_text(file_path: str) -> tuple[str, float]:
    """
    Stage 1: Try PyMuPDF native text; fall back to EasyOCR.
    Returns (text, ocr_quality_percent).
    ocr_quality_percent is already × 100 (e.g. 87.4 means 87.4 %).
    """
    native = _native_pdf_text(file_path) if file_path.lower().endswith(".pdf") else ""

    if len(native) >= MIN_NATIVE_CHARS:
        log.info(f"[Stage 1] Native PDF text ({len(native)} chars).")
        return native, _alnum_ratio(native)

    log.info("[Stage 1] Native text too short — EasyOCR fallback …")
    ocr_text = _ocr_fallback(file_path)
    quality  = _alnum_ratio(ocr_text)
    log.info(f"[Stage 1] OCR done ({len(ocr_text)} chars, quality {quality}%).")
    return ocr_text, quality


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Rule-based section locators (debug helpers + internal slicing)
# ─────────────────────────────────────────────────────────────────────────────

_SKILL_HEADING_RE = re.compile(
    r"^(skills?|technical\s+skills?|core\s+competenc\w+|technologies|"
    r"tools?\s+&?\s*technologies?|expertise|proficienc\w+)",
    re.I | re.M,
    )
_CERT_HEADING_RE = re.compile(
    r"^(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?)",
    re.I | re.M,
    )

# General next-section stopper (used by cert slicer).
_NEXT_SECTION_RE = re.compile(
    r"^\s*(experience|education|employment|projects?|awards?|publications?|"
    r"references?|volunteer|languages?|interests?|activities|summary|objective)",
    re.I | re.M,
    )

# Skills-specific stopper: stops at CERTIFICATIONS *and* all other sections.
_SKILLS_END_RE = re.compile(
    r"^\s*(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?|"
    r"experience|education|employment|projects?|awards?|publications?|"
    r"references?|volunteer|languages?|interests?|activities|summary|objective)",
    re.I | re.M,
    )


def _extract_skills_section_line_based(text: str) -> str:
    """
    Return the skills section as a comma-separated string where each entry is
    a complete skill phrase — NOT one token per word.

    "SolarWinds Network Performance Monitor" stays as one entry, not four.

    Extraction strategy
    -------------------
    1. Locate the SKILLS heading; slice the block up to the next section.
    2. Drop lines that look like a section heading (early-exit guard).
    3. Detect the natural delimiter used inside the block and split on it:

       Priority   Delimiter detected when…
       ────────   ─────────────────────────────────────────────────────────────
       1. comma   block contains "," (already comma-separated)
       2. semi    block contains ";"
       3. pipe    block contains "|"
       4. bullet  any line starts with •, -, *, ◦, ▪, or a digit+dot
       5. newline each non-empty line is one skill (most structured resumes)
       6. space   fallback — OCR smashed everything onto one line with no
                  internal delimiter; treat the whole line as one entry so
                  "SolarWinds Network Performance Monitor" is preserved as-is.

    The returned string is the skill entries joined with ", " so the LLM
    prompt and the debug field (skills_section_rule_based) receive:
      "Zendesk, AnyDesk, SolarWinds Network Performance Monitor, Ansible, ..."
    instead of the old per-word explosion:
      "Zendesk, AnyDesk, SolarWinds, Network, Performance, Monitor, Ansible, ..."
    """
    m = _SKILL_HEADING_RE.search(text)
    if not m:
        return ""
    rest = text[m.end():]

    end_m = _SKILLS_END_RE.search(rest)
    block = rest[: end_m.start()].strip() if end_m else rest.strip()

    # ── Drop lines that look like a new named section (early-exit guard) ──────
    raw_lines = block.splitlines()
    kept_lines: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        if (kept_lines
                and len(words) >= 3
                and sum(1 for w in words if w[0].isupper()) >= 2):
            break
        kept_lines.append(stripped)

    if not kept_lines:
        return ""

    joined_block = "\n".join(kept_lines)

    # ── Detect delimiter and split into skill entries ─────────────────
    _BULLET_RE = re.compile(r"^\s*[\u2022\-\*\u25e6\u25aa]|\s*\d+\.\s+", re.M)

    entries: list[str] = []

    if "," in joined_block:
        # Already comma-separated — split on commas within each line
        for line in kept_lines:
            for part in re.split(r"\s*,\s*", line):
                part = part.strip()
                if part:
                    entries.append(part)

    elif ";" in joined_block:
        for line in kept_lines:
            for part in re.split(r"\s*;\s*", line):
                part = part.strip()
                if part:
                    entries.append(part)

    elif "|" in joined_block:
        for line in kept_lines:
            for part in line.split("|"):
                part = part.strip()
                if part:
                    entries.append(part)

    elif _BULLET_RE.search(joined_block):
        # Strip bullet character; each line is one skill phrase
        for line in kept_lines:
            clean = re.sub(r"^\s*[\u2022\-\*\u25e6\u25aa]\s*|\s*\d+\.\s+", "", line).strip()
            if clean:
                entries.append(clean)

    elif len(kept_lines) > 1:
        # Multiple lines, no inline delimiter — each line is one skill phrase
        entries = [line.strip() for line in kept_lines if line.strip()]

    else:
        # Single OCR line, no delimiter — preserve as one entry per line so
        # "SolarWinds Network Performance Monitor" is not split word-by-word.
        # The LLM and direct-scan stages parse individual skills from this.
        entries = [line.strip() for line in kept_lines if line.strip()]

    return ", ".join(entries)


def _extract_cert_section(text: str) -> str:
    """
    Return ONLY the certification tokens — the content on the same line as the
    CERTIFICATIONS heading and any immediately following lines, stopping at the
    next section boundary.  The heading word itself is stripped.
    """
    m = _CERT_HEADING_RE.search(text)
    if not m:
        return ""
    rest  = text[m.end():]
    end_m = _NEXT_SECTION_RE.search(rest)
    block = rest[: end_m.start()].strip() if end_m else rest.strip()
    return block


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — LLM prompts (same structure as generate_jobs.py)
# ─────────────────────────────────────────────────────────────────────────────

_SKILLS_SYSTEM = (
    "You are a precision resume parser. Your task is to identify professional skills. "
    "1. Look for the 'SKILLS' section in the text and extract those items first. "
    "2. Cross-reference with the 'WORK EXPERIENCE' to validate and find additional tools mentioned. "
    "3. IGNORE single-letter artifacts (like 'C', 'R') or two-letter acronyms (like 'AI', 'ML') "
    "UNLESS they are clearly written as distinct skills in the document. "
    "4. Format as a JSON array of strings. Standardize names (e.g., 'Microson' to 'Microsoft'). "
    "Output ONLY the JSON array."
)

# FIX-1: Strengthened cert prompt — explicitly forbid splitting a single cert
# into multiple items.
_CERTS_SYSTEM = (
    "You are a specialized resume parser. Extract COMPLETE professional certification names. "
    "CRITICAL RULE: NEVER split a single certification into multiple array items. "
    "Each certification is ONE complete string "
    "(e.g., 'Microsoft Certified Azure Fundamentals' is ONE item, NOT two). "
    "Each entry must be 10 words or fewer. "
    "Output MUST be a valid JSON array of strings only. "
    "Do not include markdown tags like ```json or any explanation. "
    "Example: [\"AWS Certified Solutions Architect\", \"Google IT Support Professional\"]"
)


def _llm_extract_list(system: str, user: str, label: str) -> list[str]:
    """
    Call Llama via the HF Inference API and parse the returned JSON array.
    Returns [] on any failure so the caller applies a fuzzy fallback.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw   = _call_llama(system, user, max_tokens=512)
            raw   = _clean_json_fence(raw)
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end <= start:
                raise ValueError("No JSON array found in response.")
            items = json.loads(raw[start:end])
            if isinstance(items, list):
                return [str(i).strip() for i in items if str(i).strip()]
        except Exception as exc:
            log.warning(f"[LLM/{label}] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt == MAX_RETRIES:
                log.warning(f"[LLM/{label}] All retries exhausted — using fallback.")
    return []


def _fuzzy_scan(text: str, allowlist: list[str], threshold: int = 88) -> list[str]:
    """Fallback: find allowlist items that appear in text via fuzzy partial match."""
    lower = text.lower()
    return [a for a in allowlist if fuzz.partial_ratio(a.lower(), lower) >= threshold]


# ─────────────────────────────────────────────────────────────────────────────
# FIX-2 + FIX-3: Shared canonicalization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_allow_norm_sorted(allowlist_items: list[str]) -> list[tuple[str, str]]:
    """
    Return (normalized_key, canonical_value) pairs sorted by descending word
    count so that longer allowlist entries are evaluated first (FIX-3).
    """
    pairs = [(normalize(a), a) for a in allowlist_items]
    pairs.sort(key=lambda p: len(p[0].split()), reverse=True)
    return pairs


def _canonicalize(
        raw_item:      str,
        allow_norm:    dict[str, str],
        threshold:     int = 75,
        _sorted_pairs: list[tuple[str, str]] | None = None,
        corrector:     Optional[OcrCorrector] = None,
) -> str:
    """
    Map a raw extracted string to its canonical allowlist form.

    Strategy (in priority order):
      1. Embedding-based OCR correction via _apply_ocr_corrections() with the
         OcrCorrector built from the Firestore allowlist.  This replaces the old
         hardcoded _OCR_CORRECTIONS dict — garbled tokens like "Microson" or
         "Malwarcbytcs" are corrected dynamically by finding their nearest
         neighbour in the allowlist embedding space.
      2. Exact normalized match on the (possibly corrected) form.
      3. Prefix / longest-match-first substring containment (FIX-3):
         raw token ≥ 4 chars AND allowlist key starts with raw token.
      4. max(token_sort_ratio, ratio) fuzzy match (FIX-2).
      5. If best score < threshold, return the raw item unchanged (fails-open).

    Parameters
    ----------
    raw_item      : string extracted by the LLM or direct scan.
    allow_norm    : {normalized_key: canonical_value} dict.
    threshold     : minimum fuzzy score to accept a match.
    _sorted_pairs : pre-sorted (norm_key, canonical) pairs (longest first).
    corrector     : OcrCorrector for embedding-based token correction, or None.
    """
    # 1. Embedding-based OCR correction (replaces hardcoded dict)
    corrected = _apply_ocr_corrections(raw_item, corrector)
    sn = normalize(corrected)

    # 2. Exact match on corrected form
    if sn in allow_norm:
        return allow_norm[sn]

    pairs = _sorted_pairs if _sorted_pairs is not None else list(allow_norm.items())

    best_score: int = 0
    best_val:   str = raw_item   # default: keep as-is

    sn_word_count = len(sn.split())

    for akey, aval in pairs:
        # 3. Prefix substring containment — longest match first (FIX-3).
        #
        #    Guard conditions (ALL must hold to fire):
        #      a. Raw token is ≥ 4 chars (blocks short noise like "ar", "ex").
        #      b. The allowlist key starts with the raw token (prefix match).
        #      c. The raw token is NOT itself a standalone allowlist entry —
        #         if "microservices" is already in the allowlist, we must NOT
        #         promote it to "microservices architecture" just because the
        #         longer entry also starts with it.  Exact match (step 2) above
        #         already handles that case and returns early; we re-check here
        #         as a safety net for the sorted-pairs loop.
        #      d. The allowlist key is at most 2× the word count of the raw
        #         token — prevents a 1-word OCR fragment from being silently
        #         expanded into a 4- or 5-word phrase.
        akey_word_count = len(akey.split())
        prefix_eligible = (
                len(sn) >= 4
                and akey.startswith(sn)
                and sn not in allow_norm                        # guard (c)
                and akey_word_count <= sn_word_count * 2        # guard (d)
        )

        if prefix_eligible:
            score = 90
        else:
            # 4. Dual-metric fuzzy (FIX-2)
            score = max(
                fuzz.token_sort_ratio(sn, akey),
                fuzz.ratio(sn, akey),
            )

        if score > best_score:
            best_score = score
            best_val   = aval
            if score == 90:
                break   # early-exit: longest-prefix hit wins

    return best_val if best_score >= threshold else raw_item


def _direct_scan_section(
        section_text: str,
        sorted_pairs: list[tuple[str, str]],
        threshold:    int = 82,
        corrector:    Optional[OcrCorrector] = None,
        allow_norm:   dict[str, str] | None = None,
) -> list[str]:
    """
    FIX-5 — Direct token-window scan of the raw SKILLS section text.

    Slides a window of 1–N tokens over every line, checks each window
    against every allowlist entry (longest first) using dual-metric fuzzy
    scoring, and collects matches above threshold.

    Embedding-based OCR correction (_apply_ocr_corrections with corrector) is
    applied to each line before tokenising, replacing the old hardcoded
    per-token correction map.

    allow_norm is used to prevent the prefix rule from promoting a window
    that is already a standalone allowlist entry (e.g. "microservices") into
    a longer entry ("microservices architecture") — mirrors the guard added
    to _canonicalize().

    Returns a list of canonical allowlist strings found in the section.
    """
    found: list[str] = []
    seen:  set[str]  = set()
    _allow_norm = allow_norm or {}
    max_win = max((len(akey.split()) for akey, _ in sorted_pairs), default=1)

    for line in section_text.splitlines():
        # Apply embedding-based OCR corrections before building token windows
        corr_line = _apply_ocr_corrections(line, corrector)
        tokens    = normalize(corr_line).split()
        n         = len(tokens)
        for start in range(n):
            for end in range(start + 1, min(start + max_win + 1, n + 1)):
                window = " ".join(tokens[start:end])
                if not window or len(window) < 4:
                    continue
                window_word_count = len(window.split())
                for akey, aval in sorted_pairs:
                    akey_word_count = len(akey.split())
                    prefix_eligible = (
                            len(window) >= 4
                            and akey.startswith(window)
                            and window not in _allow_norm           # already exact — don't expand
                            and akey_word_count <= window_word_count * 2  # no runaway expansion
                    )
                    if prefix_eligible:
                        score = 90
                    else:
                        score = max(
                            fuzz.token_sort_ratio(window, akey),
                            fuzz.ratio(window, akey),
                        )
                    if score >= threshold:
                        canon_n = normalize(aval)
                        if canon_n not in seen:
                            seen.add(canon_n)
                            found.append(aval)
                        break   # longest-first: first hit for this window wins

    return found


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Public extraction API
# ─────────────────────────────────────────────────────────────────────────────

def discover_skills_dynamically(
        resume_text:  str,
        jobs_corpus:  list[dict],
        extra_skills: list[str] | None = None,
) -> list[str]:
    """
    Stage 2 — skill extraction via Llama (HF Inference API).

    1. Locate the skills section with rule-based locator.
    2. Build the corpus allowlist from jobs_corpus + extra_skills.
    3. Build OcrCorrector from the allowlist (embedding-based, Firestore-driven).
    4. POST to HF API with a JSON-array extraction prompt.
    5. Canonicalise LLM output against corpus allowlist using _canonicalize()
       (embedding correction → exact → prefix → dual-metric fuzzy, FIX-2/3).
    6. FIX-5: Direct token-window scan of the raw SKILLS section to recover any
       OCR-garbled tokens the LLM missed — merged with LLM results, deduped.
    7. Fuzzy-scan fallback if both LLM and direct scan produce nothing.

    Returns deduplicated, corpus-canonicalised skill strings.
    """
    # Build corpus allowlist
    allowlist: set[str] = set()
    for job in jobs_corpus:
        allowlist.update(job.get("required_skills", []))
    if extra_skills:
        allowlist.update(extra_skills)
    allowlist_list = sorted(allowlist)
    allow_norm     = {normalize(a): a for a in allowlist_list}

    # Sorted pairs: longest allowlist entry first (FIX-3)
    sorted_pairs = _build_allow_norm_sorted(allowlist_list)

    # Build the embedding-based OcrCorrector from the Firestore allowlist.
    # This is done ONCE per discover_skills_dynamically() call and reused for
    # all _canonicalize() and _direct_scan_section() calls below.
    corrector = build_ocr_corrector(allowlist_list)

    section = _extract_skills_section_line_based(resume_text) or resume_text[:3000]

    user_prompt = (
        f"Extract every technical skill, programming language, framework, tool, "
        f"methodology, and soft skill from this resume section.\n\n"
        f"The skills are listed as comma-separated tokens. Each token or short "
        f"phrase between commas is a distinct skill candidate — treat each one "
        f"individually and standardize any misspelled or OCR-garbled names "
        f"(e.g. 'Microson' -> 'Microsoft', 'Azuie' -> 'Azure').\n\n"
        f"Resume section (comma-separated):\n{section[:2500]}\n\n"
        f"Return ONLY a JSON array of skill name strings. No other text."
    )

    raw = _llm_extract_list(_SKILLS_SYSTEM, user_prompt, "SKILLS")

    # Canonicalise LLM output (embedding correction → exact → prefix → fuzzy).
    # Only keep items that successfully mapped to an allowlist entry.
    seen:      set[str]  = set()
    canonical: list[str] = []

    for skill in raw:
        corrected = _apply_ocr_corrections(skill, corrector)
        canon     = _canonicalize(
            corrected, allow_norm, threshold=75,
            _sorted_pairs=sorted_pairs, corrector=corrector,
        )
        # Drop if canonicalization failed (returned the raw/corrected input unchanged)
        if normalize(canon) == normalize(corrected) and normalize(canon) not in allow_norm:
            log.debug(f"[Skills/LLM] Dropped non-allowlist item: {skill!r}")
            continue
        cn = normalize(canon)
        if cn and cn not in seen:
            seen.add(cn)
            canonical.append(canon)

    # FIX-5: Direct token-window scan — catches OCR tokens LLM missed/dropped
    direct_hits = _direct_scan_section(
        section, sorted_pairs, threshold=82, corrector=corrector, allow_norm=allow_norm
    )
    for hit in direct_hits:
        hn = normalize(hit)
        if hn and hn not in seen:
            seen.add(hn)
            canonical.append(hit)
            log.debug(f"[Skills/DirectScan] Added: {hit}")

    if not canonical:
        log.info("[Skills] LLM + direct scan produced nothing — fuzzy fallback.")
        canonical = _fuzzy_scan(section, allowlist_list)

    log.info(f"[Skills] {len(canonical)} skills discovered.")
    return canonical


def extract_certificates(
        resume_text: str,
        allowlist:   list[str] | None = None,
) -> list[str]:
    """
    Stage 2B — certification extraction via Llama (HF Inference API).

    1. Locate the cert section with rule-based locator.
    2. POST to HF API with a JSON-array extraction prompt (FIX-1: never split).
    3. Enforce ≤ 10-word-per-item guard.
    4. Build OcrCorrector from allowlist (when provided) for embedding-based
       canonicalization; fails-open when allowlist is None.
    5. Regex fallback when LLM returns nothing.

    Returns deduplicated certification name strings.
    """
    section = _extract_cert_section(resume_text) or resume_text[:3000]

    # FIX-1: Explicit no-splitting instruction in user prompt too
    user_prompt = (
        f"Extract every professional certification, license, and credential "
        f"from this resume section. "
        f"IMPORTANT: Each certification is ONE complete string — never split "
        f"'Microsoft Certified Azure Fundamentals' into two separate items. "
        f"Each must be 10 words or fewer.\n\n"
        f"Resume section:\n{section[:2500]}\n\n"
        f"Return ONLY a JSON array of certification name strings. No other text."
    )

    raw = _llm_extract_list(_CERTS_SYSTEM, user_prompt, "CERTS")

    # ≤ 10-word guard
    guarded: list[str] = [
        " ".join(c.split()[:10]) for c in raw if c.strip()
    ]

    # Allowlist filter / dedup using shared _canonicalize() with OcrCorrector
    if allowlist:
        allow_norm   = {normalize(a): a for a in allowlist}
        sorted_pairs = _build_allow_norm_sorted(allowlist)
        # Build corrector from the certs allowlist for embedding-based correction
        corrector    = build_ocr_corrector(allowlist)
        seen:   set[str]  = set()
        result: list[str] = []
        for cert in guarded:
            canon = _canonicalize(
                cert, allow_norm, threshold=75,
                _sorted_pairs=sorted_pairs, corrector=corrector,
            )
            n = normalize(canon)
            if n and n not in seen:
                seen.add(n)
                result.append(canon)
        guarded = result
    else:
        seen_n:  set[str]  = set()
        deduped: list[str] = []
        for c in guarded:
            n = normalize(c)
            if n not in seen_n:
                seen_n.add(n)
                deduped.append(c)
        guarded = deduped

    # Regex fallback
    if not guarded:
        log.info("[Certs] LLM produced nothing — regex fallback.")
        cert_re = re.compile(
            r"(?i)(certif\w+|licen[cs]\w+|aws|gcp|azure|pmp|cpa|cfa|"
            r"ccna|comptia|cisco|oracle|google|microsoft|istqb|six\s+sigma)\b"
        )
        seen_fb: set[str] = set()
        for line in section.splitlines():
            line  = line.strip()
            words = line.split()
            if cert_re.search(line) and 1 <= len(words) <= 10:
                n = normalize(line)
                if n not in seen_fb:
                    seen_fb.add(n)
                    guarded.append(line)

    log.info(f"[Certs] {len(guarded)} certifications discovered.")
    return guarded


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — NLP validation
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_skill_verify(
        discovered_list:   list[str],
        ground_truth_list: list[str],
        threshold:         int = 80,
        resume_text:       str = "",   # kept for API compatibility
) -> dict:
    """
    Compute NLP accuracy of discovered_list against ground_truth_list.
    Matching: exact normalised → rapidfuzz token_sort_ratio ≥ threshold.

    FIX-4: ground_truth_list should be passed as the BEST-matched job's
    required_skills only (not all top-5 skills).

    Returns {"NLP_ACCURACY": "87.50%", "matched_skills": [...], "unmatched_skills": [...]}
    """
    if not ground_truth_list:
        return {
            "NLP_ACCURACY":     "N/A",
            "matched_skills":   [],
            "unmatched_skills": [],
        }

    disc_norms = {normalize(s) for s in discovered_list}
    matched:   list[str] = []
    unmatched: list[str] = []

    for truth in ground_truth_list:
        tn = normalize(truth)
        if tn in disc_norms:
            matched.append(truth)
            continue
        best = max(
            (fuzz.token_sort_ratio(tn, dn) for dn in disc_norms),
            default=0,
        )
        (matched if best >= threshold else unmatched).append(truth)

    pct = len(matched) / len(ground_truth_list) * 100
    return {
        "NLP_ACCURACY":     f"{pct:.2f}%",
        "matched_skills":   matched,
        "unmatched_skills": unmatched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — SBERT + FAISS hybrid scoring
# ─────────────────────────────────────────────────────────────────────────────

def identify_skill_gap(
        resume_skills: list[str],
        job_skills:    list[str],
) -> list[str]:
    """Return skills in job_skills absent from resume_skills."""
    have = {normalize(s) for s in resume_skills}
    return [s for s in job_skills if normalize(s) not in have]


def _encode_jobs(jobs_corpus: list[dict]) -> np.ndarray:
    """
    Encode job documents with SBERT.
    Reuses stored 'embedding' field when present; generates and writes back otherwise.
    """
    sbert   = _get_sbert()
    vectors: list[np.ndarray] = []

    for job in tqdm(jobs_corpus, desc="Encoding jobs", leave=False):
        stored = job.get("embedding") or []
        if isinstance(stored, list) and len(stored) > 0:
            vec = np.array(stored, dtype=np.float32)
        else:
            text = (
                f"{job.get('title','')} {job.get('description','')} "
                f"{' '.join(job.get('required_skills', []))} "
                f"{' '.join(job.get('required_certifications', []))}"
            )
            vec = sbert.encode(text, normalize_embeddings=True).astype(np.float32)
            job["embedding"] = vec.tolist()   # write back for cache_job_embeddings()

        vectors.append(vec)

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix


def calculate_cosine_similarity(
        resume_text:   str,
        resume_skills: list[str],
        jobs_corpus:   list[dict],
        top_k:         int = TOP_K,
) -> list[dict]:
    """
    Stage 3 — hybrid scoring: 0.70 × cosine  +  0.30 × skill_match_ratio.

    NLP validation (fuzzy_skill_verify) is run against every matched job's
    required_skills so each result carries its own accuracy breakdown — not
    just the best match.

    Returns top_k job dicts sorted by score descending:
      {job_id, title, industry, score, cosine_score, match_ratio,
       gap_count, required_skills, matched_skills, missing_skills,
       nlp_validation: {NLP_ACCURACY, matched_skills, unmatched_skills}}
    """
    sbert = _get_sbert()

    q_text = f"{resume_text[:500]} {' '.join(resume_skills)}"
    q_vec  = sbert.encode(q_text, normalize_embeddings=True).astype(np.float32)
    q_vec  = q_vec.reshape(1, -1)
    faiss.normalize_L2(q_vec)

    job_matrix = _encode_jobs(jobs_corpus)
    dim        = job_matrix.shape[1]
    index      = faiss.IndexFlatIP(dim)
    index.add(job_matrix)

    k                 = min(top_k, index.ntotal)
    cos_scores, idxs  = index.search(q_vec, k)
    resume_norms      = {normalize(s) for s in resume_skills}
    results: list[dict] = []

    for cosine_score, idx in zip(cos_scores[0], idxs[0]):
        if idx < 0:
            continue
        job     = jobs_corpus[idx]
        req     = job.get("required_skills", [])
        matched = [s for s in req if normalize(s) in resume_norms]
        missing = identify_skill_gap(resume_skills, req)

        match_ratio = len(matched) / len(req) if req else 0.0
        hybrid      = round(0.70 * float(cosine_score) + 0.30 * match_ratio, 4)

        # Per-job NLP validation: how many of THIS job's required skills did
        # the resume's discovered skill list cover (exact + fuzzy threshold)?
        nlp_validation = fuzzy_skill_verify(
            discovered_list   = resume_skills,
            ground_truth_list = req,
        )

        results.append({
            "job_id":          job.get("id", job.get("_id", "")),
            "title":           job.get("title", ""),
            "industry":        job.get("industry", ""),
            "score":           hybrid,
            "cosine_score":    round(float(cosine_score), 4),
            "match_ratio":     round(match_ratio, 4),
            "gap_count":       len(missing),
            "required_skills": req,
            "matched_skills":  matched,
            "missing_skills":  missing,
            "nlp_validation":  nlp_validation,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    log.info(
        f"[Stage 3] {len(results)} jobs scored; "
        f"top = {results[0]['score'] if results else 'N/A'}."
    )
    return results