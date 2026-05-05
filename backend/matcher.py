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
  KEPT     All FIX-1 ... FIX-5 logic intact; only the correction mechanism changed.

Changes (v5):
  ADDED    _split_ocr_blob(line, allowlist) — greedy longest-match segmenter
             for the single-line / no-delimiter OCR fallback path.

             Problem: OCR collapses the SKILLS section onto one line with no
             commas, bullets, or other delimiters, e.g.:
               "Zendosk AnyDesk SolarWinds Network Periormanice Vonitor ..."
             The old else branch returned that entire blob as ONE unsplit entry,
             so the LLM received an undelimited wall of garbled text.

             Fix: _split_ocr_blob slides a greedy token window (longest
             allowlist entry first) over the blob and emits each matched phrase
             as a separate comma-separated entry. Fuzzy matching (threshold 82)
             handles garbled OCR tokens — "Malwarcbytcs" matches "Malwarebytes",
             "Microson" matches "Microsoft", etc. Unmatched tokens fall back to
             raw single-token entries so nothing is silently dropped.

             Result for the test resume:
               BEFORE: "Zendosk AnyDesk SolarWinds Network Periormanice ..."
                        (one big unsplit blob)
               AFTER:  "Zendesk, AnyDesk, SolarWinds Network Performance
                        Monitor, Malwarebytes, macOS, Ansible, Microsoft
                        Excel, Microsoft Azure"
                        (properly comma-separated skill phrases)

  UPDATED  _extract_skills_section_line_based(text, allowlist=None):
             - Added allowlist parameter (default None — fully backward-compat).
             - The single-line else branch now calls _split_ocr_blob(blob,
               allowlist) when allowlist is provided.
             - All other delimiter paths (comma, semicolon, pipe, bullet,
               multi-line newline) are unchanged.

  UPDATED  discover_skills_dynamically — passes allowlist_list to
             _extract_skills_section_line_based so blob splitting is active
             for every real pipeline run. The debug call in main.py passes
             no allowlist (None) and keeps old behaviour for the debug field.

  ADDED    BLOB_FUZZY_THRESHOLD env var (default 82) — tune via .env without
             touching source code.

Changes (v6):
  FIXED    Native PDF early-exit guard truncation.

             Problem: When text_source is "native_pdf", the skills section is
             properly multi-line (each skill on its own line, separated by \\n).
             However the early-exit guard inside _extract_skills_section_line_based
             was firing prematurely on lines like "CRUD Application Development"
             or "Flutter & Firebase Development" — both have >=3 words with >=2
             capitalised, which the guard treats as a new section heading, so it
             stops after the very first skill line.  The _SKILLS_END_RE boundary
             regex already correctly stops at "PROJECTS" / "CERTIFICATIONS", so
             the guard is redundant and harmful for native PDF text.

             Fix: added is_native_pdf: bool = False parameter to
               _extract_skills_section_line_based and discover_skills_dynamically.
               When is_native_pdf=True the early-exit guard is skipped entirely;
               section bounding is left entirely to _SKILLS_END_RE (which already
               handles PROJECTS, CERTIFICATIONS, EXPERIENCE, etc.).
               When is_native_pdf=False (OCR path) the guard is kept as-is.

             extract_text() now returns a third element — a boolean
               text_is_native — so callers can pass it straight through without
               re-deriving it.  Return signature changes from:
                 (str, float)  →  (str, float, bool)
               Existing callers that unpack only two values are unaffected if
               they use  text, quality = extract_text(...)  but should be
               updated to  text, quality, is_native = extract_text(...).
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

# Minimum cosine similarity (0-1) for the embedding-based OCR corrector to
# accept a nearest-neighbour as the "true" form of an OCR-garbled token.
# Tune upward (e.g. 0.82) to be more conservative; downward (e.g. 0.72) to
# be more aggressive.  0.78 is a good starting point for all-MiniLM-L6-v2.
OCR_EMBED_THRESHOLD: float = float(os.getenv("OCR_EMBED_THRESHOLD", "0.78"))

# Minimum fuzzy score (0-100) for _split_ocr_blob to accept a token window
# as matching an allowlist entry.  82 balances recall vs. false positives for
# OCR-garbled skill names.  Lower = more matches (risk of false positives).
# Higher = fewer matches (risk of dropping real skills).
BLOB_FUZZY_THRESHOLD: int = int(os.getenv("BLOB_FUZZY_THRESHOLD", "82"))

# Module-level allowlist cache — populated by discover_skills_dynamically()
# so that the debug call in main.py (_extract_skills_section_line_based with
# no allowlist argument) also produces comma-separated output without needing
# main.py to be changed.
_cached_allowlist: list = []

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────────────────────────────────────

_ocr_reader: Optional[easyocr.Reader]      = None
model:       Optional[SentenceTransformer] = None   # public — imported by main.py


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        log.info("[OCR] Loading EasyOCR reader (gpu=False) ...")
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def _get_sbert() -> SentenceTransformer:
    global model
    if model is None:
        log.info(f"[SBERT] Loading {EMBEDDING_MODEL} ...")
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
    cache      : dict[str, str] — per-request token -> corrected form cache so
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

    Parameters
    ----------
    allowlist : list of canonical strings from Firestore (required_skills /
                required_certifications aggregated across all job documents).

    Returns
    -------
    OcrCorrector ready for use in _ocr_correct_token().
    """
    if not allowlist:
        sbert = _get_sbert()
        dim   = sbert.get_sentence_embedding_dimension()
        return OcrCorrector(
            index     = faiss.IndexFlatIP(dim),
            labels    = [],
            encode_fn = lambda s: sbert.encode(s, normalize_embeddings=True).astype(np.float32),
        )

    sbert = _get_sbert()
    log.info(f"[OcrCorrector] Encoding {len(allowlist)} allowlist entries ...")

    vecs = sbert.encode(
        allowlist,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
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
    Falls back to the original token when below threshold.
    """
    if not corrector.labels or not token.strip():
        return token

    stripped  = token.strip(".,;:()[]")
    cache_key = stripped.lower()

    if cache_key in corrector.cache:
        corrected = corrector.cache[cache_key]
        suffix    = token[len(stripped):]
        return corrected + suffix

    vec = corrector.encode_fn(stripped).reshape(1, -1)
    D, I = corrector.index.search(vec, 1)

    similarity = float(D[0][0])
    if similarity >= OCR_EMBED_THRESHOLD and I[0][0] >= 0:
        canonical   = corrector.labels[I[0][0]]
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
            f"[OcrCorrector] '{stripped}' -> '{corrected}' "
            f"(sim={similarity:.3f}, nearest='{canonical}')"
        )
    else:
        corrected = stripped

    corrector.cache[cache_key] = corrected
    suffix = token[len(stripped):]
    return corrected + suffix


def _apply_ocr_corrections(s: str, corrector: Optional[OcrCorrector] = None) -> str:
    """
    Token-level OCR correction using the embedding-based OcrCorrector.
    When corrector is None the function is a no-op.
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
            log.warning(f"[OCR] pdf2image failed ({exc}); direct image fallback ...")

    result = reader.readtext(file_path, detail=0, paragraph=True)
    return "\n".join(result)


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(c.isalnum() for c in text) / len(text) * 100, 2)


def extract_text(file_path: str) -> tuple[str, float, bool]:
    """
    Stage 1: Try PyMuPDF native text; fall back to EasyOCR.

    Returns
    -------
    (text, ocr_quality_percent, is_native_pdf)

    is_native_pdf is True when the native PyMuPDF path was used, False when
    EasyOCR was used.  Callers should pass this flag to
    discover_skills_dynamically so that the early-exit guard is skipped for
    native PDF text (v6 fix).

    Note: the return signature was extended from (str, float) to
    (str, float, bool) in v6.  Existing two-value unpacking
      text, quality = extract_text(...)
    still works in Python (the third value is silently ignored when unpacking
    into exactly two names only if you use extended unpacking like
      text, quality, *_ = extract_text(...)
    — update call sites to capture all three values for full functionality).
    """
    native = _native_pdf_text(file_path) if file_path.lower().endswith(".pdf") else ""

    if len(native) >= MIN_NATIVE_CHARS:
        log.info(f"[Stage 1] Native PDF text ({len(native)} chars).")
        return native, _alnum_ratio(native), True

    log.info("[Stage 1] Native text too short — EasyOCR fallback ...")
    ocr_text = _ocr_fallback(file_path)
    quality  = _alnum_ratio(ocr_text)
    log.info(f"[Stage 1] OCR done ({len(ocr_text)} chars, quality {quality}%).")
    return ocr_text, quality, False


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Rule-based section locators
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
_NEXT_SECTION_RE = re.compile(
    r"^\s*(experience|education|employment|projects?|awards?|publications?|"
    r"references?|volunteer|languages?|interests?|activities|summary|objective)",
    re.I | re.M,
    )
_SKILLS_END_RE = re.compile(
    r"^\s*(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?|"
    r"experience|education|employment|projects?|awards?|publications?|"
    r"references?|volunteer|languages?|interests?|activities|summary|objective)",
    re.I | re.M,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — OCR blob segmenter (v5)
# ─────────────────────────────────────────────────────────────────────────────

def _split_ocr_blob(line: str, allowlist: list[str]) -> list[str]:
    """
    Segment a single space-separated OCR blob into individual skill phrases
    using greedy longest-match fuzzy scanning against the allowlist.

    Why this is needed
    ------------------
    EasyOCR often collapses an entire SKILLS section onto one line with no
    commas, bullets, or other delimiters:

        "Zendosk AnyDesk SolarWinds Network Periormanice Vonitor Malwarcbytcs
         macCs Ansible Microson Excel Microson Azuie"

    The existing delimiter-detection logic falls through to the else branch and
    returns that whole string as a SINGLE entry, so the LLM receives an
    undelimited wall of garbled text with no way to know where one skill ends
    and the next begins.

    Algorithm
    ---------
    1. Sort allowlist by descending word count so multi-word entries like
       "SolarWinds Network Performance Monitor" (4 tokens) are tried before
       single-word entries like "SolarWinds" (1 token) — greedy longest match.

    2. Pre-normalise the sorted allowlist once (normalize() is idempotent but
       called O(n * max_win) times in the inner loop, so pre-computing saves
       significant work).

    3. Tokenise the input line on whitespace.

    4. At each cursor position i, slide a window from the longest possible
       length down to 1:
         a. Join the window tokens into a candidate string and normalize it.
         b. Score against every allowlist entry using
            max(token_sort_ratio, ratio) — same dual-metric as the rest of
            the pipeline so scoring behaviour is consistent.
         c. On a hit (score >= BLOB_FUZZY_THRESHOLD):
              - Emit the CANONICAL allowlist form (corrected spelling).
              - Advance the cursor by win_len (consume all matched tokens).
              - Break out of both loops (greedy: commit to this match).
         d. On a total miss (no window length produced a hit):
              - Emit the raw single token stripped of trailing punctuation.
              - Advance cursor by 1.
              - Nothing is silently dropped.

    Result for the test resume:
        BEFORE (one unsplit entry passed to LLM):
          "Zendosk AnyDesk SolarWinds Network Periormanice Vonitor ..."
        AFTER (comma-joined list of skill phrases):
          "Zendesk, AnyDesk, SolarWinds Network Performance Monitor,
           Malwarebytes, macOS, Ansible, Microsoft Excel, Microsoft Azure"

    Parameters
    ----------
    line      : single OCR text line — the entire collapsed SKILLS blob.
    allowlist : canonical skill strings from the Firestore corpus.

    Returns
    -------
    List of skill phrase strings (canonical where matched, raw token otherwise).
    Joined by the caller with ", ".
    """
    if not allowlist or not line.strip():
        return [line.strip()] if line.strip() else []

    # Sort: longest entry first for greedy longest-match behaviour
    sorted_allow = sorted(allowlist, key=lambda a: len(a.split()), reverse=True)

    # Pre-normalise the allowlist once — avoids repeated normalize() calls
    # inside the O(n * max_win * |allowlist|) inner loop
    norm_allow = [(normalize(a), a) for a in sorted_allow]
    max_win    = max(len(a.split()) for a in sorted_allow) if sorted_allow else 1

    tokens  = line.split()
    n       = len(tokens)
    result: list[str] = []
    i = 0

    while i < n:
        best_canon: str | None = None
        best_len:   int        = 0

        # Try window lengths from longest down to 1 (greedy)
        for win_len in range(min(max_win, n - i), 0, -1):
            window      = " ".join(tokens[i : i + win_len])
            window_norm = normalize(window)

            if len(window_norm) < 3:
                # Too short to be a meaningful skill token — skip
                continue

            # Length-adaptive threshold:
            # Short OCR tokens (<=5 chars, e.g. "macCs" -> "macOS") naturally
            # score lower with fuzzy string metrics because small edit distances
            # have a larger proportional impact on short strings.  We lower the
            # acceptance bar for short windows so they still match, while keeping
            # the full threshold for longer phrases where a lower score more
            # likely indicates a genuine mismatch.
            #
            #   window length   threshold used
            #   -------------   -------------------------
            #   <= 5 chars      BLOB_FUZZY_THRESHOLD - 10  (default 72)
            #   6-9 chars       BLOB_FUZZY_THRESHOLD - 5   (default 77)
            #   >= 10 chars     BLOB_FUZZY_THRESHOLD       (default 82)
            wlen = len(window_norm)
            if wlen <= 5:
                effective_threshold = BLOB_FUZZY_THRESHOLD - 10
            elif wlen <= 9:
                effective_threshold = BLOB_FUZZY_THRESHOLD - 5
            else:
                effective_threshold = BLOB_FUZZY_THRESHOLD

            for akey, aval in norm_allow:
                score = max(
                    fuzz.token_sort_ratio(window_norm, akey),
                    fuzz.ratio(window_norm, akey),
                )
                if score >= effective_threshold:
                    best_canon = aval
                    best_len   = win_len
                    break       # first (longest) hit for this window size wins

            if best_canon:
                break           # stop trying shorter windows once we have a match

        if best_canon:
            result.append(best_canon)
            i += best_len
            log.debug(f"[BlobSplit] consumed {best_len} token(s) -> '{best_canon}'")
        else:
            # No allowlist match — emit raw token so nothing is silently dropped
            raw_tok = tokens[i].strip(".,;:()[]")
            if raw_tok:
                result.append(raw_tok)
            i += 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Skills section extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_skills_section_line_based(
        text:          str,
        allowlist:     list[str] | None = None,
        is_native_pdf: bool = False,
) -> str:
    """
    Return the skills section as a comma-separated string where each entry is
    a complete skill phrase — NOT one token per word.

    "SolarWinds Network Performance Monitor" stays as one entry, not four.

    Extraction strategy
    -------------------
    1. Locate the SKILLS heading; slice the block up to the next section.
    2. Drop lines that look like a section heading (early-exit guard).
       ** SKIPPED when is_native_pdf=True ** — native PDF text already has
       clean section boundaries enforced by _SKILLS_END_RE, so skill lines
       like "Flutter & Firebase Development" or "CRUD Application Development"
       must NOT be mistaken for new section headings and dropped.
    3. Detect the natural delimiter used inside the block and split on it:

       Priority   Delimiter detected when...
       --------   -----------------------------------------------------------
       1. comma   block contains ","
       2. semi    block contains ";"
       3. pipe    block contains "|"
       4. bullet  any line starts with bullet char or digit+dot
       5. newline multiple non-empty lines with no inline delimiter
       6. blob    single line, no delimiter — _split_ocr_blob() segments it
                  using greedy fuzzy window matching against the allowlist.
                  When allowlist=None (debug call from main.py) the whole
                  line is returned as-is (backward-compatible).

    Returns a comma-separated string so the LLM prompt and debug field get:
      "Zendesk, AnyDesk, SolarWinds Network Performance Monitor, Ansible, ..."
    instead of a single unsplit blob or a per-word explosion.

    Parameters
    ----------
    text          : full resume text.
    allowlist     : canonical skill strings passed to _split_ocr_blob() for
                    the blob path. None = skip blob segmentation (old behaviour).
    is_native_pdf : when True, skip the early-exit guard so that multi-word
                    skill lines are not mistaken for section headings.
                    Set to True whenever extract_text() returned is_native=True.
    """
    m = _SKILL_HEADING_RE.search(text)
    if not m:
        return ""
    rest = text[m.end():]

    end_m = _SKILLS_END_RE.search(rest)
    block = rest[: end_m.start()].strip() if end_m else rest.strip()

    raw_lines = block.splitlines()

    # ── Early-exit guard — OCR path only ─────────────────────────────────────
    #
    # For OCR text the section boundaries are less reliable, so we apply a
    # heuristic: if a line has >=3 words with >=2 capitalised it probably marks
    # the start of a new resume section (e.g. "Work Experience Summary").
    # We stop collecting skill lines at that point.
    #
    # For native PDF text this guard is SKIPPED.  _SKILLS_END_RE already
    # correctly bounds the block at "PROJECTS", "CERTIFICATIONS", etc., so
    # legitimate skill lines like "Flutter & Firebase Development" (3 words,
    # 3 capitalised) must pass through unchanged.
    if is_native_pdf:
        kept_lines = [line.strip() for line in raw_lines if line.strip()]
    else:
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

    # ── Detect delimiter and split into skill entries ──────────────────────────
    _BULLET_RE = re.compile(r"^\s*[\u2022\-\*\u25e6\u25aa]|\s*\d+\.\s+", re.M)
    entries: list[str] = []

    if "," in joined_block:
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
        for line in kept_lines:
            clean = re.sub(
                r"^\s*[\u2022\-\*\u25e6\u25aa]\s*|\s*\d+\.\s+", "", line
            ).strip()
            if clean:
                entries.append(clean)

    elif len(kept_lines) > 1:
        # Multiple lines, no inline delimiter — each line is one skill phrase
        entries = [line.strip() for line in kept_lines if line.strip()]

    else:
        # ── Single-line OCR blob — no delimiter detected at all ───────────────
        #
        # EasyOCR collapsed the SKILLS section onto one space-separated line:
        #   "Zendosk AnyDesk SolarWinds Network Periormanice Vonitor ..."
        #
        # Without allowlist (e.g. the debug call in main.py):
        #   Return the blob as-is — one entry — same as the old behaviour so
        #   nothing downstream breaks.
        #
        # With allowlist (the live pipeline call in discover_skills_dynamically):
        #   _split_ocr_blob() segments the blob by sliding a greedy fuzzy
        #   window over it and matching against the allowlist, emitting one
        #   entry per skill phrase.  Multi-word phrases like "SolarWinds
        #   Network Performance Monitor" are preserved intact; garbled tokens
        #   like "Malwarcbytcs" are corrected to "Malwarebytes".
        blob = kept_lines[0] if kept_lines else ""
        if blob:
            # Use explicitly-passed allowlist first; fall back to the module-
            # level cache populated by discover_skills_dynamically() so that
            # the debug call in main.py (no allowlist argument) also produces
            # a comma-separated result rather than one unsplit blob.
            effective_allowlist = allowlist if allowlist is not None else _cached_allowlist
            entries = _split_ocr_blob(blob, effective_allowlist) if effective_allowlist else [blob]

    return ", ".join(entries)


def _extract_cert_section(text: str) -> str:
    """
    Return ONLY the certification tokens — content after the CERTIFICATIONS
    heading up to the next section boundary.
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
      1. Embedding-based OCR correction via _apply_ocr_corrections().
      2. Exact normalized match on the corrected form.
      3. Prefix / longest-match-first substring containment (FIX-3).
      4. max(token_sort_ratio, ratio) fuzzy match (FIX-2).
      5. If best score < threshold, return the raw item unchanged (fails-open).
    """
    corrected = _apply_ocr_corrections(raw_item, corrector)
    sn = normalize(corrected)

    if sn in allow_norm:
        return allow_norm[sn]

    pairs = _sorted_pairs if _sorted_pairs is not None else list(allow_norm.items())

    best_score: int = 0
    best_val:   str = raw_item
    sn_word_count   = len(sn.split())

    for akey, aval in pairs:
        akey_word_count = len(akey.split())
        prefix_eligible = (
                len(sn) >= 4
                and akey.startswith(sn)
                and sn not in allow_norm
                and akey_word_count <= sn_word_count * 2
        )

        if prefix_eligible:
            score = 90
        else:
            score = max(
                fuzz.token_sort_ratio(sn, akey),
                fuzz.ratio(sn, akey),
            )

        if score > best_score:
            best_score = score
            best_val   = aval
            if score == 90:
                break

    return best_val if best_score >= threshold else raw_item


def _direct_scan_section(
        section_text: str,
        sorted_pairs: list[tuple[str, str]],
        threshold:    int = 82,
        corrector:    Optional[OcrCorrector] = None,
        allow_norm:   dict[str, str] | None = None,
) -> list[str]:
    """
    FIX-6 — Atomic entry scan of the skills section.

    Previous behaviour (token-window sliding):
      Slid a 1-N token window over every line, scoring each sub-phrase against
      the allowlist. This caused entries like "SolarWinds Network Performance
      Monitor" to also match "Performance", "Monitoring", and "Azure" as
      independent hits — fragment hallucinations from words that happened to
      score high in isolation.

    New behaviour (comma-entry matching):
      When the section text is already comma-separated (as produced by
      _extract_skills_section_line_based), split on commas and treat each
      entry as an atomic unit. Each entry is OCR-corrected, normalized, then
      matched against the allowlist in full — no sub-phrase windowing within
      a single entry. This prevents "SolarWinds Network Performance Monitor"
      from leaking "Performance" or "Monitor" as separate hits.

    Fallback (non-comma text):
      If the section has no commas (raw resume text passed directly), fall
      back to per-line matching where each full line is treated as one unit.
    """
    found: list[str] = []
    seen:  set[str]  = set()
    _allow_norm = allow_norm or {}

    # Decide whether to split on commas (processed section) or lines (raw text)
    if "," in section_text:
        raw_entries = [e.strip() for e in section_text.split(",") if e.strip()]
    else:
        raw_entries = [l.strip() for l in section_text.splitlines() if l.strip()]

    for entry in raw_entries:
        corrected  = _apply_ocr_corrections(entry, corrector)
        entry_norm = normalize(corrected)

        if not entry_norm or len(entry_norm) < 3:
            continue

        # 1. Exact normalized lookup
        if entry_norm in _allow_norm:
            canon = _allow_norm[entry_norm]
            cn = normalize(canon)
            if cn not in seen:
                seen.add(cn)
                found.append(canon)
                log.debug(f"[DirectScan] Exact match: '{entry}' -> '{canon}'")
            continue

        # 2. Fuzzy match against full allowlist entries (longest first)
        best_score: int = 0
        best_val:   str = ""
        for akey, aval in sorted_pairs:
            score = max(
                fuzz.token_sort_ratio(entry_norm, akey),
                fuzz.ratio(entry_norm, akey),
            )
            if score > best_score:
                best_score = score
                best_val   = aval

        if best_score >= threshold and best_val:
            cn = normalize(best_val)
            if cn not in seen:
                seen.add(cn)
                found.append(best_val)
                log.debug(f"[DirectScan] Fuzzy match: '{entry}' -> '{best_val}' (score={best_score})")
        else:
            log.debug(f"[DirectScan] No match for entry: '{entry}' (best_score={best_score})")

    return found


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Public extraction API
# ─────────────────────────────────────────────────────────────────────────────

def discover_skills_dynamically(
        resume_text:   str,
        jobs_corpus:   list[dict],
        extra_skills:  list[str] | None = None,
        is_native_pdf: bool = False,
) -> list[str]:
    """
    Stage 2 — skill extraction via Llama (HF Inference API).

    1. Locate the skills section with rule-based locator.
       Passes allowlist_list to _extract_skills_section_line_based so that
       single-line OCR blobs are segmented by _split_ocr_blob before the
       LLM receives them (v5 fix).
       Passes is_native_pdf so the early-exit guard is skipped for native
       PDF text where skill lines look like section headings (v6 fix).
    2. Build the corpus allowlist from jobs_corpus + extra_skills.
    3. Build OcrCorrector from the allowlist (embedding-based, Firestore-driven).
    4. POST to HF API with a JSON-array extraction prompt.
    5. Canonicalise LLM output against corpus allowlist using _canonicalize().
    6. FIX-5: Direct token-window scan to recover any OCR-garbled tokens
       the LLM missed — merged with LLM results, deduped.
    7. Fuzzy-scan fallback if both LLM and direct scan produce nothing.

    Parameters
    ----------
    resume_text   : full resume text from extract_text().
    jobs_corpus   : list of job dicts from Firestore / generate_jobs.py.
    extra_skills  : additional canonical skill strings to include in the
                    allowlist (optional).
    is_native_pdf : pass the third return value of extract_text() here so
                    the early-exit guard is skipped for native PDF resumes.

    Returns deduplicated, corpus-canonicalised skill strings.
    """
    allowlist: set[str] = set()
    for job in jobs_corpus:
        allowlist.update(job.get("required_skills", []))
    if extra_skills:
        allowlist.update(extra_skills)
    allowlist_list = sorted(allowlist)
    allow_norm     = {normalize(a): a for a in allowlist_list}

    sorted_pairs = _build_allow_norm_sorted(allowlist_list)
    corrector    = build_ocr_corrector(allowlist_list)

    # Cache the allowlist at module level so the debug call in main.py can
    # also produce comma-separated skills_section_rule_based output.
    global _cached_allowlist
    _cached_allowlist = allowlist_list

    # Pass allowlist_list so _split_ocr_blob fires for single-line OCR blobs.
    # Pass is_native_pdf so the early-exit guard is bypassed for native PDF text.
    section = (
            _extract_skills_section_line_based(
                resume_text,
                allowlist=allowlist_list,
                is_native_pdf=is_native_pdf,
            )
            or resume_text[:3000]
    )

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

    seen:      set[str]  = set()
    canonical: list[str] = []

    for skill in raw:
        corrected = _apply_ocr_corrections(skill, corrector)
        canon     = _canonicalize(
            corrected, allow_norm, threshold=75,
            _sorted_pairs=sorted_pairs, corrector=corrector,
        )
        if normalize(canon) == normalize(corrected) and normalize(canon) not in allow_norm:
            log.debug(f"[Skills/LLM] Dropped non-allowlist item: {skill!r}")
            continue
        cn = normalize(canon)
        if cn and cn not in seen:
            seen.add(cn)
            canonical.append(canon)

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
    3. Enforce <= 10-word-per-item guard.
    4. Build OcrCorrector from allowlist (when provided) for embedding-based
       canonicalization; fails-open when allowlist is None.
    5. Regex fallback when LLM returns nothing.

    Returns deduplicated certification name strings.
    """
    section = _extract_cert_section(resume_text) or resume_text[:3000]

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

    guarded: list[str] = [
        " ".join(c.split()[:10]) for c in raw if c.strip()
    ]

    if allowlist:
        allow_norm   = {normalize(a): a for a in allowlist}
        sorted_pairs = _build_allow_norm_sorted(allowlist)
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

def _sliding_window_match(
        truth_norm:       str,
        disc_norms:       set,
        consumed_phrases: set,
        threshold:        int = 80,
) -> tuple:
    """
    Match one ground-truth skill against the discovered skill set using a
    greedy expanding token window. Consumed phrases are tracked at the PHRASE
    level (not token level) so that "Microsoft Excel" and "Microsoft Azure"
    can both match without blocking each other, while a single phrase like
    "SolarWinds Network Performance Monitor" is consumed as a unit and its
    tokens cannot be credited to a second ground-truth skill via that phrase.

    Algorithm
    ---------
    available = disc_norms minus consumed_phrases (phrase-level exclusion)

    1. Full exact lookup on truth_norm in available  ->  commit phrase, True.

    2. Expand window left-to-right over truth tokens:
         window_1 = t0
         window_2 = t0 t1
         ...
         window_n = t0 ... tn  (full phrase — same as step 1 but via loop)

       For each window:
         a. Exact lookup in available -> commit that phrase, True.
         b. Best fuzzy score across available phrases.
            If score >= threshold -> commit best phrase, True.
         c. If score drops vs the previous window length -> STOP expanding.
            (We have passed the natural boundary of this skill.)

    3. No window matched -> False (unmatched).

    Why phrase-level consumption?
    ------------------------------
    Token-level consumption ("microsoft" consumed by "Microsoft Excel" blocks
    "Microsoft Azure") is too aggressive for multi-word skills that share a
    common word. Phrase-level consumption removes the ENTIRE matched phrase
    from the available pool, so the word "Microsoft" can still appear in a
    second unrelated phrase. The non-reuse guarantee applies only to the
    specific discovered phrase that was committed, not to its constituent words.

    Non-reuse example
    -----------------
    discovered = {"solarwinds network performance monitor", "ansible"}
    Round 1 (truth = "solarwinds network performance monitor"):
      -> exact match, committed -> consumed = {"solarwinds network performance monitor"}
    Round 2 (truth = "network monitoring"):
      -> "solarwinds network performance monitor" is excluded (consumed)
      -> no remaining phrase matches "network monitoring" -> unmatched (correct)

    Parameters
    ----------
    truth_norm        : normalized ground-truth skill string.
    disc_norms        : set of normalized discovered skill strings.
    consumed_phrases  : mutable set of already-committed phrase strings;
                        updated in-place on a successful match.
    threshold         : minimum fuzzy score to accept as a match.

    Returns
    -------
    (matched: bool, consumed_phrases: set)  — consumed_phrases mutated in place.
    """
    # Available = everything not yet consumed as a complete phrase
    available = disc_norms - consumed_phrases

    # 1. Full exact match (fastest path, also handles single-word skills)
    if truth_norm in available:
        consumed_phrases.add(truth_norm)
        return True, consumed_phrases

    truth_tokens = truth_norm.split()
    prev_best: float = 0.0

    # 2. Expanding window
    for end in range(1, len(truth_tokens) + 1):
        window = " ".join(truth_tokens[:end])

        # 2a. Exact lookup on the window itself
        if window in available:
            consumed_phrases.add(window)
            return True, consumed_phrases

        # 2b. Best fuzzy score across all still-available phrases
        best_score: float = 0.0
        best_dn:    str   = ""
        for dn in available:
            score = max(
                fuzz.token_sort_ratio(window, dn),
                fuzz.ratio(window, dn),
            )
            if score > best_score:
                best_score = score
                best_dn    = dn

        if best_score >= threshold and best_dn:
            consumed_phrases.add(best_dn)
            return True, consumed_phrases

        # 2c. Score regressed — we have passed the skill boundary, stop
        if end > 1 and best_score < prev_best:
            break

        prev_best = best_score

    return False, consumed_phrases


def fuzzy_skill_verify(
        discovered_list:   list,
        ground_truth_list: list,
        threshold:         int = 80,
        resume_text:       str = "",   # kept for API compatibility
) -> dict:
    """
    Compute NLP accuracy using greedy sliding-window token matching with
    phrase-level consumption to prevent double-counting.

    For each ground-truth skill:
      1. Expand a token window left-to-right over the truth tokens.
      2. Commit to the LONGEST fuzzy match found (greedy).
      3. Mark the matched discovered phrase as consumed so it cannot be
         credited to a second ground-truth skill.
      4. Stop expanding when the fuzzy score stops improving.

    Phrase-level consumption means "Microsoft Excel" and "Microsoft Azure"
    can both match without blocking each other, while "SolarWinds Network
    Performance Monitor" consumed as a whole prevents "Network Monitoring"
    from matching against the same phrase a second time.

    Returns
    -------
    {"NLP_ACCURACY": "87.50%", "matched_skills": [...], "unmatched_skills": [...]}
    """
    if not ground_truth_list:
        return {
            "NLP_ACCURACY":     "N/A",
            "matched_skills":   [],
            "unmatched_skills": [],
        }

    disc_norms:       set  = {normalize(s) for s in discovered_list}
    consumed_phrases: set  = set()
    matched:          list = []
    unmatched:        list = []

    for truth in ground_truth_list:
        tn = normalize(truth)
        hit, consumed_phrases = _sliding_window_match(
            truth_norm       = tn,
            disc_norms       = disc_norms,
            consumed_phrases = consumed_phrases,
            threshold        = threshold,
        )
        (matched if hit else unmatched).append(truth)

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
            job["embedding"] = vec.tolist()

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
    Stage 3 — hybrid scoring: 0.70 x cosine  +  0.30 x skill_match_ratio.

    NLP validation (fuzzy_skill_verify) is run against every matched job's
    required_skills so each result carries its own accuracy breakdown.

    Returns top_k job dicts sorted by score descending.
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

    k                = min(top_k, index.ntotal)
    cos_scores, idxs = index.search(q_vec, k)
    resume_norms     = {normalize(s) for s in resume_skills}
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