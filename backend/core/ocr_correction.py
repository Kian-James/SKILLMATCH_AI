"""
core/ocr_correction.py
======================
Embedding-based OCR correction using SBERT + FAISS.

Looks up the nearest allowlist entry for an OCR-garbled token via a FAISS
IndexFlatIP built from the skills/certs allowlist, rather than relying on a
hardcoded correction table.

Token correction pipeline (see apply_ocr_corrections for the exact order)
--------------------------------------------------------------------------
Each whitespace-split token is run through a sequence of cheap, targeted
fixes before falling back to the FAISS nearest-neighbour lookup:

1. Protected tokens ("js", "&", etc.) are passed through untouched — these
   are meaningful abbreviations or connectors only in context (e.g. "js" in
   "Node js" should become part of "Node.js", not get expanded on its own
   to "JavaScript"), so they're never sent to FAISS.
2. Letter/letter substitution — fixes common visual OCR confusions such as
   l<->I, rn<->m, vv<->w, cl<->d, 0<->O (e.g. "UlKit" -> "UIKit").
3. Digit/letter substitution — fixes confusions like O<->0, I<->1, S<->5,
   B<->8, Z<->2 in tokens that mix letters and digits (e.g. "SAP2OOO" ->
   "SAP2000").
4. Separator recovery — restores a missing '/' or '&' that OCR collapsed or
   replaced with a stray letter in a compact uppercase token (e.g.
   "IDSIIPS" -> "IDS/IPS", "GDeT" -> "GD&T", "HTMLICSS" -> "HTML/CSS"). If
   this step turns the token into a multi-part compound, correction stops
   here — the token is handed back as-is rather than continuing to FAISS.
   FAISS only knows how to collapse a whole token down to a single nearest
   allowlist word, which would silently drop one half of a compound like
   "HTML/CSS" (e.g. reducing it to just "CSS"). Splitting a recovered
   compound into its parts is the extractor's job (_expand_slash /
   _expand_ampersand in extractor_base.py), not this function's.
5. Allowlist pre-check — if the token already scores highly against the
   allowlist as-is, skip FAISS entirely. This protects correctly-spelled
   versioned tokens like "HTML5", "CSS3", "ES6+", which might otherwise
   embed close to an unrelated but semantically similar allowlist entry.
6. FAISS nearest-neighbour lookup — the final fallback for tokens that
   survived all of the above uncorrected.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import faiss
import numpy as np
from rapidfuzz import fuzz

from config import EMBEDDING_MODEL, OCR_EMBED_THRESHOLD
from core.utils import normalize

log = logging.getLogger(__name__)

# Tokens that are meaningful tech abbreviations or connectors only in
# context, and should never be sent through the FAISS lookup on their own
# (e.g. "js" alone embeds closest to "JavaScript" and would turn
# "Node js" into "Node JavaScript" instead of "Node.js").
_PROTECTED_SUFFIXES: frozenset[str] = frozenset({
    "js",   # Node.js, Express.js, Vue.js, React.js, Next.js ...
    "ts",   # TypeScript suffix in some OCR outputs
    "py",   # occasionally appears in "Python py"
    "rb",   # Ruby
    "rs",   # Rust
    "go",   # Go language token
    "cs",   # C# / .NET context
    "&",    # ampersand connector in compound skills like "HTML5 & CSS3"
})

# If a token already scores >= this against the allowlist, skip FAISS to
# avoid wrongly substituting a correctly-spelled versioned token (HTML5,
# CSS3, ES6+). High threshold so genuinely garbled tokens still go to FAISS.
_FAISS_SKIP_THRESHOLD: int = 88

# ── Digit/letter substitution map ────────────────────────────────────────────
# Common OCR confusions. Applied bidirectionally on uppercase tokens.
_LETTER_TO_DIGIT: dict[str, str] = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "l": "1"}
_DIGIT_TO_LETTER: dict[str, str] = {v: k for k, v in _LETTER_TO_DIGIT.items()}

# Only attempt substitution when token has BOTH letters and digits (mixed).
_MIXED_ALPHANUM_RE = re.compile(r'(?=[A-Za-z])(?=.*\d)|(?=\d)(?=.*[A-Za-z])')

# ── Letter/letter substitution pairs ─────────────────────────────────────────
# Each entry is (find_str, replace_str), applied as a single-position
# find-and-replace pass over the token. Both directions are listed
# explicitly so the table is easy to audit. Multi-char entries (rn<->m,
# vv<->w, cl<->d) are substring replacements on the whole token string.
_LETTER_LETTER_SUBS: list[tuple[str, str]] = [
    # Most common: lowercase l misread as uppercase I and vice versa
    ("l", "I"),
    ("I", "l"),
    # Two-character sequence misreadings
    ("rn", "m"),
    ("m", "rn"),
    ("vv", "w"),
    ("w", "vv"),
    ("cl", "d"),
    ("d", "cl"),
    # Zero / letter-O confusion in all-letter context
    ("0", "O"),
    ("O", "0"),
]

# Minimum token length for letter/letter substitution — avoids producing
# noise on very short tokens like "l" -> "I".
_MIN_LEN_FOR_LETTER_SUB: int = 3

# ── Separator recovery constants ─────────────────────────────────────────────
# Minimum token length to attempt separator insertion.
_MIN_LEN_FOR_SEP_RECOVERY: int = 4

# Separator chars to try inserting.
_SEP_CHARS: tuple[str, ...] = ("/", "&")

# Characters that OCR commonly substitutes for a separator in compact tokens.
_OCR_SEP_SURROGATES: frozenset[str] = frozenset("Il1|")


def _normalize_sep(s: str) -> str:
    """
    Separator-aware normalize used by _separator_candidates and the matching
    logic in extractor_base.py's _restore_ocr_slash.

    Unlike the standard normalize() which strips '/' and '&', this variant
    converts them to spaces so that "IDS/IPS" -> "ids ips" and "IDS/IIPS" ->
    "ids iips" are distinguishable by fuzz.ratio (standard normalize collapses
    both to "idsips", making all split positions score identically).
    """
    s = s.lower()
    s = re.sub(r"[/&]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _letter_letter_candidates(token: str, allow_norm_keys: list[str]) -> str:
    """
    Generate variants of *token* by applying common letter/letter visual
    substitutions and return the variant that scores highest against
    *allow_norm_keys*.

    Only runs when the token is long enough (_MIN_LEN_FOR_LETTER_SUB) to
    avoid noise on short tokens. Returns the original token when no
    substitution improves the score.

    Each substitution pair is applied as a single full-string replacement
    pass. Only one substitution is applied at a time (no combinations) to
    keep the candidate space O(k) where k = len(_LETTER_LETTER_SUBS).

    Example:
        "UlKit" with sub ("l", "I") -> "UIKit"
        normalize("UIKit") -> "uikit"
        fuzz.ratio("uikit", "uikit") -> 100  => committed, returns "UIKit"
    """
    stripped = token.strip(".,;:()[]")

    if len(stripped) < _MIN_LEN_FOR_LETTER_SUB:
        return token

    original_norm = normalize(stripped)
    best_score = max(
        (fuzz.ratio(original_norm, ak) for ak in allow_norm_keys),
        default=0,
    )
    best_token = token

    for find_str, replace_str in _LETTER_LETTER_SUBS:
        if find_str not in stripped:
            continue
        candidate = stripped.replace(find_str, replace_str)
        if candidate == stripped:
            continue
        candidate_norm = normalize(candidate)
        score = max(
            (fuzz.ratio(candidate_norm, ak) for ak in allow_norm_keys),
            default=0,
        )
        if score > best_score:
            best_score = score
            best_token = candidate + token[len(stripped):]
            log.debug(
                f"[OcrCorrector/LetterSub] '{token}' -> '{best_token}' "
                f"via ('{find_str}'->'{replace_str}') score={best_score}"
            )

    return best_token


def _digit_letter_candidates(token: str, allow_norm_keys: list[str]) -> str:
    """
    Generate variants of *token* by substituting visually similar digit/letter
    pairs and return the variant that scores highest against *allow_norm_keys*.

    Only runs when the token is mixed alphanumeric — pure letters and pure
    digits are returned unchanged (no substitution needed).

    Returns the best candidate if it scores > the original, else the original.
    """
    stripped = token.strip(".,;:()[]")

    # Only process mixed alphanumeric tokens
    if not (any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped)):
        return token

    # Build substitution positions: indices where a substitutable char appears
    sub_map = {**_LETTER_TO_DIGIT, **_DIGIT_TO_LETTER}
    positions = [(i, sub_map[c]) for i, c in enumerate(stripped) if c in sub_map]

    if not positions:
        return token

    # Generate all 2^n substitution combinations (cap at 4 positions = 16 variants)
    positions = positions[:4]
    original_norm = normalize(stripped)
    best_score = max(
        (fuzz.ratio(original_norm, ak) for ak in allow_norm_keys),
        default=0,
    )
    best_token = token

    for r in range(1, len(positions) + 1):
        for combo in itertools.combinations(positions, r):
            chars = list(stripped)
            for idx, replacement in combo:
                chars[idx] = replacement
            candidate = "".join(chars)
            candidate_norm = normalize(candidate)
            score = max(
                (fuzz.ratio(candidate_norm, ak) for ak in allow_norm_keys),
                default=0,
            )
            if score > best_score:
                best_score = score
                best_token = candidate + token[len(stripped):]

    if best_token != token:
        log.debug(f"[OcrCorrector/DigitSub] '{token}' -> '{best_token}' (score={best_score})")

    return best_token


def _separator_candidates(token: str, allowlist_raw: list[str]) -> str:
    """
    Recover separator characters ('/' or '&') that OCR collapsed or replaced
    with a stray letter in compact uppercase tokens.

    Examples fixed:
        "IDSIIPS"  -> "IDS/IPS"   (OCR inserted 'I' as separator + duplicated it)
        "GDeT"     -> "GD&T"      (OCR replaced '&' with 'e')
        "QAIQC"    -> "QA/QC"     (OCR replaced '/' with 'I')
        "TCPIP"    -> "TCP/IP"    (OCR dropped '/')
        "HTMLICSS" -> "HTML/CSS"  (OCR replaced '/' with 'I')

    Algorithm
    ---------
    For every interior position i in the stripped token, two variants are tried:
      A) Insert sep at i, keep char:     token[:i] + sep + token[i:]
      B) Insert sep at i, consume char:  token[:i] + sep + token[i+1:]
         (handles cases where OCR injected an extra letter AS the separator,
          e.g. "IDSIIPS" = "IDS" + "I"[fake-slash] + "IPS")

    Scoring uses two tiers:
      Primary:   fuzz.ratio against separator-aware normalized allowlist keys
                 (_normalize_sep converts / and & to space, so "IDS/IPS" ->
                 "ids ips" and "IDS/IIPS" -> "ids iips" are distinguishable).
      Secondary: fuzz.ratio against raw lowercased allowlist strings, used as
                 a tiebreaker when primary scores are equal (e.g. "GD/T" and
                 "GD&T" both normalize_sep to "gd t", but raw "gd&t" matches
                 allowlist "gd&t" at 100 while "gd/t" scores 75).

    Only fires on purely alphanumeric tokens (no existing '/', '&', '.', etc.)
    that are at least _MIN_LEN_FOR_SEP_RECOVERY characters long.

    Runs before the allowlist pre-check in apply_ocr_corrections, since a
    collapsed token like "IDSIIPS" can otherwise score deceptively high
    against the allowlist once '/' is stripped by standard normalize (both
    become "idsips"), which would skip correction before it's even tried.

    The caller (apply_ocr_corrections) must not pass a token that just
    gained a separator from this function on to the FAISS lookup — FAISS
    replaces a whole token with a single nearest canonical word and would
    collapse a freshly-recovered compound like "HTML/CSS" down to just
    "CSS". See apply_ocr_corrections for that guard.
    """
    stripped = token.strip(".,;:()[]")

    if not stripped or len(stripped) < _MIN_LEN_FOR_SEP_RECOVERY:
        return token

    # Only attempt on purely alphanumeric tokens (no existing separators).
    if not re.match(r"^[A-Za-z0-9]+$", stripped):
        return token

    allow_sep_keys = [_normalize_sep(a) for a in allowlist_raw]

    def _score(candidate: str) -> tuple[float, float]:
        primary   = max(
            (fuzz.ratio(_normalize_sep(candidate), ak) for ak in allow_sep_keys),
            default=0,
        )
        secondary = max(
            (fuzz.ratio(candidate.lower(), a.lower()) for a in allowlist_raw),
            default=0,
        )
        return (primary, secondary)

    best       = _score(stripped)
    best_token = token
    suffix     = token[len(stripped):]

    for sep in _SEP_CHARS:
        for i in range(1, len(stripped)):
            # Variant A: insert separator, keep char at position i.
            ca = stripped[:i] + sep + stripped[i:]

            # Variant B: insert separator, consume char at position i.
            # Only emit when there is still content on the right side.
            cb = (stripped[:i] + sep + stripped[i + 1:]) if i < len(stripped) - 1 else None

            for candidate in filter(None, [ca, cb]):
                # Ensure both sides of the separator are non-empty.
                parts = candidate.split(sep)
                if not all(p.strip() for p in parts):
                    continue
                s = _score(candidate)
                if s > best:
                    best       = s
                    best_token = candidate + suffix

    if best_token != token:
        log.debug(
            f"[OcrCorrector/SepRecover] '{token}' -> '{best_token}' "
            f"(primary={best[0]:.0f}, secondary={best[1]:.0f})"
        )

    return best_token


# ── SBERT singleton ───────────────────────────────────────────────────────────

_model = None


def get_sbert():
    """Lazy-load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"[SBERT] Loading {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("[SBERT] Ready.")
    return _model


# ── OcrCorrector ─────────────────────────────────────────────────────────────

@dataclass
class OcrCorrector:
    """FAISS index built from allowlist embeddings for nearest-neighbour correction."""
    index:           faiss.IndexFlatIP
    labels:          list[str]
    encode_fn:       Callable[[str], np.ndarray]
    cache:           dict[str, str] = field(default_factory=dict)
    allow_norm_keys: list[str]      = field(default_factory=list)
    # Raw allowlist strings, used by _separator_candidates for tiebreaking.
    allowlist_raw:   list[str]      = field(default_factory=list)


def build_ocr_corrector(allowlist: list[str]) -> OcrCorrector:
    """Build an OcrCorrector from a list of canonical skill/cert strings."""
    sbert = get_sbert()
    dim   = sbert.get_sentence_embedding_dimension()

    if not allowlist:
        return OcrCorrector(
            index          = faiss.IndexFlatIP(dim),
            labels         = [],
            encode_fn      = lambda s: sbert.encode(s, normalize_embeddings=True).astype(np.float32),
            allow_norm_keys= [],
            allowlist_raw  = [],
        )

    log.info(f"[OcrCorrector] Encoding {len(allowlist)} allowlist entries ...")
    vecs = sbert.encode(
        allowlist,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    log.info(f"[OcrCorrector] Index built ({index.ntotal} entries, dim={dim}).")

    def _encode_one(s: str) -> np.ndarray:
        v = sbert.encode(s, normalize_embeddings=True).astype(np.float32)
        faiss.normalize_L2(v.reshape(1, -1))
        return v

    return OcrCorrector(
        index          = index,
        labels         = list(allowlist),
        encode_fn      = _encode_one,
        allow_norm_keys= [normalize(a) for a in allowlist],
        allowlist_raw  = list(allowlist),
    )


def ocr_correct_token(token: str, corrector: OcrCorrector) -> str:
    """
    Return the nearest canonical allowlist form if similarity >= threshold.

    When the nearest canonical form is multi-word, we pick the single word
    from it that best matches the OCR token — using normalize() consistently
    on both sides, identical to matcher.py behaviour.
    """
    if not corrector.labels or not token.strip():
        return token

    stripped  = token.strip(".,;:()[]")
    cache_key = stripped.lower()

    if cache_key in corrector.cache:
        return corrector.cache[cache_key] + token[len(stripped):]

    vec = corrector.encode_fn(stripped).reshape(1, -1)
    D, I = corrector.index.search(vec, 1)
    similarity = float(D[0][0])

    if similarity >= OCR_EMBED_THRESHOLD and I[0][0] >= 0:
        canonical   = corrector.labels[I[0][0]]
        canon_words = canonical.split()
        if len(canon_words) > 1:
            norm_stripped = normalize(stripped)
            corrected = max(
                canon_words,
                key=lambda w: fuzz.ratio(normalize(w), norm_stripped),
            )
        else:
            corrected = canonical
        log.debug(f"[OcrCorrector] '{stripped}' -> '{corrected}' (sim={similarity:.3f})")
    else:
        corrected = stripped

    corrector.cache[cache_key] = corrected
    return corrected + token[len(stripped):]


def apply_ocr_corrections(s: str, corrector: Optional[OcrCorrector] = None) -> str:
    """
    Token-level OCR correction. No-op when corrector is None.

    Processing order per token (see module docstring for the reasoning
    behind each step):
      1. Skip tokens in _PROTECTED_SUFFIXES.
      2. _letter_letter_candidates — fix l<->I, rn<->m, etc.
      3. _digit_letter_candidates  — fix O<->0, I<->1, etc. in mixed tokens.
      4. _separator_candidates     — recover missing '/' or '&' in compact
                                     uppercase tokens.
      4b. If step 4 just resolved the token into a multi-part compound,
          stop here and return it as-is — splitting it further is the
          extractor's job (_expand_slash / _expand_ampersand), not this
          function's, and FAISS would otherwise collapse it to one word.
      5. Allowlist pre-check — skip FAISS if the token already scores well.
      6. ocr_correct_token — FAISS nearest-neighbour lookup.
    """
    if corrector is None:
        return s

    corrected_tokens = []
    for tok in s.split():
        stripped_lower = tok.strip(".,;:()[]").lower()

        # Step 1: skip protected tokens entirely (includes "&")
        if stripped_lower in _PROTECTED_SUFFIXES:
            log.debug(f"[OcrCorrector] '{tok}' protected — skipping correction")
            corrected_tokens.append(tok)
            continue

        # Step 2: fix letter/letter OCR confusions
        tok = _letter_letter_candidates(tok, corrector.allow_norm_keys)

        # Step 3: fix digit/letter OCR confusions
        tok = _digit_letter_candidates(tok, corrector.allow_norm_keys)

        # Step 4: recover missing '/' or '&' in compact uppercase tokens.
        pre_sep_tok = tok
        tok = _separator_candidates(tok, corrector.allowlist_raw)

        # Step 4b: if separator recovery just turned this token into a
        # multi-part compound, hand it back now rather than letting it fall
        # through to FAISS, which would collapse it to a single nearest word.
        if tok != pre_sep_tok and ("/" in tok or "&" in tok):
            log.debug(
                f"[OcrCorrector] '{pre_sep_tok}' -> '{tok}' resolved to a "
                f"compound — skipping FAISS, leaving split to the caller"
            )
            corrected_tokens.append(tok)
            continue

        # Step 5: if the token already matches the allowlist well, skip FAISS
        tok_norm = normalize(tok.strip(".,;:()[]"))
        if tok_norm and corrector.allow_norm_keys:
            pre_score = max(
                (fuzz.ratio(tok_norm, ak) for ak in corrector.allow_norm_keys),
                default=0,
            )
            if pre_score >= _FAISS_SKIP_THRESHOLD:
                log.debug(
                    f"[OcrCorrector] '{tok}' pre-score={pre_score} >= "
                    f"{_FAISS_SKIP_THRESHOLD} — skipping FAISS"
                )
                corrected_tokens.append(tok)
                continue

        # Step 6: FAISS nearest-neighbour lookup
        corrected_tokens.append(ocr_correct_token(tok, corrector))

    return " ".join(corrected_tokens)