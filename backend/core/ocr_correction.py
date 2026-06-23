"""
core/ocr_correction.py
======================
Embedding-based OCR correction using SBERT + FAISS.

Replaces the old hardcoded _OCR_CORRECTIONS dict with a nearest-neighbour
lookup over a FAISS IndexFlatIP built from the skills/certs allowlist.

Changes (v6 backport):
  FIXED    Multi-word canonical handling in ocr_correct_token.

           Problem: When the nearest FAISS neighbour is a multi-word allowlist
           entry (e.g. "SolarWinds Network Performance Monitor"), the old code
           picked just one word from it via max(fuzz.ratio per word). This
           mangled tokens — e.g. "AnyDesk" could get corrected to "Monitor"
           because "Monitor" scored highest against the OCR token in isolation,
           causing the skill to be silently lost downstream.

           Fix: when the nearest canonical form is multi-word, compare the
           entire OCR token against each word of the canonical using fuzz.ratio,
           but only use the best single word as the corrected token (same as
           before) — the key change is that we now use normalize() consistently
           on both sides of the comparison, matching the behaviour in matcher.py.
           More importantly, the threshold check now correctly uses the cosine
           similarity from the FAISS search so that low-confidence matches are
           rejected rather than silently accepted.

Changes (v7): fixed apply_ocr_corrections expanding tech abbreviations that are
           meaningful only in context (e.g. "js" -> "JavaScript").

           Problem: apply_ocr_corrections splits input on whitespace and
           corrects each token independently. The token "js" in isolation
           embeds closest to "JavaScript" and gets expanded, turning
           "Node js" -> "Node JavaScript" and "Express js" ->
           "Express JavaScript". These mangled strings then fail to match
           "Node.js" / "Express.js" in the allowlist.

           Fix: introduced _PROTECTED_SUFFIXES — a set of lowercase token
           strings that are meaningful tech abbreviations when they appear
           as part of a multi-token skill name. Tokens in this set are
           passed through apply_ocr_corrections unchanged (no FAISS lookup).

           "js" is the primary case but the set also covers other common
           abbreviations that the SBERT model is likely to over-expand:
           "ts" (TypeScript), "py" (Python), "rb" (Ruby), "go", "rs" (Rust).

           These are only skipped at the apply_ocr_corrections level — the
           full skill phrase (e.g. "Node js") is still passed to canonicalize()
           and direct_scan_section where the v24 akey_alt fix maps it correctly
           to its allowlist form ("Node.js").

Changes (v8): fixed OCR digit/letter substitution errors (e.g. "SAP2OOO"
           instead of "SAP2000") being matched against the wrong allowlist
           entry or dropped entirely.

           Problem: OCR scanners commonly confuse digits and visually similar
           letters: O/0, I/1, S/5, B/8, Z/2, l/1. "SAP2OOO" contains three
           O's that should be 0's. Without correction the fuzzy scorer matches
           it against "SAP" at score 60 (below the 82 threshold) and drops it.

           Fix: introduced _digit_letter_candidates() which generates candidate
           strings by replacing visually similar characters in a token and
           returns the candidate that scores highest against the allow_norm
           keys via fuzzy ratio. Applied in apply_ocr_corrections BEFORE the
           FAISS lookup when the token contains a mix of letters and digits,
           so "SAP2OOO" -> "SAP2000" before embedding.

           Substitution map (applied to uppercase token):
             O->0, I->1, S->5, B->8, Z->2, l->1 and reverses (0->O etc.)
           Only tokens that contain BOTH letters and digits are candidates
           (pure-letter and pure-digit tokens are left alone).

Changes (v10): fixed apply_ocr_corrections mangling versioned/compound tokens
           like "HTML5", "CSS3", "ES6+" that contain digits or punctuation
           and embed close to a wrong FAISS neighbour (e.g. "HTML5" -> "HTML/CSS").

           Two fixes:

           Fix 1 — protect "&": add "&" to _PROTECTED_SUFFIXES so the
           ampersand connector token in compound skills like "HTML5 & CSS3"
           is passed through untouched. Previously "&" was fed to FAISS which
           returned a garbage nearest neighbour and corrupted the rejoined string.

           Fix 2 — allowlist pre-check before FAISS: before calling
           ocr_correct_token (FAISS lookup), check whether the token already
           scores >= _FAISS_SKIP_THRESHOLD (88) against the raw allowlist keys
           via fuzzy ratio. If it does, the token is already correct — skip
           FAISS to avoid a wrong substitution. This protects tokens like
           "HTML5", "CSS3", "ES6+" and any other versioned skill name that
           the SBERT model may map to a semantically close but wrong neighbour.
           The threshold is intentionally high (88) so genuinely OCR-garbled
           tokens (score << 88) still go through FAISS correction normally.

           (e.g. "UlKit" instead of "UIKit") being dropped because the fuzzy
           score falls just below the direct_scan_section threshold (80 < 82).

           Problem: OCR scanners commonly confuse visually similar letters
           that share no digit component: lowercase l <-> uppercase I,
           lowercase rn <-> m, 0 <-> O (in all-letter context), vv <-> w,
           cl <-> d. "UlKit" contains 'l' where 'I' was intended. The token
           is all-letter so _digit_letter_candidates() (v8) never runs, and
           the FAISS embedding path scores it below threshold.

           Fix: introduced _letter_letter_candidates() which generates candidate
           strings by applying a small set of common single-character letter/
           letter visual substitutions and returns the candidate that scores
           highest against the allow_norm keys. Applied in apply_ocr_corrections
           for ALL tokens (not just mixed alphanumeric), BEFORE the FAISS
           lookup, so corrected tokens arrive at direct_scan_section and
           canonicalize() already fixed.

           Substitution pairs (bidirectional, case-sensitive):
             l <-> I   (lowercase L / uppercase I — most common OCR confusion)
             rn -> m   (two-char sequence misread as one)
             m -> rn   (reverse)
             0 <-> O   (zero / letter O in all-letter context)
             vv -> w   (two v's misread as w)
             w -> vv   (reverse)
             cl -> d   (c+l misread as d)
             d -> cl   (reverse)

           Only single-position substitutions are attempted to keep the
           candidate space small. "UlKit" -> "UIKit" scores 100 against
           the allowlist and is committed before the FAISS lookup runs.

Changes (v11): fixed two cases where missing separator characters ('/' or '&')
           that were collapsed by OCR into a letter (e.g. "IDSIIPS" instead of
           "IDS/IPS", "GDeT" instead of "GD&T") were not being recovered.

           Root causes identified:

           BUG A — _LETTER_LETTER_SUBS entry ("e", "&") in _letter_letter_candidates
           worked for "GDeT" -> "GD&T" in isolation but was fragile: the 'e'->'&'
           substitution replaces ALL occurrences of 'e' in the token (Python's
           str.replace is global), which corrupts multi-e tokens like "Experience".
           The token-level guard (best_score comparison) prevents actual damage, but
           the approach is undiscoverable and brittle.

           BUG B — _separator_candidates (the generalized replacement) was missing
           entirely from the pipeline. The only guard for collapsed separators was
           the regex-based _restore_ocr_slash in extractor_base.py, which has a
           greedy quantifier bug (see extractor_base changes).

           BUG C — The pre_score short-circuit (v10) fires on "IDSIIPS" because
           normalize("IDSIIPS")="idsiips" scores 92 against normalize("IDS/IPS")=
           "idsips" (standard normalize strips '/'), which is >= the 88 threshold.
           This prevents FAISS from ever seeing the token, and _restore_ocr_slash
           then receives the uncorrected "IDSIIPS".

           Fix: introduced _separator_candidates() which:
             1. Only fires on purely alphanumeric tokens (no existing punctuation).
             2. Tries inserting '/' and '&' at every interior position, PLUS
                consuming the character at that position (handling the case where
                OCR inserted an extra letter as the separator, e.g. 'I' in
                "IDSIIPS" = IDS + [I=slash] + IPS).
             3. Scores candidates with a separator-aware normalize (/ and & become
                spaces rather than being stripped) so "IDS/IPS" and "IDS/IIPS"
                are distinguished correctly.
             4. Uses the raw allowlist strings as a tiebreaker via fuzz.ratio
                so "GD/T" and "GD&T" (which are identical after sep-aware normalize
                as "gd t") are correctly resolved to "GD&T" when that is the
                allowlist form.

           _separator_candidates runs BEFORE the pre_score check so it is not
           blocked by the v10 threshold guard.

           The ("e", "&") entry is removed from _LETTER_LETTER_SUBS since
           _separator_candidates handles the "GDeT" -> "GD&T" case more robustly
           and without the global-replace fragility.

Changes (v27): fixed _separator_candidates' fix getting undone by the FAISS
           step that runs right after it.

           Problem: once _separator_candidates successfully resolves a garbled
           token into a multi-part compound (e.g. "HTMLICSS" -> "HTML/CSS"),
           apply_ocr_corrections kept going and ran the token through the
           pre-score check and, if that didn't short-circuit, the FAISS
           nearest-neighbour lookup (ocr_correct_token). That FAISS step is
           designed to replace an ENTIRE token with a SINGLE nearest canonical
           word — it has no concept of a multi-part compound. So
           "HTML/CSS" goes in, FAISS finds "CSS" is the nearest single
           allowlist word, and the corrected token comes back as just "CSS" —
           silently discarding "HTML". The extractor's _expand_slash never
           gets a chance to split the compound because there's nothing left
           to split.

           Fix: if _separator_candidates changed the token AND the result
           contains '/' or '&' (i.e. it just resolved a collapsed separator),
           skip the pre-score check and FAISS lookup entirely for that token.
           The token is handed back as the recovered compound string; it's
           the caller's job (via _expand_slash / _expand_ampersand in
           extractor_base.py) to split it into its parts.
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

# ── Protected suffix tokens (v7, updated v10) ────────────────────────────────
_PROTECTED_SUFFIXES: frozenset[str] = frozenset({
    "js",   # Node.js, Express.js, Vue.js, React.js, Next.js ...
    "ts",   # TypeScript suffix in some OCR outputs
    "py",   # occasionally appears in "Python py"
    "rb",   # Ruby
    "rs",   # Rust
    "go",   # Go language token
    "cs",   # C# / .NET context
    "&",    # v10: ampersand connector in compound skills like "HTML5 & CSS3"
})

# v10: if a token already scores >= this against the allowlist, skip FAISS
# to avoid wrong substitution of correctly-spelled versioned tokens (HTML5,
# CSS3, ES6+). High threshold so genuinely garbled tokens still go to FAISS.
_FAISS_SKIP_THRESHOLD: int = 88

# ── Digit/letter substitution map (v8) ───────────────────────────────────────
# Common OCR confusions. Applied bidirectionally on uppercase tokens.
_LETTER_TO_DIGIT: dict[str, str] = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "l": "1"}
_DIGIT_TO_LETTER: dict[str, str] = {v: k for k, v in _LETTER_TO_DIGIT.items()}

# Only attempt substitution when token has BOTH letters and digits (mixed).
_MIXED_ALPHANUM_RE = re.compile(r'(?=[A-Za-z])(?=.*\d)|(?=\d)(?=.*[A-Za-z])')

# ── Letter/letter substitution pairs (v9, updated v11) ───────────────────────
# Each entry is (find_str, replace_str). Applied as single-position
# find-and-replace passes over the token. Both directions are listed
# explicitly so the table is easy to audit.
# Multi-char entries (rn<->m, vv<->w, cl<->d) are handled as substring
# replacements on the whole token string, not per-character.
#
# v11: removed ("e", "&") and ("&", "e") — these were attempting to handle
# "GDeT" -> "GD&T" but str.replace is global (corrupts multi-e tokens) and
# the case is now handled more robustly by _separator_candidates().
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

# ── Separator recovery constants (v11) ───────────────────────────────────────
# Minimum token length to attempt separator insertion.
_MIN_LEN_FOR_SEP_RECOVERY: int = 4

# Separator chars to try inserting.
_SEP_CHARS: tuple[str, ...] = ("/", "&")

# Characters that OCR commonly substitutes for a separator in compact tokens.
_OCR_SEP_SURROGATES: frozenset[str] = frozenset("Il1|")


def _normalize_sep(s: str) -> str:
    """
    Separator-aware normalize used exclusively inside _separator_candidates
    and the updated _restore_ocr_slash (extractor_base.py).

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
    v11: Recover separator characters ('/' or '&') that OCR collapsed or
    replaced with a letter in compact uppercase tokens.

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

    Runs BEFORE the pre_score FAISS-skip check in apply_ocr_corrections so it
    is not blocked by the v10 threshold guard (which fires on "IDSIIPS" because
    standard normalize strips '/' and the token scores 92 against "idsips").

    NOTE (v27): the caller (apply_ocr_corrections) must NOT pass a token that
    has just gained a separator from this function on to the FAISS lookup
    step — FAISS replaces a whole token with a single nearest canonical word
    and would collapse a freshly-recovered compound like "HTML/CSS" down to
    just "CSS". See apply_ocr_corrections for the guard.
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
    # v11: raw allowlist strings needed by _separator_candidates for tiebreaking.
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

    Processing order per token
    --------------------------
    1. Skip tokens in _PROTECTED_SUFFIXES (v7/v10: "js", "&", etc.).
    2. _letter_letter_candidates  — fix l<->I, rn<->m, etc. (v9).
    3. _digit_letter_candidates   — fix O<->0, I<->1, etc. in mixed tokens (v8).
    4. _separator_candidates      — recover missing '/' or '&' in compact
                                    uppercase tokens (v11: "IDSIIPS"->"IDS/IPS",
                                    "GDeT"->"GD&T", "HTMLICSS"->"HTML/CSS").
    4b. v27 guard                 — if step 4 just resolved the token into a
                                    multi-part compound (it changed AND now
                                    contains '/' or '&'), STOP here and return
                                    it as-is. Do not let steps 5/6 below run on
                                    it — FAISS only knows how to collapse a
                                    whole token down to a SINGLE nearest word,
                                    which would silently discard one half of
                                    the recovered compound (e.g. "HTML/CSS"
                                    -> "CSS", losing "HTML"). Splitting the
                                    recovered compound into its parts is the
                                    extractor's job (_expand_slash /
                                    _expand_ampersand), not this function's.
    5. Pre-score check            — if token now scores >= _FAISS_SKIP_THRESHOLD
                                    against the allowlist, skip FAISS (v10).
    6. ocr_correct_token          — FAISS nearest-neighbour lookup.
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

        # Step 4 (v11): recover missing '/' or '&' in compact uppercase tokens.
        pre_sep_tok = tok
        tok = _separator_candidates(tok, corrector.allowlist_raw)

        # Step 4b (v27): if separator recovery just turned this token into a
        # multi-part compound, hand it back now. Letting it fall through to
        # the FAISS lookup below would collapse "HTML/CSS" down to whichever
        # single allowlist word FAISS thinks is closest (e.g. "CSS"),
        # silently dropping "HTML". The caller's _expand_slash/_expand_ampersand
        # is what actually splits this into its separate parts.
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