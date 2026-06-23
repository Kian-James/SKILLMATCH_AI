"""
evaluate_pipeline.py
====================
Comprehensive accuracy evaluator for the OCR -> NLP (SBERT) -> FAISS pipeline.

Modes
-----
  --mode ocr       : OCR text extraction quality (character yield, alnum ratio,
                     expected-text similarity if labels supply expected_text)
  --mode nlp       : Skill AND cert extraction accuracy (precision / recall / F1
                     vs expected_skills / expected_certs in labels.json)
  --mode faiss     : FAISS job-ranking accuracy (did the expected_job_title land
                     in top-k? cosine / hybrid score of the best title match,
                     plus skill and cert match ratios for that best match)
  --mode pipeline  : Full end-to-end run of all three stages for every sample,
                     prints a combined per-sample table + aggregate summary

Usage
-----
  # Run both types together
  python evaluate_pipeline.py --labels labels.json --samples-pdf ./resumes/ --samples-img ./resumes-images/

  # Run PDFs only (won't touch Firestore for image labels)
  python evaluate_pipeline.py --labels labels.json --samples-pdf ./resumes/ --type pdf

  # Run images only
  python evaluate_pipeline.py --labels labels.json --samples-img ./resumes-images/ --type img

  # Save report + charts for a PDF-only run
  python evaluate_pipeline.py --labels labels.json --samples-pdf ./resumes/ --type pdf --output report_pdf.json --charts

labels.json schema (per file)
------------------------------
{
  "Com - Marvin Santos.jpg": {
    "expected_text":      "optional raw OCR ground truth string",
    "expected_skills":    ["PHP", "MySQL", ...],
    "expected_certs":     ["AWS Certified Cloud Practitioner", ...],
    "expected_job_title": "CodeIgniter PHP Developer",
    "min_match_score":    50
  }
}

NOTE: expected_certs is optional. If omitted, cert metrics simply report as
N/A for that sample (same convention as expected_skills/expected_text).
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import sys
import time
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz.distance import Levenshtein as _LevenshteinDistance

# -- silence noisy sub-loggers before any app import -------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("evaluator")
log.setLevel(logging.INFO)

# -- colour helpers (graceful fallback on Windows without ANSI) --------------
try:
    import colorama; colorama.init()
    _C = dict(
        RESET="\033[0m", BOLD="\033[1m",
        GREEN="\033[92m", RED="\033[91m", YELLOW="\033[93m",
        CYAN="\033[96m",  BLUE="\033[94m", MAGENTA="\033[95m",
        DIM="\033[2m",
    )
except ImportError:
    _C = {k: "" for k in ("RESET","BOLD","GREEN","RED","YELLOW","CYAN","BLUE","MAGENTA","DIM")}


def _c(colour: str, text: str) -> str:
    return f"{_C.get(colour,'')}{text}{_C['RESET']}"


# ----------------------------------------------------------------------------
# Academic-standard OCR accuracy benchmark (CER / WER, order-tolerant)
# ----------------------------------------------------------------------------
# Reports the two metrics that dominate OCR literature and thesis work
# (ICDAR robust-reading competitions, the UNLV-ISRI OCR Accuracy
# benchmarking toolset, and most journal papers on OCR systems):
#
#   - CER -- Character Error Rate  = Levenshtein(hyp_chars, ref_chars) / len(ref_chars)
#   - WER -- Word Error Rate       = Levenshtein(hyp_words, ref_words) / len(ref_words)
#
# Both are edit-distance ratios in [0, infinity) where 0 = perfect and lower is
# better. Character/Word Accuracy (1 - CER / 1 - WER) are reported
# alongside since most thesis result tables report accuracy rather than
# error rate.
#
# THE REORDERING PROBLEM
# -----------------------
# Standard CER/WER assume the hypothesis and reference are in the same
# linear reading order. Real OCR pipelines routinely break this assumption
# when layout reconstruction misjudges multi-column or non-linear resume
# layouts -- e.g. extracting the "Skills" block before "Work Experience"
# when the ground-truth reference lists them in the opposite order.
#
# Plain Levenshtein distance has no concept of "this whole block just
# moved" -- it can only express a block move as a delete of every character
# at the old position plus an insert of every character at the new
# position, which inflates the edit count by roughly 2x the size of the
# moved block even though not a single character was actually misread.
# That isn't a measurement of OCR quality; it's a measurement of layout
# ordering, and conflating the two understates true character-recognition
# accuracy.
#
# Below, both are reported:
#   - "raw"      CER/WER : standard, order-sensitive, directly comparable
#                          to numbers reported in other OCR papers.
#   - "aligned"  CER/WER : same Levenshtein computation, but the
#                          hypothesis is first split into blocks
#                          (paragraphs) and greedily reordered to match
#                          the reference block order via fuzzy
#                          block-to-block matching, BEFORE the
#                          character/word edit distance is computed. This
#                          isolates genuine recognition errors from
#                          section-ordering differences.
#
# The raw vs. aligned gap is itself a useful reportable statistic for a
# methodology section: a large gap indicates the layout/ordering stage is
# the dominant error source, while a small gap with still-low aligned
# accuracy indicates genuine character-recognition error.
#
# NOTE on this block-alignment step specifically: it is matching whole
# paragraphs of OCR'd text to find re-ordering, not comparing discovered
# skills/certs to a job's requirements -- a different problem from the
# skill/cert semantic matching used elsewhere in this file. rapidfuzz
# remains appropriate here.

# Minimum token_sort_ratio (0-100) for two blocks to be considered the
# "same section" during alignment. Below this, a hypothesis block is
# treated as unmatched (extra content) rather than forced into a
# low-confidence pairing that would distort the aligned score.
_BLOCK_MATCH_THRESHOLD = 40


def _ocr_tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _ocr_normalize_chars(text: str) -> str:
    """Collapse whitespace runs to single spaces so OCR line-wrap/spacing
    artifacts don't get counted as character errors."""
    return re.sub(r"\s+", " ", text).strip()


def _ocr_split_blocks(text: str) -> list[str]:
    """
    Split text into logical blocks for reordering-tolerant alignment.
    Primary split: blank-line-separated paragraphs (this is how most
    section breaks -- "Skills", "Work Experience", "Education" -- survive
    OCR/layout reconstruction). Falls back to individual non-empty lines
    if the text has no blank-line separation at all (some OCR engines
    don't preserve paragraph spacing).
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return blocks if blocks else ([text.strip()] if text.strip() else [])


def _ocr_align_blocks(hyp_blocks: list[str], ref_blocks: list[str]) -> list[str]:
    """
    Reorder hyp_blocks to match the sequence of ref_blocks, using greedy
    highest-similarity-first bipartite matching (rapidfuzz token_sort_ratio,
    itself order-invariant at the word level so a block isn't penalized for
    internal word order while we're solving for BLOCK order).

    Unmatched ref blocks are filled with "" (counted as fully missing
    content in the resulting CER/WER -- this is correct: if nothing in the
    hypothesis resembles that reference block, that's a real omission, not
    an ordering artifact). Unmatched hyp blocks are appended at the end
    (counted as fully extra content -- also correct: genuinely extraneous
    OCR output, not content that's just been moved).

    This only normalizes BLOCK order. It does not touch the actual
    characters within a matched block, so genuine character-level OCR
    errors are still fully captured by the CER/WER computed afterward.
    """
    from rapidfuzz import fuzz

    pairs: list[tuple[float, int, int]] = []
    for hi, h in enumerate(hyp_blocks):
        for ri, r in enumerate(ref_blocks):
            pairs.append((fuzz.token_sort_ratio(h, r), hi, ri))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_hyp: set[int] = set()
    used_ref: set[int] = set()
    match_for_ref: dict[int, int] = {}

    for score, hi, ri in pairs:
        if hi in used_hyp or ri in used_ref:
            continue
        if score < _BLOCK_MATCH_THRESHOLD:
            continue
        match_for_ref[ri] = hi
        used_hyp.add(hi)
        used_ref.add(ri)

    reordered = [
        hyp_blocks[match_for_ref[ri]] if ri in match_for_ref else ""
        for ri in range(len(ref_blocks))
    ]
    leftover = [hyp_blocks[hi] for hi in range(len(hyp_blocks)) if hi not in used_hyp]
    reordered.extend(leftover)
    return reordered


def character_error_rate(hyp: str, ref: str) -> float:
    """
    CER = character-level Levenshtein edit distance / len(ref characters).
    The standard OCR benchmark metric. 0 = perfect; values above 1.0 are
    possible (hypothesis far longer/noisier than reference).
    """
    h, r = _ocr_normalize_chars(hyp), _ocr_normalize_chars(ref)
    if not r:
        return 0.0 if not h else 1.0
    return round(_LevenshteinDistance.distance(h, r) / len(r), 4)


def word_error_rate(hyp: str, ref: str) -> float:
    """
    WER = word-level Levenshtein edit distance / number of ref words.
    The standard OCR/ASR benchmark metric. 0 = perfect.
    """
    h, r = _ocr_tokenize_words(hyp), _ocr_tokenize_words(ref)
    if not r:
        return 0.0 if not h else 1.0
    return round(_LevenshteinDistance.distance(h, r) / len(r), 4)


def bag_of_words_f1(hyp: str, ref: str) -> dict:
    """
    Order-invariant word-level precision / recall / F1 (multiset overlap,
    case-insensitive). Included as a sanity cross-check alongside CER/WER
    specifically because it's immune to ordering by construction -- every
    word counts once regardless of where it sits in either text. If this
    is high while raw WER is poor, that's strong evidence the gap is an
    ordering artifact rather than a recognition error.
    """
    h_words = Counter(w.lower() for w in _ocr_tokenize_words(hyp))
    r_words = Counter(w.lower() for w in _ocr_tokenize_words(ref))

    overlap = sum((h_words & r_words).values())
    tp = overlap
    fp = sum(h_words.values()) - overlap
    fn = sum(r_words.values()) - overlap

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    return dict(precision=round(precision, 4), recall=round(recall, 4),
                f1=round(f1, 4), tp=tp, fp=fp, fn=fn)


@dataclass
class OCRBenchmarkResult:
    cer_raw:     float    # standard CER, sensitive to block/section order
    wer_raw:     float    # standard WER, sensitive to block/section order
    cer_aligned: float    # CER after block-reordering normalization
    wer_aligned: float    # WER after block-reordering normalization
    bow_f1:      dict     # order-invariant bag-of-words precision/recall/F1

    char_accuracy_raw:     float = field(init=False)
    char_accuracy_aligned: float = field(init=False)
    word_accuracy_raw:     float = field(init=False)
    word_accuracy_aligned: float = field(init=False)
    ordering_gap:          float = field(init=False)  # cer_raw - cer_aligned

    def __post_init__(self):
        self.char_accuracy_raw     = round(max(0.0, 1 - self.cer_raw), 4)
        self.char_accuracy_aligned = round(max(0.0, 1 - self.cer_aligned), 4)
        self.word_accuracy_raw     = round(max(0.0, 1 - self.wer_raw), 4)
        self.word_accuracy_aligned = round(max(0.0, 1 - self.wer_aligned), 4)
        self.ordering_gap          = round(self.cer_raw - self.cer_aligned, 4)

    def as_dict(self) -> dict:
        return dict(
            cer_raw=self.cer_raw, wer_raw=self.wer_raw,
            cer_aligned=self.cer_aligned, wer_aligned=self.wer_aligned,
            char_accuracy_raw=self.char_accuracy_raw,
            char_accuracy_aligned=self.char_accuracy_aligned,
            word_accuracy_raw=self.word_accuracy_raw,
            word_accuracy_aligned=self.word_accuracy_aligned,
            ordering_gap=self.ordering_gap,
            bow_precision=self.bow_f1["precision"],
            bow_recall=self.bow_f1["recall"],
            bow_f1=self.bow_f1["f1"],
        )


def benchmark_ocr(hyp_text: str, ref_text: str) -> OCRBenchmarkResult:
    """
    Run the full OCR accuracy benchmark for one (hypothesis, reference)
    pair. See the module-level comment block above for the rationale
    behind raw vs. aligned scoring.
    """
    cer_raw = character_error_rate(hyp_text, ref_text)
    wer_raw = word_error_rate(hyp_text, ref_text)

    hyp_blocks = _ocr_split_blocks(hyp_text)
    ref_blocks = _ocr_split_blocks(ref_text)
    aligned_hyp_text = "\n\n".join(_ocr_align_blocks(hyp_blocks, ref_blocks))

    cer_aligned = character_error_rate(aligned_hyp_text, ref_text)
    wer_aligned = word_error_rate(aligned_hyp_text, ref_text)

    bow = bag_of_words_f1(hyp_text, ref_text)

    return OCRBenchmarkResult(
        cer_raw=cer_raw, wer_raw=wer_raw,
        cer_aligned=cer_aligned, wer_aligned=wer_aligned,
        bow_f1=bow,
    )


# ----------------------------------------------------------------------------
# Lazy corpus loader (cached across calls)
# ----------------------------------------------------------------------------
_CORPUS_CACHE: Optional[tuple] = None   # (firestore_skills, jobs_corpus, firestore_certs)


def _load_corpus():
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        log.info("Loading corpus from Firestore ...")
        from db import load_analyzer_data
        _CORPUS_CACHE = load_analyzer_data()
    return _CORPUS_CACHE


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
# Skill/cert matching below intentionally reuses the LIVE pipeline's own
# semantic matcher (core.scoring._semantic_item_match / _encode_norm) rather
# than reimplementing fuzzy-string matching here. The old version of this
# file had its own local _fuzzy_skill_match (rapidfuzz token_sort_ratio) that
# "mirrored" core/scoring.py's matcher in name only -- once scoring.py moved
# to SBERT semantic matching, that mirror silently went stale and this
# evaluator was scoring against an algorithm production no longer uses.
# Importing the real function instead means this file can't drift out of
# sync with the pipeline again.
def _item_metrics(predicted: list[str], expected: list[str]) -> dict:
    """
    Compute precision / recall / F1 (plus matched / missed / extra) for a
    predicted vs. expected list of skills or certs, using the same SBERT
    semantic matcher the live pipeline uses for skill/cert comparison.
    """
    if not expected:
        return dict(precision=None, recall=None, f1=None,
                    matched=[], missed=[], extra=[],
                    tp=0, fp=0, fn=0)

    from core.scoring import _semantic_item_match, _encode_norm
    from core.utils    import normalize

    pred_norms = [normalize(p) for p in predicted]
    pred_vecs  = _encode_norm(pred_norms)
    exp_norms  = [normalize(e) for e in expected]
    exp_vecs   = _encode_norm(exp_norms)

    matched = [e for e in expected  if     _semantic_item_match(e, pred_norms, pred_vecs)]
    missed  = [e for e in expected  if not _semantic_item_match(e, pred_norms, pred_vecs)]
    extra   = [p for p in predicted if not _semantic_item_match(p, exp_norms,  exp_vecs)]

    tp = len(matched)
    fp = len(extra)
    fn = len(missed)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return dict(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        matched=matched,
        missed=missed,
        extra=extra,
        tp=tp, fp=fp, fn=fn,
    )


def _ocr_text_similarity(predicted: str, expected: str) -> float:
    """Character-level sequence similarity (0-1) -- used when expected_text is set."""
    from rapidfuzz import fuzz
    return round(fuzz.ratio(predicted, expected) / 100.0, 4)


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(c.isalnum() for c in text) / len(text) * 100, 2)


def _title_in_topk(
        expected_title: str,
        top_matches: list[dict],
        top_k: int,
) -> tuple[bool, Optional[int], float]:
    """
    Check whether expected_job_title appears in the top-k results.

    Returns (found, rank_1based, best_score).
    Matching is fuzzy (token_sort_ratio >= 70) to handle slight wording
    differences like "PHP Developer" vs "PHP / CodeIgniter Developer".

    NOTE: this stays rapidfuzz-based deliberately -- it's resolving a job
    TITLE string against the corpus, the same exact/substring-driven
    resolution approach used in main.py's /target-career-analysis (not the
    skill/cert content matching that was moved to SBERT semantic matching).
    """
    from rapidfuzz import fuzz
    from core.utils import normalize

    exp_norm = normalize(expected_title)
    best_rank  = None
    best_score = 0.0

    for rank, job in enumerate(top_matches[:top_k], start=1):
        title_norm = normalize(job.get("title", ""))
        ratio = fuzz.token_sort_ratio(exp_norm, title_norm)
        if ratio > best_score * 100:
            best_score = ratio / 100.0
        if ratio >= 70 and best_rank is None:
            best_rank = rank

    found = best_rank is not None
    return found, best_rank, round(best_score, 4)


# ----------------------------------------------------------------------------
# Pretty-print helpers
# ----------------------------------------------------------------------------
_SEP  = _c("DIM", "-" * 80)
_SEP2 = _c("DIM", "=" * 80)


def _header(title: str) -> None:
    print()
    print(_SEP2)
    print(_c("BOLD", f"  {title}"))
    print(_SEP2)


def _section(title: str) -> None:
    print()
    print(_SEP)
    print(_c("CYAN", f"  >  {title}"))
    print(_SEP)


def _kv(label: str, value, colour: str = "RESET") -> None:
    print(f"  {_c('DIM', label+':'): <32} {_c(colour, str(value))}")


def _pass_fail(cond: bool, true_label: str = "PASS", false_label: str = "FAIL") -> str:
    return _c("GREEN", f"OK  {true_label}") if cond else _c("RED", f"X  {false_label}")


def _pct(val: Optional[float]) -> str:
    if val is None:
        return _c("DIM", "N/A")
    p = val * 100
    colour = "GREEN" if p >= 75 else ("YELLOW" if p >= 50 else "RED")
    return _c(colour, f"{p:.1f}%")


def _list_preview(items: list[str], colour: str = "RESET", limit: int = 8) -> str:
    shown = items[:limit]
    tail  = f"  (+{len(items)-limit} more)" if len(items) > limit else ""
    return _c(colour, ", ".join(shown)) + _c("DIM", tail) if shown else _c("DIM", "none")


def _print_methodology(mode: str) -> None:
    """
    Print a short, plain-language explanation of what each metric means and
    how it's computed -- so a report is self-documenting rather than just a
    wall of numbers. Shown once per run (via --explain), not per-sample.
    """
    _header("HOW THESE METRICS ARE MEASURED")

    if mode in ("ocr", "pipeline"):
        _section("OCR")
        print(textwrap.dedent("""\
          OCR quality (alnum%)     % of extracted characters that are letters or
                                    digits. A noise heuristic, not an accuracy
                                    score -- needs no ground truth.

          Text similarity (legacy) Single rapidfuzz character-ratio score (0-100%)
                                    against expected_text. Coarse; kept only for
                                    backward compatibility with older reports.

          CER / WER (raw)          Character / Word Error Rate: edit distance from
                                    extracted text to expected_text, divided by
                                    reference length. Computed in reading order --
                                    sensitive to section ordering.

          CER / WER (aligned)      Same computation, but paragraphs are first
                                    greedily reordered to match the reference's
                                    section order before measuring edit distance.
                                    Isolates genuine character misreads from whole
                                    sections coming out in a different order.

          Bag-of-words F1          Word-level precision/recall/F1 with order
                                    ignored entirely. High bow-F1 + low WER means
                                    the gap is an ordering/scrambling artifact, not
                                    a recognition error.

          Section-ordering impact  cer_raw - cer_aligned. Large gap -> ordering is
                                    the dominant error source. Small gap -> genuine
                                    character-level misreads.

          All of the above comparison metrics only run when the label has
          expected_text -- otherwise OCR only reports quality/char-count/time.
        """))

    if mode in ("nlp", "pipeline"):
        _section("NLP (skill + cert extraction)")
        print(textwrap.dedent("""\
          Discovered skills/certs are compared against expected_skills /
          expected_certs -- NOT against any job's requirements. This measures
          extraction quality against a human-annotated answer key, separately
          from how well a resume matches a job (that's the FAISS stage).

          Matching is semantic: SBERT cosine similarity, threshold 0.80 -- the
          SAME matcher function core/scoring.py uses for live scoring, imported
          directly here so this evaluator can't silently drift out of sync with
          what production actually does.

            matched   expected items discovery found        (true positives)
            missed    expected items discovery failed to find (false negatives)
            extra     discovered items with no semantic match
                      among expected items                  (false positives)

          Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean of both.
          Skills and certs are scored independently; either side reports N/A if
          its expected_* label list is missing for that file.
        """))

    if mode in ("faiss", "pipeline"):
        _section("FAISS (job matching)")
        print(textwrap.dedent("""\
          Runs the real calculate_cosine_similarity() from core/scoring.py --
          the same function production uses -- with whichever skills/certs came
          out of the NLP stage (or from the label directly, in --mode faiss
          standalone runs).

          Two separate things get checked:

            1. Title resolution -- is expected_job_title found in the top-k
               results? Matched via rapidfuzz token_sort_ratio (>=70) on the
               title STRING. This is identity resolution (which job is this),
               not skill/cert content matching, so it intentionally stays
               string-fuzzy rather than semantic -- same approach main.py uses
               to resolve a target job title.

            2. Best-match quality -- the top-ranked job's hybrid score, raw
               cosine similarity, and skill/cert match ratios, read directly off
               that job's matched_skills/matched_certs -- the same semantic
               partition production computes, not recomputed here.

          score_pass = best hybrid score >= the label's min_match_score.
        """))

    print()


# ----------------------------------------------------------------------------
# Stage runners
# ----------------------------------------------------------------------------
def run_ocr_stage(file_path: str, label: dict) -> dict:
    """Run Stage 1 OCR and return metrics."""
    from core.extraction import extract_text

    t0 = time.perf_counter()
    text, quality, is_native = extract_text(file_path)
    elapsed = time.perf_counter() - t0

    result = dict(
        text=text,
        quality=quality,
        is_native=is_native,
        char_count=len(text),
        elapsed_s=round(elapsed, 2),
        error=None,
    )

    expected_text = label.get("expected_text")
    if expected_text:
        # Legacy single-number similarity (kept for backward compatibility
        # with any code/reports that still read "text_similarity").
        result["text_similarity"] = _ocr_text_similarity(text, expected_text)

        # Academic-standard CER/WER benchmark, with order-tolerant
        # alignment so OCR extracting sections in a different order than
        # the reference (e.g. "Skills" before "Work Experience") isn't
        # mistaken for a character-recognition error.
        bench = benchmark_ocr(text, expected_text)
        result["benchmark"] = bench.as_dict()
    else:
        result["text_similarity"] = None
        result["benchmark"] = None

    return result


def run_nlp_stage(
        resume_text: str,
        label: dict,
        jobs_corpus: list[dict],
        firestore_skills: list[str],
        firestore_certs: list[str],
        is_native: bool,
) -> dict:
    """Run Stage 2 skill AND cert discovery and return metrics for both."""
    from core.skills import discover_skills_dynamically
    from core.certs  import extract_certificates

    t0 = time.perf_counter()
    skills = discover_skills_dynamically(
        resume_text=resume_text,
        jobs_corpus=jobs_corpus,
        extra_skills=firestore_skills,
        is_native_pdf=is_native,
    )
    certs = extract_certificates(
        resume_text=resume_text,
        jobs_corpus=jobs_corpus,
        extra_certs=firestore_certs if firestore_certs else None,
        is_native_pdf=is_native,
    )
    elapsed = time.perf_counter() - t0

    skill_metrics = _item_metrics(skills, label.get("expected_skills", []))
    cert_metrics  = _item_metrics(certs,  label.get("expected_certs", []))

    # Top-level keys stay skill-shaped for backward compatibility with any
    # existing report consumers; cert_metrics carries the parallel cert view.
    metrics = dict(skill_metrics)
    metrics["skills"] = skills
    metrics["certs"] = certs
    metrics["cert_metrics"] = cert_metrics
    metrics["elapsed_s"] = round(elapsed, 2)
    metrics["error"] = None
    return metrics


_NLP_ERROR_FALLBACK = dict(
    error="no_text", skills=[], certs=[],
    precision=None, recall=None, f1=None, tp=0, fp=0, fn=0,
    matched=[], missed=[], extra=[],
    cert_metrics=dict(precision=None, recall=None, f1=None,
                      matched=[], missed=[], extra=[], tp=0, fp=0, fn=0),
    elapsed_s=0,
)


def run_faiss_stage(
        resume_text: str,
        resume_skills: list[str],
        resume_certs: list[str],
        label: dict,
        jobs_corpus: list[dict],
        top_k: int,
) -> dict:
    """Run Stage 3 FAISS + semantic scoring and return metrics for skills AND certs."""
    from core.scoring import calculate_cosine_similarity

    t0 = time.perf_counter()
    top_matches = calculate_cosine_similarity(
        resume_text=resume_text,
        resume_skills=resume_skills,
        jobs_corpus=jobs_corpus,
        top_k=top_k,
        resume_certs=resume_certs,
    )
    elapsed = time.perf_counter() - t0

    expected_title = label.get("expected_job_title", "")
    min_score      = label.get("min_match_score", 50)

    found, rank, title_similarity = _title_in_topk(expected_title, top_matches, top_k)
    best = top_matches[0] if top_matches else {}

    req_certs     = best.get("required_certifications", [])
    cert_ratio    = (round(len(best.get("matched_certs", [])) / len(req_certs), 4)
                     if req_certs else None)
    cert_accuracy = best.get("nlp_validation_certs", {}).get("NLP_ACCURACY", "N/A")

    return dict(
        top_matches=top_matches,
        best_title=best.get("title", ""),
        best_score=best.get("score", 0.0),
        best_cosine=best.get("cosine_score", 0.0),
        best_match_ratio=best.get("match_ratio", 0.0),
        best_cert_match_ratio=cert_ratio,
        best_cert_accuracy=cert_accuracy,
        expected_title_found=found,
        expected_title_rank=rank,
        title_similarity=title_similarity,
        score_pass=best.get("score", 0.0) >= min_score,
        min_score=min_score,
        elapsed_s=round(elapsed, 2),
        error=None,
    )


_FAISS_ERROR_FALLBACK = dict(
    error="no_input", top_matches=[],
    best_title="", best_score=0.0, best_cosine=0.0,
    best_match_ratio=0.0, best_cert_match_ratio=None, best_cert_accuracy="N/A",
    expected_title_found=False,
    expected_title_rank=None, title_similarity=0.0,
    score_pass=False, min_score=0, elapsed_s=0,
)


# ----------------------------------------------------------------------------
# Per-sample display
# ----------------------------------------------------------------------------
def _display_ocr_alignment_process(name: str, hyp_text: str, ref_text: str) -> None:
    """
    Show the actual block-alignment process that produces the raw vs.
    aligned CER/WER numbers for THIS file -- which extracted paragraph got
    matched to which expected paragraph (and with what score), what was
    left over on either side, and where the concrete word-level mismatches
    are. This is the "how" behind the numbers in _display_ocr_result.
    """
    from rapidfuzz import fuzz

    hyp_blocks = _ocr_split_blocks(hyp_text)
    ref_blocks = _ocr_split_blocks(ref_text)

    _section(f"OCR alignment process  >  {name}")
    _kv("Hypothesis (extracted) blocks", len(hyp_blocks))
    _kv("Reference (expected) blocks",   len(ref_blocks))

    # Recompute the same greedy block matching _ocr_align_blocks() uses
    # internally, but keep the score for each pairing so it can be shown.
    pairs: list[tuple[float, int, int]] = []
    for hi, h in enumerate(hyp_blocks):
        for ri, r in enumerate(ref_blocks):
            pairs.append((fuzz.token_sort_ratio(h, r), hi, ri))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_hyp: set[int] = set()
    used_ref: set[int] = set()
    match_for_ref: dict[int, tuple[int, float]] = {}
    for score, hi, ri in pairs:
        if hi in used_hyp or ri in used_ref:
            continue
        if score < _BLOCK_MATCH_THRESHOLD:
            continue
        match_for_ref[ri] = (hi, score)
        used_hyp.add(hi)
        used_ref.add(ri)

    print(f"\n  {_c('DIM', f'Block-to-block matching (min score {_BLOCK_MATCH_THRESHOLD}%):')}")
    for ri, r in enumerate(ref_blocks):
        ref_preview = textwrap.shorten(r.replace("\n", " "), width=46, placeholder="...")
        if ri in match_for_ref:
            hi, score = match_for_ref[ri]
            hyp_preview = textwrap.shorten(hyp_blocks[hi].replace("\n", " "), width=46, placeholder="...")
            scol = "GREEN" if score >= 80 else ("YELLOW" if score >= _BLOCK_MATCH_THRESHOLD else "RED")
            print(f"    ref[{ri:>2}] {ref_preview:<48} <-({_c(scol, f'{score:3.0f}%')})- hyp[{hi}] {hyp_preview}")
        else:
            print(f"    ref[{ri:>2}] {ref_preview:<48} <- {_c('RED', 'NO MATCH (this whole block counts as missing)')}")

    leftover = [hi for hi in range(len(hyp_blocks)) if hi not in used_hyp]
    if leftover:
        print(f"\n  {_c('YELLOW', 'Unmatched hypothesis blocks (counted as extra/extraneous content):')}")
        for hi in leftover:
            preview = textwrap.shorten(hyp_blocks[hi].replace("\n", " "), width=70, placeholder="...")
            print(f"    hyp[{hi}]  {preview}")

    # Concrete word-level diff -- this is what's actually driving the WER
    # number, independent of the block-ordering question above.
    hyp_words = _ocr_tokenize_words(hyp_text)
    ref_words = _ocr_tokenize_words(ref_text)
    sm = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    diff_lines: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ref_chunk = " ".join(ref_words[i1:i2])
        hyp_chunk = " ".join(hyp_words[j1:j2])
        if tag == "replace":
            diff_lines.append(f"    {_c('RED', ref_chunk)}  ->  {_c('YELLOW', hyp_chunk)}")
        elif tag == "delete":
            diff_lines.append(f"    {_c('RED', ref_chunk)}  ->  {_c('DIM', '(dropped)')}")
        elif tag == "insert":
            diff_lines.append(f"    {_c('DIM', '(nothing expected)')}  ->  {_c('YELLOW', hyp_chunk)}")

    if diff_lines:
        shown = diff_lines[:10]
        print(f"\n  {_c('DIM', f'Word-level diff, expected -> extracted (showing {len(shown)} of {len(diff_lines)} differences):')}")
        for line in shown:
            print(line)
    print()


def _display_ocr_result(name: str, ocr: dict, label: dict, show_process: bool = False) -> None:
    _section(f"OCR  >  {name}")
    colour = "GREEN" if ocr["quality"] >= 70 else ("YELLOW" if ocr["quality"] >= 40 else "RED")
    _kv("Characters extracted", ocr["char_count"])
    _kv("OCR quality (alnum%)",  f"{ocr['quality']:.1f}%", colour)
    _kv("Time",                  f"{ocr['elapsed_s']}s")

    if ocr["text_similarity"] is not None:
        sim   = ocr["text_similarity"]
        scol  = "GREEN" if sim >= 0.85 else ("YELLOW" if sim >= 0.65 else "RED")
        _kv("Text similarity (legacy, vs ground truth)", _pct(sim), scol)

    bench = ocr.get("benchmark")
    if bench is not None:
        def _acc_col(v):
            return "GREEN" if v >= 0.85 else ("YELLOW" if v >= 0.65 else "RED")

        print(f"\n  {_c('DIM', '-- Academic OCR Benchmark (CER/WER) --')}")
        _kv("Character Accuracy (raw, order-sensitive)",
            f"{bench['char_accuracy_raw']*100:.1f}%  (CER {bench['cer_raw']:.4f})",
            _acc_col(bench["char_accuracy_raw"]))
        _kv("Character Accuracy (aligned, order-tolerant)",
            f"{bench['char_accuracy_aligned']*100:.1f}%  (CER {bench['cer_aligned']:.4f})",
            _acc_col(bench["char_accuracy_aligned"]))
        _kv("Word Accuracy (raw, order-sensitive)",
            f"{bench['word_accuracy_raw']*100:.1f}%  (WER {bench['wer_raw']:.4f})",
            _acc_col(bench["word_accuracy_raw"]))
        _kv("Word Accuracy (aligned, order-tolerant)",
            f"{bench['word_accuracy_aligned']*100:.1f}%  (WER {bench['wer_aligned']:.4f})",
            _acc_col(bench["word_accuracy_aligned"]))
        _kv("Bag-of-words F1 (order-invariant sanity check)",
            f"{bench['bow_f1']*100:.1f}%")

        gap = bench["ordering_gap"]
        if gap > 0.05:
            _kv("Section-ordering impact", f"{gap*100:.1f} pts of the raw CER were due to "
                                           "section reordering, not character errors", "YELLOW")

    if ocr["error"]:
        _kv("Error", ocr["error"], "RED")

    # show a short preview
    preview = textwrap.shorten(ocr["text"].replace("\n", " "), width=120, placeholder=" ...")
    print(f"\n  {_c('DIM','Preview:')} {preview}\n")

    # Show the actual block-alignment + word-diff process behind the
    # benchmark numbers above, when there's ground truth to compare against
    # and the caller asked to see it (--explain).
    expected_text = label.get("expected_text")
    if show_process and bench is not None and expected_text:
        _display_ocr_alignment_process(name, ocr["text"], expected_text)


def _display_nlp_result(name: str, nlp: dict, label: dict) -> None:
    _section(f"NLP  >  {name}")
    expected       = label.get("expected_skills", [])
    expected_certs = label.get("expected_certs", [])
    cm             = nlp.get("cert_metrics", {})

    print(f"  {_c('BOLD','-- Skills --')}")
    _kv("Expected skills",  f"{len(expected)}")
    _kv("Predicted skills", f"{len(nlp['skills'])}")
    _kv("True positives",   nlp["tp"],  "GREEN")
    _kv("False positives",  nlp["fp"],  "YELLOW" if nlp["fp"] else "GREEN")
    _kv("False negatives",  nlp["fn"],  "RED"    if nlp["fn"] else "GREEN")
    _kv("Precision",        _pct(nlp["precision"]))
    _kv("Recall",           _pct(nlp["recall"]))
    _kv("F1",               _pct(nlp["f1"]))

    if nlp["matched"]:
        print(f"  {_c('GREEN','  Matched:')}  {_list_preview(nlp['matched'], 'GREEN')}")
    if nlp["missed"]:
        print(f"  {_c('RED','  Missed:')}   {_list_preview(nlp['missed'],  'RED')}")
    if nlp["extra"]:
        print(f"  {_c('YELLOW','  Extra:')}    {_list_preview(nlp['extra'],  'YELLOW')}")

    print(f"\n  {_c('BOLD','-- Certifications --')}")
    if expected_certs:
        _kv("Expected certs",  f"{len(expected_certs)}")
        _kv("Predicted certs", f"{len(nlp.get('certs', []))}")
        _kv("True positives",  cm.get("tp", 0), "GREEN")
        _kv("False positives", cm.get("fp", 0), "YELLOW" if cm.get("fp") else "GREEN")
        _kv("False negatives", cm.get("fn", 0), "RED"    if cm.get("fn") else "GREEN")
        _kv("Precision",       _pct(cm.get("precision")))
        _kv("Recall",          _pct(cm.get("recall")))
        _kv("F1",              _pct(cm.get("f1")))
        if cm.get("matched"):
            print(f"  {_c('GREEN','  Matched:')}  {_list_preview(cm['matched'], 'GREEN')}")
        if cm.get("missed"):
            print(f"  {_c('RED','  Missed:')}   {_list_preview(cm['missed'],  'RED')}")
        if cm.get("extra"):
            print(f"  {_c('YELLOW','  Extra:')}    {_list_preview(cm['extra'],  'YELLOW')}")
    else:
        _kv("Expected certs", _c("DIM", "N/A (no expected_certs label)"))
        _kv("Predicted certs", f"{len(nlp.get('certs', []))}")

    _kv("Time", f"{nlp['elapsed_s']}s")
    print()


def _display_faiss_result(name: str, faiss_r: dict, label: dict, top_k: int) -> None:
    _section(f"FAISS  >  {name}")
    expected_title = label.get("expected_job_title", "N/A")
    score_pct = faiss_r["best_score"]

    _kv("Expected job title",   expected_title)
    _kv("Best matched title",   faiss_r["best_title"],
        "GREEN" if faiss_r["expected_title_found"] else "YELLOW")
    _kv("Hybrid score",         f"{score_pct:.1f} / 100",
        "GREEN" if faiss_r["score_pass"] else "RED")
    _kv("Cosine score (raw)",   f"{faiss_r['best_cosine']:.4f}")
    _kv("Skill match ratio",    _pct(faiss_r["best_match_ratio"]))
    _kv("Cert match ratio",     _pct(faiss_r.get("best_cert_match_ratio")))
    _kv("Cert NLP accuracy",    faiss_r.get("best_cert_accuracy", "N/A"))
    _kv("Min score threshold",  f"{faiss_r['min_score']}")
    _kv("Score pass",           _pass_fail(faiss_r["score_pass"]))

    rank_str = (f"Rank {faiss_r['expected_title_rank']}"
                if faiss_r["expected_title_rank"] else f"Not in top-{top_k}")
    rank_col = ("GREEN" if (faiss_r["expected_title_rank"] or 99) <= 3
                else ("YELLOW" if faiss_r["expected_title_found"] else "RED"))
    _kv(f"Expected title in top-{top_k}", rank_str, rank_col)
    _kv("Title similarity",     _pct(faiss_r["title_similarity"]))
    _kv("Time",                 f"{faiss_r['elapsed_s']}s")

    # show the top-3 ranked jobs inline
    print(f"\n  {_c('DIM', 'Top matches:')}")
    for i, job in enumerate(faiss_r["top_matches"][:min(3, top_k)], 1):
        marker = _c("GREEN", "*") if i == faiss_r["expected_title_rank"] else " "
        score  = job.get("score", 0.0)
        scol   = "GREEN" if score >= 75 else ("YELLOW" if score >= 50 else "RED")
        print(f"    {marker} {i}. {_c(scol, f'{score:5.1f}')}  {job.get('title','')}")
    print()


# ----------------------------------------------------------------------------
# Aggregate summary table
# ----------------------------------------------------------------------------
def _aggregate_summary(mode: str, results: list[dict], top_k: int) -> dict:
    """Compute aggregate stats across all samples and print a summary table."""
    _header(f"AGGREGATE SUMMARY -- mode={mode.upper()}  samples={len(results)}")

    # -- OCR aggregates ------------------------------------------------------
    if mode in ("ocr", "pipeline"):
        ocrs = [r["ocr"] for r in results if r.get("ocr") and not r["ocr"].get("error")]
        if ocrs:
            avg_quality = sum(o["quality"] for o in ocrs) / len(ocrs)
            avg_chars   = sum(o["char_count"] for o in ocrs) / len(ocrs)
            sims        = [o["text_similarity"] for o in ocrs if o["text_similarity"] is not None]
            avg_sim     = sum(sims) / len(sims) if sims else None

            benches = [o["benchmark"] for o in ocrs if o.get("benchmark") is not None]
            if benches:
                avg_char_acc_raw     = sum(b["char_accuracy_raw"]     for b in benches) / len(benches)
                avg_char_acc_aligned = sum(b["char_accuracy_aligned"] for b in benches) / len(benches)
                avg_word_acc_raw     = sum(b["word_accuracy_raw"]     for b in benches) / len(benches)
                avg_word_acc_aligned = sum(b["word_accuracy_aligned"] for b in benches) / len(benches)
                avg_ordering_gap     = sum(b["ordering_gap"]          for b in benches) / len(benches)
                avg_bow_f1           = sum(b["bow_f1"]                for b in benches) / len(benches)
            else:
                avg_char_acc_raw = avg_char_acc_aligned = None
                avg_word_acc_raw = avg_word_acc_aligned = None
                avg_ordering_gap = avg_bow_f1 = None

            _section("OCR Aggregates")
            _kv("Samples evaluated",       len(ocrs))
            _kv("Avg OCR quality (alnum%)", f"{avg_quality:.1f}%",
                "GREEN" if avg_quality >= 70 else "YELLOW")
            _kv("Avg characters extracted", f"{avg_chars:.0f}")
            if avg_sim is not None:
                _kv("Avg text similarity (legacy)",  _pct(avg_sim))

            if benches:
                _kv("Samples with CER/WER benchmark", len(benches))
                _kv("Avg Character Accuracy (raw)",      _pct(avg_char_acc_raw))
                _kv("Avg Character Accuracy (aligned)",  _pct(avg_char_acc_aligned))
                _kv("Avg Word Accuracy (raw)",           _pct(avg_word_acc_raw))
                _kv("Avg Word Accuracy (aligned)",       _pct(avg_word_acc_aligned))
                _kv("Avg Bag-of-words F1",               _pct(avg_bow_f1))
                _kv("Avg section-ordering impact on CER",
                    f"{avg_ordering_gap*100:.1f} pts",
                    "YELLOW" if avg_ordering_gap > 0.05 else "GREEN")

    # -- NLP aggregates (skills + certs) -------------------------------------
    if mode in ("nlp", "pipeline"):
        nlps = [r["nlp"] for r in results if r.get("nlp") and not r["nlp"].get("error")]
        nlps_with_expected = [n for n in nlps if n.get("precision") is not None]

        if nlps_with_expected:
            avg_p  = sum(n["precision"] for n in nlps_with_expected) / len(nlps_with_expected)
            avg_r  = sum(n["recall"]    for n in nlps_with_expected) / len(nlps_with_expected)
            avg_f1 = sum(n["f1"]        for n in nlps_with_expected) / len(nlps_with_expected)
            total_tp = sum(n["tp"] for n in nlps_with_expected)
            total_fp = sum(n["fp"] for n in nlps_with_expected)
            total_fn = sum(n["fn"] for n in nlps_with_expected)

            _section("NLP (Skill Extraction) Aggregates")
            _kv("Samples with expected skills", len(nlps_with_expected))
            _kv("Avg Precision",  _pct(avg_p))
            _kv("Avg Recall",     _pct(avg_r))
            _kv("Avg F1",         _pct(avg_f1))
            _kv("Total TP / FP / FN", f"{total_tp} / {total_fp} / {total_fn}")

            # Per-sample mini table
            print()
            hdr = f"  {'File':<35} {'P':>6} {'R':>6} {'F1':>6}  {'TP':>3} {'FP':>3} {'FN':>3}"
            print(_c("BOLD", hdr))
            print(_c("DIM", "  " + "-" * 70))
            for r in results:
                n = r.get("nlp", {})
                if n.get("precision") is None:
                    continue
                name = Path(r["file"]).name[:34]
                p_col = "GREEN" if n["precision"] >= 0.75 else ("YELLOW" if n["precision"] >= 0.50 else "RED")
                r_col = "GREEN" if n["recall"]    >= 0.75 else ("YELLOW" if n["recall"]    >= 0.50 else "RED")
                f_col = "GREEN" if n["f1"]        >= 0.75 else ("YELLOW" if n["f1"]        >= 0.50 else "RED")
                p_str = f"{n['precision']*100:5.1f}%"
                r_str = f"{n['recall']*100:5.1f}%"
                f_str = f"{n['f1']*100:5.1f}%"
                print(
                    f"  {name:<35}"
                    f" {_c(p_col, p_str)}"
                    f" {_c(r_col, r_str)}"
                    f" {_c(f_col, f_str)}"
                    f"  {n['tp']:>3} {n['fp']:>3} {n['fn']:>3}"
                )

        cert_metrics_list = [r["nlp"]["cert_metrics"] for r in results
                             if r.get("nlp") and not r["nlp"].get("error")
                             and r["nlp"].get("cert_metrics", {}).get("precision") is not None]
        if cert_metrics_list:
            avg_cp  = sum(c["precision"] for c in cert_metrics_list) / len(cert_metrics_list)
            avg_cr  = sum(c["recall"]    for c in cert_metrics_list) / len(cert_metrics_list)
            avg_cf1 = sum(c["f1"]        for c in cert_metrics_list) / len(cert_metrics_list)
            total_ctp = sum(c["tp"] for c in cert_metrics_list)
            total_cfp = sum(c["fp"] for c in cert_metrics_list)
            total_cfn = sum(c["fn"] for c in cert_metrics_list)

            _section("NLP (Certification Extraction) Aggregates")
            _kv("Samples with expected certs", len(cert_metrics_list))
            _kv("Avg Precision",  _pct(avg_cp))
            _kv("Avg Recall",     _pct(avg_cr))
            _kv("Avg F1",         _pct(avg_cf1))
            _kv("Total TP / FP / FN", f"{total_ctp} / {total_cfp} / {total_cfn}")

    # -- FAISS aggregates (skills + certs) -----------------------------------
    if mode in ("faiss", "pipeline"):
        faiss_rs = [r["faiss"] for r in results if r.get("faiss") and not r["faiss"].get("error")]

        if faiss_rs:
            n_found     = sum(1 for f in faiss_rs if f["expected_title_found"])
            n_pass      = sum(1 for f in faiss_rs if f["score_pass"])
            avg_score   = sum(f["best_score"] for f in faiss_rs) / len(faiss_rs)
            avg_cosine  = sum(f["best_cosine"] for f in faiss_rs) / len(faiss_rs)
            rank_1      = sum(1 for f in faiss_rs if f.get("expected_title_rank") == 1)
            rank_top3   = sum(1 for f in faiss_rs if (f.get("expected_title_rank") or 99) <= 3)

            cert_ratios = [f["best_cert_match_ratio"] for f in faiss_rs
                           if f.get("best_cert_match_ratio") is not None]
            avg_cert_ratio = sum(cert_ratios) / len(cert_ratios) if cert_ratios else None

            _section(f"FAISS (Top-{top_k} Job Matching) Aggregates")
            _kv("Samples evaluated",                  len(faiss_rs))
            _kv(f"Expected title in top-{top_k}",
                f"{n_found}/{len(faiss_rs)}  ({n_found/len(faiss_rs)*100:.0f}%)",
                "GREEN" if n_found/len(faiss_rs) >= 0.8 else "YELLOW")
            _kv("Expected title ranked #1",           f"{rank_1}/{len(faiss_rs)}")
            _kv("Expected title ranked top-3",        f"{rank_top3}/{len(faiss_rs)}")
            _kv("Score >= min_match_score",
                f"{n_pass}/{len(faiss_rs)}  ({n_pass/len(faiss_rs)*100:.0f}%)",
                "GREEN" if n_pass/len(faiss_rs) >= 0.8 else ("YELLOW" if n_pass/len(faiss_rs) >= 0.5 else "RED"))
            _kv("Avg hybrid score",    f"{avg_score:.1f} / 100",
                "GREEN" if avg_score >= 70 else ("YELLOW" if avg_score >= 50 else "RED"))
            _kv("Avg cosine (raw)",    f"{avg_cosine:.4f}")
            if avg_cert_ratio is not None:
                _kv("Avg cert match ratio (best job)", _pct(avg_cert_ratio))
            else:
                _kv("Avg cert match ratio (best job)", _c("DIM", "N/A (no jobs with required_certifications)"))

            # Per-sample mini table
            print()
            hdr = f"  {'File':<35} {'Score':>7} {'Cos':>7} {'Rank':>6}  {'Title Found':<14} Result"
            print(_c("BOLD", hdr))
            print(_c("DIM", "  " + "-" * 80))
            for r in results:
                f = r.get("faiss", {})
                if not f or f.get("error"):
                    continue
                name   = Path(r["file"]).name[:34]
                score  = f["best_score"]
                scol   = "GREEN" if score >= 75 else ("YELLOW" if score >= 50 else "RED")
                rank   = str(f["expected_title_rank"]) if f["expected_title_rank"] else f">{top_k}"
                found  = _c("GREEN", "found") if f["expected_title_found"] else _c("RED", "missing")
                result = _pass_fail(f["score_pass"])
                print(
                    f"  {name:<35}"
                    f" {_c(scol, f'{score:6.1f}')}"
                    f" {f['best_cosine']:7.4f}"
                    f" {rank:>6}"
                    f"  {found:<22}"
                    f" {result}"
                )

    # -- collect aggregates for JSON export ----------------------------------
    agg: dict = {}
    if mode in ("ocr", "pipeline") and "ocrs" in dir():
        agg["ocr"] = dict(
            samples=len(ocrs),
            avg_quality=round(avg_quality, 2),
            avg_chars=round(avg_chars, 0),
            avg_text_similarity=round(avg_sim, 4) if avg_sim is not None else None,
            benchmark_samples=len(benches) if benches else 0,
            avg_char_accuracy_raw=round(avg_char_acc_raw, 4) if avg_char_acc_raw is not None else None,
            avg_char_accuracy_aligned=round(avg_char_acc_aligned, 4) if avg_char_acc_aligned is not None else None,
            avg_word_accuracy_raw=round(avg_word_acc_raw, 4) if avg_word_acc_raw is not None else None,
            avg_word_accuracy_aligned=round(avg_word_acc_aligned, 4) if avg_word_acc_aligned is not None else None,
            avg_bow_f1=round(avg_bow_f1, 4) if avg_bow_f1 is not None else None,
            avg_ordering_gap=round(avg_ordering_gap, 4) if avg_ordering_gap is not None else None,
        )
    if mode in ("nlp", "pipeline") and "nlps_with_expected" in dir() and nlps_with_expected:
        agg["nlp"] = dict(
            samples=len(nlps_with_expected),
            avg_precision=round(avg_p, 4),
            avg_recall=round(avg_r, 4),
            avg_f1=round(avg_f1, 4),
            total_tp=total_tp, total_fp=total_fp, total_fn=total_fn,
        )
    if mode in ("nlp", "pipeline") and "cert_metrics_list" in dir() and cert_metrics_list:
        agg["nlp_certs"] = dict(
            samples=len(cert_metrics_list),
            avg_precision=round(avg_cp, 4),
            avg_recall=round(avg_cr, 4),
            avg_f1=round(avg_cf1, 4),
            total_tp=total_ctp, total_fp=total_cfp, total_fn=total_cfn,
        )
    if mode in ("faiss", "pipeline") and "faiss_rs" in dir() and faiss_rs:
        agg["faiss"] = dict(
            samples=len(faiss_rs),
            title_found_rate=round(n_found / len(faiss_rs), 4),
            rank_1_rate=round(rank_1 / len(faiss_rs), 4),
            rank_top3_rate=round(rank_top3 / len(faiss_rs), 4),
            score_pass_rate=round(n_pass / len(faiss_rs), 4),
            avg_hybrid_score=round(avg_score, 2),
            avg_cosine=round(avg_cosine, 4),
            avg_cert_match_ratio=round(avg_cert_ratio, 4) if avg_cert_ratio is not None else None,
        )
    return agg


# ----------------------------------------------------------------------------
# Main evaluation runner
# ----------------------------------------------------------------------------
def evaluate(
        labels_path: str,
        samples_pdf: Optional[str] = None,
        samples_img: Optional[str] = None,
        mode: str = "pipeline",
        file_type: str = "all",   # "all" | "pdf" | "img"
        top_k: int = 5,
        output_path: Optional[str] = None,
        verbose: bool = True,
        explain: bool = False,
        charts: bool = False,
        charts_dir: str = ".",
        charts_png: bool = False,
        charts_prefix: str = "eval",
) -> dict:
    # Load labels
    labels_file = Path(labels_path)
    if not labels_file.exists():
        log.error(f"Labels file not found: {labels_path}")
        sys.exit(1)
    with labels_file.open() as f:
        labels: dict = json.load(f)

    _PDF_EXTS = {".pdf"}
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

    # Validate only the dirs that were actually provided
    dir_pdf = Path(samples_pdf) if samples_pdf else None
    dir_img = Path(samples_img) if samples_img else None

    for label, d in (("--samples-pdf", dir_pdf), ("--samples-img", dir_img)):
        if d is not None and not d.exists():
            log.error(f"Samples directory not found ({label}): {d}")
            sys.exit(1)

    if file_type not in ("all", "pdf", "img"):
        log.error(f"--type must be one of: all, pdf, img  (got '{file_type}')")
        sys.exit(1)

    # Warn if a dir is needed but wasn't supplied
    want_pdf = file_type in ("all", "pdf")
    want_img = file_type in ("all", "img")
    if want_pdf and dir_pdf is None:
        log.warning("PDF labels exist but --samples-pdf was not set; PDF files will be skipped.")
    if want_img and dir_img is None:
        log.warning("Image labels exist but --samples-img was not set; image files will be skipped.")

    # Resolve each labelled filename to its directory by extension
    sample_files: list[Path] = []
    skipped_type: list[str]  = []

    for name in labels:
        ext = Path(name).suffix.lower()
        is_pdf = ext in _PDF_EXTS
        is_img = ext in _IMG_EXTS

        # Skip types excluded by --type flag or missing dir
        if is_pdf and (not want_pdf or dir_pdf is None):
            skipped_type.append(name)
            continue
        if is_img and (not want_img or dir_img is None):
            skipped_type.append(name)
            continue

        if is_pdf:
            candidate = dir_pdf / name
        elif is_img:
            candidate = dir_img / name
        else:
            # Unknown extension -- try whichever dirs are available
            candidate = None
            for d in filter(None, (dir_pdf, dir_img)):
                if (d / name).exists():
                    candidate = d / name
                    break
            if candidate is None:
                log.warning(f"Unknown extension and file not found anywhere, skipping: {name}")
                continue

        if candidate.exists():
            sample_files.append(candidate)
        else:
            log.warning(f"Sample file not found, skipping: {candidate}")

    if skipped_type:
        log.info(f"Skipped {len(skipped_type)} file(s) excluded by --type={file_type}: "
                 + ", ".join(skipped_type[:5]) + ("..." if len(skipped_type) > 5 else ""))

    if not sample_files:
        log.error(
            f"No sample files could be resolved (--type={file_type}). "
            + (f"PDFs looked in: {dir_pdf}  " if dir_pdf else "")
            + (f"Images looked in: {dir_img}" if dir_img else "")
        )
        sys.exit(1)

    _header(
        f"PIPELINE EVALUATOR  --  mode={mode.upper()}  "
        f"samples={len(sample_files)}  top_k={top_k}"
    )
    print(f"  Labels      : {labels_path}")
    if dir_pdf:
        print(f"  Samples PDF : {dir_pdf}")
    if dir_img:
        print(f"  Samples IMG : {dir_img}")
    if file_type != "all":
        print(f"  Type filter : {_c('YELLOW', file_type.upper() + ' only')}")
    print(f"  Mode    : {_c('CYAN', mode.upper())}")
    print(f"  Top-K   : {top_k}")
    print()

    # Load corpus once (if NLP or FAISS stages are needed)
    corpus_needed = mode in ("nlp", "faiss", "pipeline")
    if corpus_needed:
        firestore_skills, jobs_corpus, firestore_certs = _load_corpus()
        log.info(f"Corpus loaded: {len(jobs_corpus)} jobs, {len(firestore_skills)} skills")

    all_results: list[dict] = []

    for file_path in sample_files:
        name  = file_path.name
        label = labels.get(name, {})

        print(_c("BOLD", f"\n{'='*80}"))
        print(_c("BOLD", f"  FILE: {name}"))
        print(_c("BOLD", f"{'='*80}"))

        result: dict = {"file": str(file_path), "label": name}
        elapsed_total = 0.0

        # -- Stage 1 : OCR ----------------------------------------------------
        if mode in ("ocr", "pipeline"):
            try:
                ocr_result = run_ocr_stage(str(file_path), label)
                result["ocr"] = ocr_result
                elapsed_total += ocr_result["elapsed_s"]
                if verbose:
                    _display_ocr_result(name, ocr_result, label)
            except Exception as exc:
                log.exception(f"[OCR] Error on {name}")
                result["ocr"] = {"error": str(exc), "text": "", "quality": 0.0,
                                 "char_count": 0, "elapsed_s": 0, "is_native": False,
                                 "text_similarity": None, "benchmark": None}

        # -- Stage 2 : NLP (skills + certs) -----------------------------------
        if mode in ("nlp", "pipeline"):
            resume_text = result.get("ocr", {}).get("text", "")
            is_native   = result.get("ocr", {}).get("is_native", False)

            # If mode == "nlp" only (no OCR stage ran), try to get text from label
            if mode == "nlp" and not resume_text:
                resume_text = label.get("expected_text", "")
                if not resume_text:
                    log.warning(
                        f"[NLP] No resume text available for {name} "
                        "(run with --mode pipeline or add expected_text to label)"
                    )

            if resume_text:
                try:
                    nlp_result = run_nlp_stage(
                        resume_text=resume_text,
                        label=label,
                        jobs_corpus=jobs_corpus,
                        firestore_skills=firestore_skills,
                        firestore_certs=firestore_certs,
                        is_native=is_native,
                    )
                    result["nlp"] = nlp_result
                    elapsed_total += nlp_result["elapsed_s"]
                    if verbose:
                        _display_nlp_result(name, nlp_result, label)
                except Exception as exc:
                    log.exception(f"[NLP] Error on {name}")
                    result["nlp"] = {**_NLP_ERROR_FALLBACK, "error": str(exc)}
            else:
                result["nlp"] = dict(_NLP_ERROR_FALLBACK)

        # -- Stage 3 : FAISS (skills + certs) ---------------------------------
        if mode in ("faiss", "pipeline"):
            resume_text   = result.get("ocr", {}).get("text", "")
            resume_skills = result.get("nlp", {}).get("skills", [])
            resume_certs  = result.get("nlp", {}).get("certs", [])

            if mode == "faiss" and not resume_text:
                resume_text   = label.get("expected_text", "")
                resume_skills = label.get("expected_skills", [])
                resume_certs  = label.get("expected_certs", [])
                log.info(f"[FAISS-only] Using label skills/certs for {name}")

            if resume_text or resume_skills or resume_certs:
                try:
                    faiss_result = run_faiss_stage(
                        resume_text=resume_text,
                        resume_skills=resume_skills,
                        resume_certs=resume_certs,
                        label=label,
                        jobs_corpus=jobs_corpus,
                        top_k=top_k,
                    )
                    result["faiss"] = faiss_result
                    elapsed_total += faiss_result["elapsed_s"]
                    if verbose:
                        _display_faiss_result(name, faiss_result, label, top_k)
                except Exception as exc:
                    log.exception(f"[FAISS] Error on {name}")
                    result["faiss"] = {**_FAISS_ERROR_FALLBACK, "error": str(exc)}
            else:
                result["faiss"] = dict(_FAISS_ERROR_FALLBACK)

        result["elapsed_total_s"] = round(elapsed_total, 2)
        all_results.append(result)
        print(f"\n  {_c('DIM','Total time for this file:')} {elapsed_total:.2f}s")

    # -- Aggregate summary -----------------------------------------------------
    aggregates = _aggregate_summary(mode, all_results, top_k)

    # -- JSON export -------------------------------------------------------------
    report = {
        "mode": mode,
        "top_k": top_k,
        "samples": len(all_results),
        "aggregates": aggregates,
        "results": [
            {
                "file": r["file"],
                "label": r["label"],
                "elapsed_total_s": r["elapsed_total_s"],
                # Strip bulk text/embeddings to keep report concise
                "ocr": {k: v for k, v in r.get("ocr", {}).items()
                        if k not in ("text",)} if r.get("ocr") else None,
                "nlp": {k: v for k, v in r.get("nlp", {}).items()
                        if k not in ("skills", "certs")} if r.get("nlp") else None,
                "faiss": {k: v for k, v in r.get("faiss", {}).items()
                          if k not in ("top_matches",)} if r.get("faiss") else None,
            }
            for r in all_results
        ],
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  {_c('GREEN','OK')}  Report saved -> {output_path}\n")

    # -- Chart generation --------------------------------------------------------
    if charts:
        try:
            from charts import generate_charts
            chart_files = generate_charts(
                report,
                output_dir=charts_dir,
                prefix=charts_prefix,
                save_png=charts_png,
            )
            for cf in chart_files:
                print(f"  {_c('GREEN','OK')}  Chart saved -> {cf}")
        except Exception as _chart_exc:
            log.warning(f"Chart generation failed: {_chart_exc}")

    print()
    print(_SEP2)
    print(_c("BOLD", f"  Done. {len(all_results)} file(s) evaluated."))
    print(_SEP2)
    print()

    return report


# ----------------------------------------------------------------------------
# Standalone report summarizer  (reads a saved report.json, prints compact text)
# ----------------------------------------------------------------------------
def summarize_report(report_path: str) -> None:
    """
    Load a previously saved report.json and print a short, compact text
    summary -- aggregates only, no per-sample dump. Mirrors the aggregate
    block shown during a live run but works standalone against a saved file.
    """
    path_obj = Path(report_path)
    if not path_obj.exists():
        log.error(f"Report file not found: {report_path}")
        sys.exit(1)

    with path_obj.open() as f:
        report = json.load(f)

    mode    = report.get("mode", "unknown")
    top_k   = report.get("top_k", 5)
    samples = report.get("samples", 0)
    agg     = report.get("aggregates", {})
    results = report.get("results", [])

    _header(f"REPORT SUMMARY  --  {path_obj.name}")
    print(f"  Mode      : {_c('CYAN', mode.upper())}")
    print(f"  Samples   : {samples}")
    print(f"  Top-K     : {top_k}")

    if not agg:
        print(f"\n  {_c('YELLOW','No aggregate data found in this report.')}")
        return

    # -- OCR ------------------------------------------------------------------
    if "ocr" in agg:
        o = agg["ocr"]
        _section("OCR Summary")
        _kv("Samples with OCR",         o.get("samples", 0))
        _kv("Avg OCR quality (alnum%)", f"{o.get('avg_quality', 0):.1f}%",
            "GREEN" if o.get("avg_quality", 0) >= 70 else "YELLOW")
        _kv("Avg characters extracted", f"{o.get('avg_chars', 0):.0f}")
        sim = o.get("avg_text_similarity")
        if sim is not None:
            _kv("Avg text similarity (legacy)",  _pct(sim))
        else:
            _kv("Avg text similarity (legacy)",  _c("DIM", "N/A (no expected_text labels)"))

        if o.get("benchmark_samples"):
            _kv("Samples with CER/WER benchmark", o.get("benchmark_samples", 0))
            _kv("Avg Character Accuracy (raw)",     _pct(o.get("avg_char_accuracy_raw")))
            _kv("Avg Character Accuracy (aligned)", _pct(o.get("avg_char_accuracy_aligned")))
            _kv("Avg Word Accuracy (raw)",          _pct(o.get("avg_word_accuracy_raw")))
            _kv("Avg Word Accuracy (aligned)",      _pct(o.get("avg_word_accuracy_aligned")))
            _kv("Avg Bag-of-words F1",              _pct(o.get("avg_bow_f1")))
            gap = o.get("avg_ordering_gap")
            if gap is not None:
                _kv("Avg section-ordering impact on CER", f"{gap*100:.1f} pts",
                    "YELLOW" if gap > 0.05 else "GREEN")

    # -- NLP (skills) -----------------------------------------------------------
    if "nlp" in agg:
        n = agg["nlp"]
        total_expected = n.get("total_tp", 0) + n.get("total_fn", 0)
        _section("NLP (Skill Extraction) Summary")
        _kv("Samples with expected skills", n.get("samples", 0))
        _kv("Expected skills",              total_expected)
        _kv("Avg Precision",                _pct(n.get("avg_precision")))
        _kv("Avg Recall",                   _pct(n.get("avg_recall")))
        _kv("Avg F1",                       _pct(n.get("avg_f1")))
        _kv("Total TP / FP / FN",
            f"{n.get('total_tp', 0)} / {n.get('total_fp', 0)} / {n.get('total_fn', 0)}")

    # -- NLP (certs) --------------------------------------------------------------
    if "nlp_certs" in agg:
        nc = agg["nlp_certs"]
        total_expected_c = nc.get("total_tp", 0) + nc.get("total_fn", 0)
        _section("NLP (Certification Extraction) Summary")
        _kv("Samples with expected certs", nc.get("samples", 0))
        _kv("Expected certs",               total_expected_c)
        _kv("Avg Precision",                _pct(nc.get("avg_precision")))
        _kv("Avg Recall",                   _pct(nc.get("avg_recall")))
        _kv("Avg F1",                       _pct(nc.get("avg_f1")))
        _kv("Total TP / FP / FN",
            f"{nc.get('total_tp', 0)} / {nc.get('total_fp', 0)} / {nc.get('total_fn', 0)}")

    # -- FAISS --------------------------------------------------------------------
    if "faiss" in agg:
        fa = agg["faiss"]
        n_samples = fa.get("samples", 0)
        _section(f"FAISS (Top-{top_k} Job Matching) Summary")
        _kv("Samples evaluated",            n_samples)
        _kv(f"Expected title in top-{top_k}",
            f"{fa.get('title_found_rate', 0)*100:.0f}%",
            "GREEN" if fa.get("title_found_rate", 0) >= 0.8 else "YELLOW")
        _kv("Expected title ranked #1",     f"{fa.get('rank_1_rate', 0)*100:.0f}%")
        _kv("Expected title ranked top-3",  f"{fa.get('rank_top3_rate', 0)*100:.0f}%")
        _kv("Score >= min_match_score",
            f"{fa.get('score_pass_rate', 0)*100:.0f}%",
            "GREEN" if fa.get("score_pass_rate", 0) >= 0.8 else
            ("YELLOW" if fa.get("score_pass_rate", 0) >= 0.5 else "RED"))
        _kv("Avg hybrid score",  f"{fa.get('avg_hybrid_score', 0):.1f} / 100")
        _kv("Avg cosine (raw)",  f"{fa.get('avg_cosine', 0):.4f}")
        cert_ratio = fa.get("avg_cert_match_ratio")
        if cert_ratio is not None:
            _kv("Avg cert match ratio (best job)", _pct(cert_ratio))

    # -- Timing (derived from results, since aggregates doesn't store it) -------
    if results:
        times = [r.get("elapsed_total_s", 0) for r in results if r.get("elapsed_total_s") is not None]
        if times:
            _section("Timing Summary")
            _kv("Total time (all samples)", f"{sum(times):.1f}s")
            _kv("Avg time per sample",       f"{sum(times)/len(times):.2f}s")
            _kv("Slowest sample",            f"{max(times):.2f}s")

    print()
    print(_SEP2)
    print()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate OCR -> NLP -> FAISS pipeline accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        examples:
          # Run both types (PDFs + images) together
          python evaluate_pipeline.py --labels labels.json --samples-pdf ./resumes/ --samples-img ./resumes-images/

          # PDFs only -- won't touch Firestore for image-labelled files
          python evaluate_pipeline.py --labels labels.json --samples-pdf ./resumes/ --type pdf

          # Images only
          python evaluate_pipeline.py --labels labels.json --samples-img ./resumes-images/ --type img

          # NLP-only pass on images
          python evaluate_pipeline.py --labels labels.json --samples-img ./resumes-images/ --type img --mode nlp

          # PDF run with report + charts
          python evaluate_pipeline.py --labels labels.json --samples-pdf ./resumes/ --type pdf --output report_pdf.json --charts

          # Print a compact summary from a previously saved report.json (no re-run needed)
          python evaluate_pipeline.py --summarize report_pdf.json
        """),
    )
    p.add_argument("--summarize", metavar="REPORT_JSON", default=None,
                   help="Print a compact summary from a saved report.json and exit "
                        "(skips running the pipeline entirely)")
    p.add_argument("--labels",      default=None, help="Path to labels.json")
    p.add_argument("--samples-pdf", default=None,
                   help="Directory containing PDF resumes")
    p.add_argument("--samples-img", default=None,
                   help="Directory containing image resumes (JPG/PNG/etc.)")
    p.add_argument("--type", dest="file_type", default="all",
                   choices=["all", "pdf", "img"],
                   help="Limit evaluation to one file type to avoid overloading Firestore (default: all)")
    p.add_argument("--mode", default="pipeline",
                   choices=["ocr", "nlp", "faiss", "pipeline"],
                   help="Which stage(s) to evaluate (default: pipeline)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of FAISS results to consider (default: 5)")
    p.add_argument("--output", default=None,
                   help="Optional path to write a JSON report")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-sample verbose output; show only the summary")
    p.add_argument("--charts", action="store_true",
                   help="Generate charts PDF (requires matplotlib)")
    p.add_argument("--charts-dir", default=".",
                   help="Directory to write charts into (default: current dir)")
    p.add_argument("--charts-png", action="store_true",
                   help="Also save individual PNG files alongside the PDF")
    p.add_argument("--charts-prefix", default="eval",
                   help="Filename prefix for chart files (default: eval)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.summarize:
        summarize_report(args.summarize)
        sys.exit(0)

    if not args.labels:
        log.error("--labels is required unless using --summarize")
        sys.exit(1)

    evaluate(
        labels_path=args.labels,
        samples_pdf=args.samples_pdf,
        samples_img=args.samples_img,
        file_type=args.file_type,
        mode=args.mode,
        top_k=args.top_k,
        output_path=args.output,
        verbose=not args.quiet,
        charts=args.charts,
        charts_dir=args.charts_dir,
        charts_png=args.charts_png,
        charts_prefix=args.charts_prefix,
    )