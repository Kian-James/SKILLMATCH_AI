"""
evaluate_pipeline.py
====================
Offline evaluation harness for the full resume-analysis pipeline.

Usage
-----
    python evaluate_pipeline.py                          # auto-detect resumes/ folder
    python evaluate_pipeline.py --resumes resumes/      # explicit folder
    python evaluate_pipeline.py --resumes resumes/Aaron_Sterling.pdf  # single PDF
    python evaluate_pipeline.py --check                 # coverage audit only
    python evaluate_pipeline.py --resumes resumes/ --limit 20
    python evaluate_pipeline.py --resumes resumes/ --out report.json
    python evaluate_pipeline.py --resumes resumes/ --quiet
    python evaluate_pipeline.py --resumes resumes/ --charts            # generate charts
    python evaluate_pipeline.py --resumes resumes/ --charts --charts-dir my_charts/
    python evaluate_pipeline.py --out report.json --charts-only        # charts from saved report

Ground Truth  (labels.json)
----------------------------
Each entry is keyed by the original PDF filename:

    {
        "expected_skills":       ["Python", "SQL", ...],
        "expected_job_title":    "Data Scientist",      # used for title-sim AND FAISS rank check
        "min_match_score":       50,                    # hybrid score threshold (0–100)
        "expected_certs":        ["AWS CCP"],           # optional
        "expected_text_source":  "native_pdf"           # optional: "native_pdf" | "ocr" | "camera_scan_mlkit"
    }

Metrics produced
----------------
OCR
  ocr_quality_pct     — pipeline's alnum-ratio for the extracted text (0–100)
  text_source_correct — 1/0 whether the right extraction path was used
                        (only scored when expected_text_source is in the label)

NLP  (skill extraction quality — your ground truth vs pipeline output)
  skill_recall        — TP / len(expected_skills)
  skill_precision     — TP / len(discovered_skills)
  skill_f1            — harmonic mean of the above

  nlp_pipeline_accuracy — the pipeline's own self-reported NLP_ACCURACY figure
                          (fuzzy match of discovered skills vs best-job's required
                          skills list).  Reported separately so you can see whether
                          the system's self-assessment tracks your ground truth F1.

FAISS  (ranking quality)
  faiss_top1_match    — 1 if expected_job_title ranked #1 (title similarity ≥ threshold)
  faiss_top5_match    — 1 if expected_job_title appeared anywhere in top-5
  faiss_best_rank     — rank position 1–5, or 6 when not found in top-5
  faiss_cosine_score  — SBERT cosine score for the best-matched job
  faiss_match_ratio   — skill match ratio for the best-matched job
  best_hybrid_score   — hybrid score (0–100) of the #1 ranked job
  score_pass          — 1 if best_hybrid_score >= min_match_score

Charts (--charts flag)
----------------------
Requires matplotlib. Produces 10 diagnostic PNGs in eval_charts/ (or --charts-dir):
  01_skill_f1_distribution.png
  02_recall_vs_precision.png
  03_per_resume_f1.png
  04_missing_skills_freq.png
  05_extra_skills_freq.png
  06_nlp_accuracy_dist.png
  07_score_distribution.png
  08_f1_by_group.png
  09_title_pass_fail.png
  10_recall_precision_gap.png
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

try:
    from rapidfuzz import fuzz as _rfuzz
    def _title_similarity(a: str, b: str) -> float:
        return _rfuzz.token_sort_ratio(a.lower(), b.lower()) / 100.0
except ImportError:
    def _title_similarity(a: str, b: str) -> float:
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

from core.utils import normalize

log = logging.getLogger("evaluate_pipeline")

_PIPELINE_LOGGERS = [
    "db.client", "db.jobs", "db.results",
    "core.extraction", "core.skills", "core.certs", "core.scoring",
    "pipeline", "easyocr", "easyocr.easyocr",
]

SCRIPT_DIR            = Path(__file__).parent
LABELS_PATH           = SCRIPT_DIR / "labels-2.json"
DEFAULT_RESUMES_DIR   = SCRIPT_DIR / "resumes-images"
DEFAULT_CHARTS_DIR    = SCRIPT_DIR / "eval_charts"
TITLE_MATCH_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Label / resume loading
# ---------------------------------------------------------------------------

def load_labels(path: Path = LABELS_PATH) -> dict:
    if not path.exists():
        log.error("labels.json not found at %s", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_resumes(source: Optional[str] = None) -> list[tuple[str, str]]:
    source_path = Path(source) if source else DEFAULT_RESUMES_DIR
    if not source_path.exists():
        log.error("Source path does not exist: %s", source_path)
        sys.exit(1)
    if source_path.is_file():
        if source_path.suffix.lower() not in (".pdf", ".jpg", ".jpeg", ".png"):
            log.error("Unsupported file type: %s", source_path.name)
            sys.exit(1)
        return [(source_path.name, str(source_path))]
    if source_path.is_dir():
        files = sorted(
            [p for p in source_path.iterdir()
             if p.suffix.lower() in (".pdf", ".jpg", ".jpeg", ".png")]
        )
        if not files:
            log.error("No resume files found in %s", source_path)
            sys.exit(1)
        log.info("Found %d file(s) in %s", len(files), source_path)
        return [(p.name, str(p)) for p in files]
    log.error("Unsupported source: %s", source_path)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Skill matching helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def _skills_match(a: str, b: str) -> bool:
    from rapidfuzz import fuzz as _fuzz
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return _fuzz.token_sort_ratio(na, nb) >= 85


def _match_skill_sets(
        expected:   list[str],
        discovered: list[str],
) -> tuple[set[str], set[str], set[str]]:
    matched_exp  = set()
    matched_disc = set()
    for exp in expected:
        for disc in discovered:
            if _skills_match(exp, disc):
                matched_exp.add(exp)
                matched_disc.add(disc)
                break
    missing_exp = set(expected)   - matched_exp
    extra_disc  = set(discovered) - matched_disc
    return matched_exp, missing_exp, extra_disc


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

def check_coverage(resumes_source: Optional[str] = None) -> dict:
    labels      = load_labels()
    all_resumes = collect_resumes(resumes_source)
    label_names  = set(labels.keys())
    folder_names = {fname for fname, _ in all_resumes}
    matched   = sorted(label_names & folder_names)
    missing   = sorted(label_names - folder_names)
    unlabeled = sorted(folder_names - label_names)
    sep = "-" * 62
    print("\n" + "=" * 62)
    print("  COVERAGE AUDIT")
    print("=" * 62)
    print(f"  labels.json entries : {len(label_names)}")
    print(f"  Files in folder     : {len(folder_names)}")
    print(f"  Matched (runnable)  : {len(matched)}")
    print(f"  Missing from folder : {len(missing)}")
    print(f"  Unlabeled files     : {len(unlabeled)}")
    print()
    if matched:
        print(f"  [OK] MATCHED ({len(matched)}) — will be evaluated:")
        print(f"  {sep}")
        for name in matched:
            label      = labels[name]
            certs_note = (f"  +{len(label.get('expected_certs', []))} certs"
                          if label.get("expected_certs") else "")
            src_note   = (f"  src={label['expected_text_source']}"
                          if label.get("expected_text_source") else "")
            print(f"    {name:<50}  {len(label.get('expected_skills', []))} skills{certs_note}{src_note}")
        print()
    if missing:
        print(f"  [!] MISSING from folder ({len(missing)}) — skipped:")
        print(f"  {sep}")
        for name in missing:
            print(f"    {name}")
        print()
    if unlabeled:
        print(f"  [?] UNLABELED files ({len(unlabeled)}) — no ground truth, skipped:")
        print(f"  {sep}")
        for name in unlabeled:
            print(f"    {name}")
        print()
    return {"matched": matched, "missing": missing, "unlabeled": unlabeled,
            "n_matched": len(matched), "n_missing": len(missing), "n_unlabeled": len(unlabeled)}


# ---------------------------------------------------------------------------
# NLP pipeline accuracy parser  (the pipeline's own self-reported figure)
# ---------------------------------------------------------------------------

def _parse_nlp_accuracy(value) -> float:
    """
    Parse the pipeline's NLP_ACCURACY field to a 0–1 float.

    The pipeline returns this as a string like "87.50%" (from fuzzy_skill_verify).
    We normalise it here for consistent arithmetic in aggregate().
    """
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 100.0 if v > 1.0 else v
    try:
        return float(str(value).replace("%", "").strip()) / 100.0
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# FAISS rank helpers
# ---------------------------------------------------------------------------

def _find_job_rank_in_top5(
        top_5_matches: list[dict],
        expected_title: str,
) -> int:
    """
    Return the 1-based rank (1–5) of the first job whose title is
    sufficiently similar to expected_title, or 6 if not found.

    Uses the same TITLE_MATCH_THRESHOLD (0.70) applied to job_title_match.
    """
    for rank, match in enumerate(top_5_matches, start=1):
        if _title_similarity(match.get("title", ""), expected_title) >= TITLE_MATCH_THRESHOLD:
            return rank
    return 6  # sentinel: not in top-5


# ---------------------------------------------------------------------------
# compute_metrics  — one dict per resume
# ---------------------------------------------------------------------------

def compute_metrics(result: dict, label: dict) -> dict:
    # ── NLP skill-extraction quality (your ground truth vs pipeline output) ──
    expected_skills:   list[str] = label.get("expected_skills", [])
    discovered_skills: list[str] = result.get("discovered_skills", [])
    matched_exp, missing_exp, extra_disc = _match_skill_sets(expected_skills, discovered_skills)
    n_exp  = len(expected_skills)
    n_disc = len(discovered_skills)
    n_tp   = len(matched_exp)
    recall    = n_tp / n_exp  if n_exp  else 1.0
    precision = n_tp / n_disc if n_disc else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # ── NLP pipeline self-accuracy (system's own fuzzy NLP_ACCURACY figure) ──
    # This is what fuzzy_skill_verify() computed inside the pipeline against
    # the best-matched job's required_skills list — NOT your ground truth.
    # Tracking it alongside skill_f1 lets you see whether the pipeline's
    # self-assessment is calibrated against your external ground truth.
    nlp_pipeline_accuracy = _parse_nlp_accuracy(result.get("nlp_accuracy", "N/A"))

    # ── Title similarity (job title match) ───────────────────────────────────
    expected_title: str = label.get("expected_job_title", "")
    best_match          = result.get("best", {})
    best_title: str     = best_match.get("title", "")
    title_sim           = _title_similarity(best_title, expected_title) if expected_title else 0.0
    job_title_match     = int(title_sim >= TITLE_MATCH_THRESHOLD)

    # ── FAISS ranking quality ─────────────────────────────────────────────────
    top_5_matches: list[dict] = result.get("top_5_matches", [])

    # Rank of the expected job in the top-5 list (1 = best, 6 = not found)
    faiss_best_rank  = (
        _find_job_rank_in_top5(top_5_matches, expected_title)
        if expected_title else 6
    )
    faiss_top1_match = int(faiss_best_rank == 1)
    faiss_top5_match = int(faiss_best_rank <= 5)

    # Decomposed score components from the #1 ranked job
    # (best_match is already top_5_matches[0] from the pipeline)
    faiss_cosine_score: float = float(best_match.get("cosine_score", 0.0))
    faiss_match_ratio:  float = float(best_match.get("match_ratio",  0.0))

    # Hybrid score (pipeline uses 0–1 float; multiply by 100 for readability)
    best_hybrid_score: float = float(best_match.get("score", 0.0)) * 100.0
    min_score: float         = label.get("min_match_score", 50)
    score_pass               = int(best_hybrid_score >= min_score)

    # ── OCR quality ───────────────────────────────────────────────────────────
    # ocr_quality is the pipeline's alnum-ratio (float, 0–100).
    # text_source is "native_pdf", "ocr", or "camera_scan_mlkit".
    ocr_quality_pct: float = float(result.get("ocr_quality", 0.0))
    text_source: str       = result.get("text_source", "unknown")

    # text_source_correct: optional check — only scored when label provides
    # expected_text_source, so you can verify the right extraction path fired.
    expected_text_source: Optional[str] = label.get("expected_text_source")
    if expected_text_source:
        text_source_correct: Optional[int] = int(
            text_source.strip().lower() == expected_text_source.strip().lower()
        )
    else:
        text_source_correct = None  # not labelled — not scored

    # ── Certification recall ──────────────────────────────────────────────────
    expected_certs:   list[str] = label.get("expected_certs", [])
    discovered_certs: list[str] = result.get("discovered_certs", [])
    if expected_certs:
        cert_tp       = {normalize(c) for c in expected_certs} & {normalize(c) for c in discovered_certs}
        cert_recall: Optional[float] = len(cert_tp) / len(expected_certs)
    else:
        cert_recall = None

    return {
        # ── NLP skill-extraction quality (vs your ground truth) ──
        "skill_recall":             round(recall,    4),
        "skill_precision":          round(precision, 4),
        "skill_f1":                 round(f1,        4),
        "matched_skills":           sorted(matched_exp),
        "missing_skills":           sorted(missing_exp),
        "extra_skills":             sorted(extra_disc),

        # ── Pipeline self-reported NLP accuracy ──
        "nlp_pipeline_accuracy":    round(nlp_pipeline_accuracy, 4),

        # ── Job title match ──
        "job_title_match":          job_title_match,
        "expected_title":           expected_title,
        "best_title":               best_title,
        "title_similarity":         round(title_sim, 4),

        # ── FAISS ranking quality ──
        "faiss_top1_match":         faiss_top1_match,
        "faiss_top5_match":         faiss_top5_match,
        "faiss_best_rank":          faiss_best_rank,
        "faiss_cosine_score":       round(faiss_cosine_score, 4),
        "faiss_match_ratio":        round(faiss_match_ratio,  4),

        # ── Hybrid score / pass gate ──
        "best_hybrid_score":        round(best_hybrid_score, 2),
        "min_match_score":          min_score,
        "score_pass":               score_pass,

        # ── OCR quality ──
        "ocr_quality_pct":          round(ocr_quality_pct, 2),
        "text_source":              text_source,
        "text_source_correct":      text_source_correct,

        # ── Cert recall ──
        "cert_recall":              round(cert_recall, 4) if cert_recall is not None else None,
    }


# ---------------------------------------------------------------------------
# Aggregate  — summary stats across all resumes
# ---------------------------------------------------------------------------

def _safe_mean(vals):   return round(statistics.mean(vals),   4) if vals else 0.0
def _safe_median(vals): return round(statistics.median(vals), 4) if vals else 0.0
def _safe_std(vals):    return round(statistics.stdev(vals),  4) if len(vals) > 1 else 0.0


def aggregate(per_resume: list[dict]) -> dict:
    def _col(key): return [r[key] for r in per_resume if r.get(key) is not None]

    total    = len(per_resume)
    n_passed = sum(1 for r in per_resume if r["job_title_match"] and r["score_pass"])

    summary: dict = {}

    # ── NLP skill-extraction metrics ──────────────────────────────────────────
    for metric in ("skill_recall", "skill_precision", "skill_f1"):
        vals = _col(metric)
        summary[metric] = {
            "mean": _safe_mean(vals), "median": _safe_median(vals),
            "std":  _safe_std(vals),  "n": len(vals),
        }

    # ── Pipeline self-reported NLP accuracy ───────────────────────────────────
    nlp_vals = _col("nlp_pipeline_accuracy")
    summary["nlp_pipeline_accuracy"] = {
        "mean":   _safe_mean(nlp_vals),   "median": _safe_median(nlp_vals),
        "std":    _safe_std(nlp_vals),    "n":      len(nlp_vals),
    }

    # ── FAISS ranking metrics ─────────────────────────────────────────────────
    faiss_ranks = _col("faiss_best_rank")
    summary["faiss_top1_match"] = {
        "rate": round(sum(_col("faiss_top1_match")) / total, 4) if total else 0.0,
        "n":    total,
    }
    summary["faiss_top5_match"] = {
        "rate": round(sum(_col("faiss_top5_match")) / total, 4) if total else 0.0,
        "n":    total,
    }
    summary["faiss_best_rank"] = {
        "mean":   _safe_mean(faiss_ranks),   "median": _safe_median(faiss_ranks),
        "std":    _safe_std(faiss_ranks),    "n":      len(faiss_ranks),
    }
    for metric in ("faiss_cosine_score", "faiss_match_ratio", "best_hybrid_score"):
        vals = _col(metric)
        summary[metric] = {
            "mean": _safe_mean(vals), "median": _safe_median(vals),
            "std":  _safe_std(vals),  "n":      len(vals),
        }

    # ── OCR metrics ───────────────────────────────────────────────────────────
    ocr_vals = _col("ocr_quality_pct")
    summary["ocr_quality_pct"] = {
        "mean":   _safe_mean(ocr_vals),   "median": _safe_median(ocr_vals),
        "std":    _safe_std(ocr_vals),    "n":      len(ocr_vals),
    }
    src_correct_vals = _col("text_source_correct")
    if src_correct_vals:
        summary["text_source_correct"] = {
            "rate": round(sum(src_correct_vals) / len(src_correct_vals), 4),
            "n":    len(src_correct_vals),
        }
    else:
        summary["text_source_correct"] = {"rate": None, "n": 0}

    text_sources = [r.get("text_source", "unknown") for r in per_resume]
    summary["text_source_breakdown"] = {
        src: text_sources.count(src)
        for src in sorted(set(text_sources))
    }

    # ── Pass-rate gates ───────────────────────────────────────────────────────
    summary["job_title_match"] = {
        "rate": round(sum(_col("job_title_match")) / total, 4) if total else 0.0,
        "n": total,
    }
    summary["score_pass"] = {
        "rate": round(sum(_col("score_pass")) / total, 4) if total else 0.0,
        "n":    total,
    }
    summary["overall_pass_rate"] = {
        "rate":   round(n_passed / total, 4) if total else 0.0,
        "passed": n_passed,
        "total":  total,
    }

    # ── Cert recall ───────────────────────────────────────────────────────────
    cert_vals = [r["cert_recall"] for r in per_resume if r.get("cert_recall") is not None]
    summary["cert_recall"] = {
        "mean":   _safe_mean(cert_vals),   "median": _safe_median(cert_vals),
        "std":    _safe_std(cert_vals),    "n":      len(cert_vals),
    }

    return summary


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _pct(val: float) -> str: return f"{val * 100:.1f}%"
def _bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return "█" * filled + "░" * (width - filled)
def _display_name(filename: str) -> str:
    name = Path(filename).stem
    if " - " in name:
        name = name.split(" - ", 1)[1]
    return name.strip()


# ---------------------------------------------------------------------------
# Per-resume verbose output
# ---------------------------------------------------------------------------

def print_per_resume(results_list: list[dict], quiet: bool = False) -> None:
    if quiet:
        return
    for r in results_list:
        status = "✓" if (r["job_title_match"] and r["score_pass"]) else "✗"
        name   = _display_name(r["filename"])

        print(f"\n  [{status}] {name}")

        # OCR
        src_note = ""
        if r.get("text_source_correct") is not None:
            src_note = f"  ({'✓' if r['text_source_correct'] else '✗'} expected {r.get('expected_text_source','?')})"
        print(f"       OCR Quality    : {r['ocr_quality_pct']:.1f}%  source={r['text_source']}{src_note}")

        # NLP skill extraction
        print(f"       Skill F1       : {_pct(r['skill_f1'])}  "
              f"(recall={_pct(r['skill_recall'])}  prec={_pct(r['skill_precision'])})")
        print(f"       NLP Self-Score : {_pct(r['nlp_pipeline_accuracy'])}  "
              f"(pipeline vs best-job required skills)")
        if r["missing_skills"]:
            print(f"       Missing Skills : {r['missing_skills']}")
        if r["extra_skills"]:
            print(f"       Extra Skills   : {r['extra_skills']}")

        # FAISS
        rank_str = str(r["faiss_best_rank"]) if r["faiss_best_rank"] <= 5 else "not in top-5"
        print(f"       FAISS Rank     : {rank_str}  "
              f"(cosine={r['faiss_cosine_score']:.3f}  ratio={r['faiss_match_ratio']:.3f}  "
              f"hybrid={r['best_hybrid_score']:.1f})")

        if not r["job_title_match"]:
            print(f"       Title Mismatch : expected {r['expected_title']!r}, got {r['best_title']!r}")
        if r.get("cert_recall") is not None and r["cert_recall"] < 1.0:
            print(f"       Cert Recall    : {_pct(r['cert_recall'])}")
    print()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(summary: dict, label_coverage: float, n_total: int, n_labeled: int) -> None:
    sep = "-" * 64
    w   = 64

    print("\n" + "=" * w)
    print("  PIPELINE EVALUATION SUMMARY")
    print("=" * w)
    print(f"  Resumes evaluated : {n_labeled}/{n_total} ({_pct(label_coverage)} had labels)")

    # ── OCR ──────────────────────────────────────────────────────────────────
    print(f"\n  {'── OCR / Text Acquisition':<{w}}")
    print(f"  {sep}")
    ocr = summary["ocr_quality_pct"]
    print(f"  {'OCR Alnum Quality':<24}  "
          f"mean {_pct(ocr['mean'])}  median {_pct(ocr['median'])}  std {_pct(ocr['std'])}  "
          f"{_bar(ocr['mean'] / 100.0)}")

    src_bd = summary.get("text_source_breakdown", {})
    if src_bd:
        breakdown_str = "  ".join(f"{k}:{v}" for k, v in src_bd.items())
        print(f"  {'Text Source Breakdown':<24}  {breakdown_str}")

    src_corr = summary.get("text_source_correct", {})
    if src_corr.get("n", 0) > 0:
        print(f"  {'Source Path Correct':<24}  {_pct(src_corr['rate'])}  "
              f"({src_corr['n']} labelled)")
    else:
        print(f"  {'Source Path Correct':<24}  (no expected_text_source labels — add to labels.json to enable)")

    # ── NLP skill extraction ──────────────────────────────────────────────────
    print(f"\n  {'── NLP Skill Extraction  (your ground truth)':<{w}}")
    print(f"  {sep}")
    print(f"  {'Metric':<24}  {'Mean':>7}  {'Median':>7}  {'Std':>7}  Bar (mean)")
    print(f"  {sep}")
    for label_str, key in [
        ("Skill Recall",    "skill_recall"),
        ("Skill Precision", "skill_precision"),
        ("Skill F1",        "skill_f1"),
    ]:
        s = summary[key]
        print(f"  {label_str:<24}  {_pct(s['mean']):>7}  {_pct(s['median']):>7}  "
              f"{_pct(s['std']):>7}  {_bar(s['mean'])}")

    nlp_s = summary["nlp_pipeline_accuracy"]
    print(f"\n  {'NLP Self-Score':<24}  "
          f"mean {_pct(nlp_s['mean'])}  median {_pct(nlp_s['median'])}  std {_pct(nlp_s['std'])}")
    print(f"  {'':24}  (pipeline's own fuzzy NLP_ACCURACY vs best-matched job's required skills)")
    print(f"  {'':24}  Compare this to Skill F1 above — a big gap means the")
    print(f"  {'':24}  system over/under-estimates its own extraction quality.")

    # ── FAISS ranking ─────────────────────────────────────────────────────────
    print(f"\n  {'── FAISS Ranking Quality':<{w}}")
    print(f"  {sep}")
    t1 = summary["faiss_top1_match"]
    t5 = summary["faiss_top5_match"]
    br = summary["faiss_best_rank"]
    cs = summary["faiss_cosine_score"]
    mr = summary["faiss_match_ratio"]
    hs = summary["best_hybrid_score"]
    print(f"  {'Top-1 Rank Accuracy':<24}  {_pct(t1['rate'])}  ({t1['n']} evaluated)")
    print(f"  {'Top-5 Rank Accuracy':<24}  {_pct(t5['rate'])}  ({t5['n']} evaluated)")
    print(f"  {'Avg Rank of Expected Job':<24}  mean {br['mean']:.2f}  median {br['median']:.2f}  "
          f"(lower = better; 6 = not found)")
    print(f"  {'Cosine Score (SBERT)':<24}  mean {cs['mean']:.3f}  median {cs['median']:.3f}  "
          f"std {cs['std']:.3f}")
    print(f"  {'Skill Match Ratio':<24}  mean {_pct(mr['mean'])}  median {_pct(mr['median'])}  "
          f"std {_pct(mr['std'])}")
    print(f"  {'Best Hybrid Score':<24}  mean {hs['mean']:.1f}  median {hs['median']:.1f}  "
          f"std {hs['std']:.1f}")

    # ── Pass-rate gates ───────────────────────────────────────────────────────
    print(f"\n  {'── Pass-Rate Gates':<{w}}")
    print(f"  {sep}")
    jt = summary["job_title_match"]
    sp = summary["score_pass"]
    op = summary["overall_pass_rate"]
    print(f"  {'Job Title Match':<24}  {_pct(jt['rate'])}  ({jt['n']} evaluated)")
    print(f"  {'Score >= Threshold':<24}  {_pct(sp['rate'])}  ({sp['n']} evaluated)")
    print(f"  {'Both Pass (overall)':<24}  {_pct(op['rate'])}  ({op['passed']}/{op['total']})")

    # ── Certs ─────────────────────────────────────────────────────────────────
    cr = summary["cert_recall"]
    if cr["n"] > 0:
        print(f"\n  {'Cert Recall (n=' + str(cr['n']) + ')':<24}  "
              f"mean {_pct(cr['mean'])}  median {_pct(cr['median'])}")

    print()


# ---------------------------------------------------------------------------
# Evaluate  — main driver
# ---------------------------------------------------------------------------

def evaluate(
        source:     Optional[str] = None,
        limit:      Optional[int] = None,
        out_path:   Optional[str] = None,
        quiet:      bool          = False,
        charts:     bool          = False,
        charts_dir: str           = str(DEFAULT_CHARTS_DIR),
) -> dict:
    labels      = load_labels()
    all_resumes = collect_resumes(source)
    if limit:
        all_resumes = all_resumes[:limit]

    print("  Loading corpus from Firebase...")
    from db import load_analyzer_data
    firestore_skills, jobs_corpus, firestore_certs = load_analyzer_data()
    print(f"  Corpus ready — {len(jobs_corpus)} jobs, "
          f"{len(firestore_skills)} skills, {len(firestore_certs)} certs\n")

    from pipeline import run_pipeline as _orig_run_pipeline

    def run_pipeline(temp_path: str, file_name: str) -> dict:
        """
        Thin wrapper that suppresses all Firestore side-effects so the
        evaluation run is read-only and idempotent.  All actual analysis
        logic (extraction, LLM, SBERT, FAISS) runs as normal.
        """
        import db as _db
        import db.jobs as _db_jobs
        import pipeline as _pipeline_mod
        _orig_load       = _db.load_analyzer_data
        _orig_save_db    = _db.save_resume_result
        _orig_cache_jobs = _db_jobs.cache_job_embeddings
        _orig_cache_pipe = _pipeline_mod.cache_job_embeddings
        _orig_save_pipe  = _pipeline_mod.save_resume_result
        _noop = lambda *a, **kw: None
        _db.load_analyzer_data             = lambda: (firestore_skills, jobs_corpus, firestore_certs)
        _db.save_resume_result             = _noop
        _db_jobs.cache_job_embeddings      = _noop
        _pipeline_mod.cache_job_embeddings = _noop
        _pipeline_mod.save_resume_result   = _noop
        try:
            return _orig_run_pipeline(temp_path=temp_path, file_name=file_name)
        finally:
            _db.load_analyzer_data             = _orig_load
            _db.save_resume_result             = _orig_save_db
            _db_jobs.cache_job_embeddings      = _orig_cache_jobs
            _pipeline_mod.cache_job_embeddings = _orig_cache_pipe
            _pipeline_mod.save_resume_result   = _orig_save_pipe

    per_resume: list[dict] = []
    skipped:    list[str]  = []
    n_total = len(all_resumes)

    for filename, disk_path in tqdm(all_resumes, desc="Evaluating", unit="resume"):
        label = labels.get(filename)
        if label is None:
            skipped.append(filename)
            continue
        try:
            result = run_pipeline(temp_path=disk_path, file_name=filename)
        except Exception as exc:
            log.error("Pipeline failed for %s: %s", filename, exc, exc_info=True)
            skipped.append(filename)
            continue

        metrics             = compute_metrics(result, label)
        metrics["filename"] = filename
        per_resume.append(metrics)

        # Inline progress line — show F1, FAISS rank, and OCR quality
        name       = _display_name(filename)
        status     = "✓" if (metrics["job_title_match"] and metrics["score_pass"]) else "✗"
        rank_str   = f"#{metrics['faiss_best_rank']}" if metrics["faiss_best_rank"] <= 5 else "rank>5"
        print(
            f"  [{status}] {name:<35}  "
            f"F1={_pct(metrics['skill_f1'])}  "
            f"FAISS={rank_str}  "
            f"OCR={metrics['ocr_quality_pct']:.0f}%  "
            f"Score={metrics['best_hybrid_score']:.1f}"
        )

    n_labeled      = len(per_resume)
    label_coverage = n_labeled / n_total if n_total else 0.0

    if not per_resume:
        log.error("No resumes were successfully evaluated.")
        sys.exit(1)

    summary = aggregate(per_resume)
    print_per_resume(per_resume, quiet=quiet)
    print_summary(summary, label_coverage, n_total, n_labeled)

    if skipped:
        print(f"  Skipped ({len(skipped)}) — no label or pipeline error:")
        for s in skipped[:10]:
            print(f"    * {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")
        print()

    report = {
        "per_resume":     per_resume,
        "summary":        summary,
        "label_coverage": label_coverage,
        "n_total":        n_total,
        "n_labeled":      n_labeled,
        "n_skipped":      len(skipped),
        "skipped_files":  skipped,
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Full report saved -> {out_path}")

    if charts:
        _run_charts(report, charts_dir)

    return report


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _run_charts(report: dict, charts_dir: str) -> None:
    try:
        from generate_charts import generate_charts
    except ImportError:
        print("  [charts] generate_charts.py not found — place it next to evaluate_pipeline.py.")
        return
    generate_charts(report, out_dir=charts_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--resumes",    "-r", default=None, metavar="PATH")
    p.add_argument("--check",      "-c", action="store_true",
                   help="Audit label<->file coverage only.")
    p.add_argument("--limit",      "-n", type=int, default=None, metavar="N")
    p.add_argument("--out",        "-o", default=None, metavar="FILE",
                   help="Write full JSON report to FILE.")
    p.add_argument("--quiet",      "-q", action="store_true")
    p.add_argument("--charts",           action="store_true",
                   help="Generate diagnostic charts after the eval run.")
    p.add_argument("--charts-dir",       default=str(DEFAULT_CHARTS_DIR), metavar="DIR",
                   help="Output directory for chart PNGs (default: eval_charts/).")
    p.add_argument("--charts-only",      action="store_true",
                   help="Skip the pipeline run — load report from --out and generate charts only.")
    p.add_argument("--log-level",  default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-8s  %(name)s  %(message)s")
    if args.log_level != "DEBUG":
        for _name in _PIPELINE_LOGGERS:
            logging.getLogger(_name).setLevel(logging.ERROR)

    if args.check:
        check_coverage(args.resumes)

    elif args.charts_only:
        if not args.out:
            print("Error: --charts-only requires --out <report.json> to load from.")
            sys.exit(1)
        report_path = Path(args.out)
        if not report_path.exists():
            print(f"Error: report file not found: {report_path}")
            sys.exit(1)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        print(f"  Loaded {len(report.get('per_resume', []))} resume results from {report_path}")
        _run_charts(report, args.charts_dir)

    else:
        evaluate(
            source=args.resumes,
            limit=args.limit,
            out_path=args.out,
            quiet=args.quiet,
            charts=args.charts,
            charts_dir=args.charts_dir,
        )