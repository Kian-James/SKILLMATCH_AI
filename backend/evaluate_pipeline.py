"""
evaluate_pipeline.py  (LOCAL MODE)
====================================
Runs the resume matching pipeline LOCALLY — no FastAPI server needed.

Imports matcher.py and firestore_client.py directly and calls the exact same
functions that main.py calls, so results are identical to the API version but
without any HTTP overhead or server setup.

Folder structure required
--------------------------
  your_project/
  ├── evaluate_pipeline.py   ← this script
  ├── matcher.py             ← your pipeline code
  ├── firestore_client.py    ← your Firestore client
  ├── labels.json            ← ground truth labels
  ├── .env                   ← HF_TOKEN, FIREBASE_CREDENTIALS_PATH, etc.
  └── resumes/               ← folder with all 202 PDFs
      ├── Com - Abigail Cambell.pdf
      ├── Eng - Aaron Lim.pdf
      └── ...

Quick start
-----------
  # Dry-run first — validates filenames vs labels, no pipeline calls:
  python evaluate_pipeline.py --dry-run

  # Full evaluation (loads Firestore + runs pipeline per resume):
  python evaluate_pipeline.py

  # Resume a killed run (skips already-completed resumes):
  python evaluate_pipeline.py --resume

  # Only evaluate Computer Science resumes:
  python evaluate_pipeline.py --filter-prefix "Com -"

  # Only evaluate Engineering resumes:
  python evaluate_pipeline.py --filter-prefix "Eng -"

CLI options
-----------
  --resumes-dir     Folder containing PDF resumes        (default: ./resumes)
  --labels          Path to labels.json                  (default: ./labels.json)
  --output          Where to write eval_results.json     (default: ./eval_results.json)
  --workers         Parallel workers                     (default: 1)
                    WARNING: keep at 1 unless you know your pipeline is
                    thread-safe. SBERT / FAISS share global state in matcher.py.
  --fuzzy-skill     Fuzzy threshold for skill matching   (default: 80)
  --fuzzy-title     Fuzzy threshold for job title match  (default: 70)
  --fuzzy-cert      Fuzzy threshold for cert matching    (default: 75)
  --resume          Skip resumes already in output file  (flag)
  --dry-run         Validate folder vs labels only       (flag)
  --filter-prefix   Only files whose name starts with X  (default: all)

Metrics per resume
------------------
  skill_recall        fraction of expected_skills found in discovered_skills
  skill_precision     fraction of discovered_skills matching an expected skill
  skill_f1            harmonic mean of recall and precision
  job_title_match     1/0 — top-5 titles contain expected_job_title (null if "")
  cert_recall         fraction of expected_certs found (null if none expected)
  score_ok            1 if best_match_score >= min_match_score, else 0

Output
------
  eval_results.json  — full results with aggregate + per_resume breakdown
  Terminal           — live progress bar + summary table after completion
"""

from __future__ import annotations

# ── OpenMP / FAISS guard — must be FIRST before any torch/faiss import ────────
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import re
import shutil
import statistics
import string
import sys
import tempfile
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rapidfuzz import fuzz
from tqdm import tqdm

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,       # suppress matcher.py's verbose INFO logs
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Keep our own logger at INFO so eval progress is visible
log = logging.getLogger("evaluate_pipeline")
log.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy pipeline imports
# These are imported once at startup so import errors surface immediately
# with a clear message rather than failing silently on the first resume.
# ─────────────────────────────────────────────────────────────────────────────

def _import_pipeline():
    """Import matcher + firestore_client. Raises ImportError with a helpful message."""
    try:
        import matcher as _matcher
    except ImportError as e:
        raise ImportError(
            f"Cannot import matcher.py: {e}\n"
            "  Make sure evaluate_pipeline.py is in the same folder as matcher.py."
        ) from e

    try:
        import firestore_client as _fc
    except ImportError as e:
        raise ImportError(
            f"Cannot import firestore_client.py: {e}\n"
            "  Make sure evaluate_pipeline.py is in the same folder as firestore_client.py."
        ) from e

    return _matcher, _fc


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy helpers  (mirrors matcher.py normalize() exactly)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip().lower()


def fuzzy_match_any(query: str, candidates: list[str], threshold: int) -> bool:
    qn = normalize(query)
    for c in candidates:
        cn = normalize(c)
        score = max(
            fuzz.token_sort_ratio(qn, cn),
            fuzz.ratio(qn, cn),
            fuzz.partial_ratio(qn, cn),
        )
        if score >= threshold:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_metrics(
        expected:   list[str],
        discovered: list[str],
        threshold:  int,
) -> dict[str, float]:
    if not expected and not discovered:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
    if not expected:
        return {"recall": 1.0, "precision": 0.0, "f1": 0.0}
    if not discovered:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    matched_exp  = sum(1 for e in expected  if fuzzy_match_any(e, discovered, threshold))
    matched_disc = sum(1 for d in discovered if fuzzy_match_any(d, expected,  threshold))

    recall    = matched_exp  / len(expected)
    precision = matched_disc / len(discovered)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "recall":    round(recall,    4),
        "precision": round(precision, 4),
        "f1":        round(f1,        4),
    }


def compute_cert_recall(
        expected_certs:   list[str],
        discovered_certs: list[str],
        threshold:        int,
) -> float | None:
    if not expected_certs:
        return None
    if not discovered_certs:
        return 0.0
    matched = sum(1 for e in expected_certs if fuzzy_match_any(e, discovered_certs, threshold))
    return round(matched / len(expected_certs), 4)


def compute_title_match(
        expected_title: str,
        top_matches:    list[dict],
        threshold:      int,
) -> int | None:
    if not expected_title.strip():
        return None
    titles = [m.get("title", "") for m in top_matches]
    return 1 if fuzzy_match_any(expected_title, titles, threshold) else 0


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run — reconcile folder vs labels, no pipeline calls
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(resumes_dir: Path, labels: dict, filter_prefix: str) -> None:
    all_pdfs = sorted(resumes_dir.glob("*.pdf"))
    if filter_prefix:
        all_pdfs = [p for p in all_pdfs if p.name.startswith(filter_prefix)]

    pdf_names   = {p.name for p in all_pdfs}
    label_names = set(labels.keys())

    in_both    = sorted(pdf_names & label_names)
    pdf_only   = sorted(pdf_names - label_names)
    label_only = sorted(label_names - pdf_names)

    print("\n" + "=" * 70)
    print("  DRY-RUN — Folder vs Labels Reconciliation")
    print("=" * 70)
    print(f"  PDFs found in folder : {len(pdf_names)}")
    print(f"  Labels in JSON       : {len(label_names)}")
    print(f"  Matched (will run)   : {len(in_both)}")
    print(f"  PDFs missing label   : {len(pdf_only)}")
    print(f"  Labels missing PDF   : {len(label_only)}")

    if pdf_only:
        print(f"\n  WARNING — PDFs with NO label entry (will be skipped):")
        for name in pdf_only:
            print(f"    {name}")

    if label_only:
        print(f"\n  WARNING — Labels with NO matching PDF on disk (will be skipped):")
        for name in label_only:
            print(f"    {name}")

    if in_both:
        print(f"\n  READY — {len(in_both)} resumes will be evaluated:")
        print(f"  {'File':<52} {'Skills':>6}  {'Certs':>5}  Expected Job Title")
        print("  " + "-" * 100)
        for name in in_both:
            lbl      = labels[name]
            n_skills = len(lbl.get("expected_skills", []))
            n_certs  = len(lbl.get("expected_certs",  []))
            title    = lbl.get("expected_job_title", "") or "(no title)"
            print(f"  {name:<52} {n_skills:>6}  {n_certs:>5}  {title}")

    print()
    print("  Run without --dry-run to start the evaluation.")
    print("=" * 70 + "\n")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Core: run the pipeline for one resume (LOCAL — no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_local(
        pdf_path:     Path,
        jobs_corpus:  list[dict],
        extra_skills: list[str],
        extra_certs:  list[str],
        matcher,                    # the imported matcher module
) -> dict[str, Any]:
    """
    Replicates exactly what main.py's /thesis-full-analysis endpoint does,
    but inline — no HTTP, no FastAPI, no server needed.

    Returns the same keys that main.py returns so the rest of the evaluator
    (metric computation, aggregation, summary printing) works unchanged.
    """
    # Stage 1 — text acquisition
    resume_text, ocr_quality = matcher.extract_text(str(pdf_path))

    import fitz as _fitz
    _doc    = _fitz.open(str(pdf_path))
    _native = " ".join(p.get_text("text") for p in _doc).strip()
    _doc.close()
    text_source = "native_pdf" if len(_native) >= 50 else "ocr"

    if not resume_text.strip():
        raise ValueError("Could not extract any text from the PDF.")

    # Stage 2 — skill extraction
    discovered_skills = matcher.discover_skills_dynamically(
        resume_text  = resume_text,
        jobs_corpus  = jobs_corpus,
        extra_skills = extra_skills,
    )

    # Stage 2B — cert extraction
    discovered_certificates = matcher.extract_certificates(
        resume_text = resume_text,
        allowlist   = extra_certs if extra_certs else None,
    )

    # Stage 3 — hybrid scoring + per-job NLP validation
    top_matches = matcher.calculate_cosine_similarity(
        resume_text   = resume_text,
        resume_skills = discovered_skills,
        jobs_corpus   = jobs_corpus,
    )

    return {
        "text_source":        text_source,
        "ocr_quality":        ocr_quality,
        "discovered_skills":  discovered_skills,
        "discovered_certs":   discovered_certificates,
        "top_matches":        top_matches,
        "best_score":         top_matches[0]["score"] if top_matches else 0.0,
        "nlp_accuracy":       top_matches[0]["nlp_validation"]["NLP_ACCURACY"] if top_matches else "N/A",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-resume evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one(
        pdf_path:     Path,
        label:        dict,
        jobs_corpus:  list[dict],
        extra_skills: list[str],
        extra_certs:  list[str],
        matcher,
        skill_th:     int,
        title_th:     int,
        cert_th:      int,
) -> dict[str, Any]:
    """
    Run the full local pipeline for one resume and compare against its label.
    Always returns a result dict — never raises — so one bad resume doesn't
    kill the entire batch.
    """
    result: dict[str, Any] = {
        "file":   pdf_path.name,
        "status": "ok",
        "error":  None,
    }
    t0 = time.perf_counter()

    try:
        pipeline_out = run_pipeline_local(
            pdf_path     = pdf_path,
            jobs_corpus  = jobs_corpus,
            extra_skills = extra_skills,
            extra_certs  = extra_certs,
            matcher      = matcher,
        )
    except Exception as exc:
        result.update({
            "status":    "error",
            "error":     str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        })
        return result

    result["elapsed_s"] = round(time.perf_counter() - t0, 2)

    discovered_skills = pipeline_out["discovered_skills"]
    discovered_certs  = pipeline_out["discovered_certs"]
    top_matches       = pipeline_out["top_matches"]
    best_score        = pipeline_out["best_score"]
    nlp_accuracy      = pipeline_out["nlp_accuracy"]
    text_source       = pipeline_out["text_source"]

    # ── Label fields ──────────────────────────────────────────────────────────
    expected_skills = label.get("expected_skills", [])
    expected_title  = label.get("expected_job_title", "")
    expected_certs  = label.get("expected_certs", [])
    min_score_pct   = label.get("min_match_score", 50) / 100.0

    # ── Skill metrics ─────────────────────────────────────────────────────────
    skill_metrics      = compute_skill_metrics(expected_skills, discovered_skills, skill_th)
    matched_expected   = [e for e in expected_skills if     fuzzy_match_any(e, discovered_skills, skill_th)]
    unmatched_expected = [e for e in expected_skills if not fuzzy_match_any(e, discovered_skills, skill_th)]
    extra_discovered   = [d for d in discovered_skills if not fuzzy_match_any(d, expected_skills, skill_th)]

    # ── Title match ───────────────────────────────────────────────────────────
    title_match = compute_title_match(expected_title, top_matches, title_th)
    top_titles  = [m.get("title", "") for m in top_matches[:5]]

    # ── Cert recall ───────────────────────────────────────────────────────────
    cert_recall = compute_cert_recall(expected_certs, discovered_certs, cert_th)

    # ── Score OK ──────────────────────────────────────────────────────────────
    score_ok = 1 if best_score >= min_score_pct else 0

    result.update({
        "text_source":               text_source,
        # Skills
        "expected_skills":           expected_skills,
        "discovered_skills":         discovered_skills,
        "skill_recall":              skill_metrics["recall"],
        "skill_precision":           skill_metrics["precision"],
        "skill_f1":                  skill_metrics["f1"],
        "matched_expected_skills":   matched_expected,
        "unmatched_expected_skills": unmatched_expected,
        "extra_discovered_skills":   extra_discovered,
        # Title
        "expected_job_title":        expected_title,
        "top_matched_titles":        top_titles,
        "job_title_match":           title_match,
        # Certs
        "expected_certs":            expected_certs,
        "discovered_certs":          discovered_certs,
        "cert_recall":               cert_recall,
        # Scoring
        "min_match_score":           label.get("min_match_score", 50),
        "best_match_score":          round(best_score * 100, 2),
        "score_ok":                  score_ok,
        # Pipeline internals
        "stage_nlp_accuracy":        nlp_accuracy,
    })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict[str, Any]:
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return {"total_resumes": len(results), "successful": 0, "failed": len(results)}

    def _mean(v):   return round(statistics.mean(v),   4) if v else None
    def _median(v): return round(statistics.median(v), 4) if v else None

    recalls    = [r["skill_recall"]    for r in ok]
    precisions = [r["skill_precision"] for r in ok]
    f1s        = [r["skill_f1"]        for r in ok]

    title_scores = [r["job_title_match"] for r in ok if r["job_title_match"] is not None]
    cert_scores  = [r["cert_recall"]     for r in ok if r["cert_recall"]     is not None]

    recall_brackets: dict[str, dict] = {}
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        count = sum(1 for r in ok if r["skill_recall"] >= thr)
        recall_brackets[f">={int(thr*100)}%"] = {
            "count": count,
            "pct":   round(count / len(ok) * 100, 1),
        }

    source_counts: dict[str, int] = {}
    for r in ok:
        src = r.get("text_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "total_resumes":         len(results),
        "successful":            len(ok),
        "failed":                len(results) - len(ok),
        "text_source_breakdown": source_counts,
        "skill_recall":          {"mean": _mean(recalls),    "median": _median(recalls)},
        "skill_precision":       {"mean": _mean(precisions), "median": _median(precisions)},
        "skill_f1":              {"mean": _mean(f1s),        "median": _median(f1s)},
        "recall_distribution":   recall_brackets,
        "job_title_match_rate":  round(sum(title_scores) / len(title_scores), 4) if title_scores else None,
        "cert_recall_rate":      _mean(cert_scores),
        "score_pass_rate":       round(sum(r["score_ok"] for r in ok) / len(ok), 4),
        "elapsed_s": {
            "total": round(sum(r.get("elapsed_s", 0) for r in ok), 2),
            "mean":  round(_mean([r.get("elapsed_s", 0) for r in ok]), 2),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(agg: dict, results: list[dict]) -> None:
    ok     = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Resumes evaluated  : {agg['total_resumes']}")
    print(f"  Successful         : {agg['successful']}")
    print(f"  Failed             : {agg['failed']}")

    src = agg.get("text_source_breakdown", {})
    if src:
        print("  Text source        : " + "  |  ".join(f"{k}: {v}" for k, v in src.items()))

    print()
    print("  SKILL EXTRACTION")
    r = agg.get("skill_recall",    {})
    p = agg.get("skill_precision", {})
    f = agg.get("skill_f1",        {})
    print(f"    Recall     mean: {r.get('mean', 0):.2%}   median: {r.get('median', 0):.2%}")
    print(f"    Precision  mean: {p.get('mean', 0):.2%}   median: {p.get('median', 0):.2%}")
    print(f"    F1         mean: {f.get('mean', 0):.2%}   median: {f.get('median', 0):.2%}")

    print()
    print("  RECALL DISTRIBUTION")
    for bracket, vals in agg.get("recall_distribution", {}).items():
        bar = "█" * int(vals["pct"] / 5)
        print(f"    {bracket:>6}  {bar:<20}  {vals['count']:3d} resumes  ({vals['pct']}%)")

    print()
    n_titled = sum(1 for r in ok if r.get("job_title_match") is not None)
    n_certed = sum(1 for r in ok if r.get("cert_recall")    is not None)
    jt = agg.get("job_title_match_rate")
    cr = agg.get("cert_recall_rate")
    sp = agg.get("score_pass_rate")

    print(f"  Job Title Match    : {jt:.2%}  (over {n_titled} resumes with expected title)" if jt is not None else "  Job Title Match    : N/A")
    print(f"  Cert Recall        : {cr:.2%}  (over {n_certed} resumes with expected certs)" if cr is not None else "  Cert Recall        : N/A")
    print(f"  Score Pass Rate    : {sp:.2%}" if sp is not None else "  Score Pass Rate    : N/A")

    el = agg.get("elapsed_s", {})
    print(f"\n  Elapsed            : {el.get('total', 0):.1f}s total  /  {el.get('mean', 0):.1f}s per resume")
    print("=" * 70)

    # ── Worst 10 by recall ────────────────────────────────────────────────────
    if ok:
        worst = sorted(ok, key=lambda r: r["skill_recall"])[:10]
        print("\n  LOWEST RECALL RESUMES (worst 10)")
        print(f"  {'File':<50} {'Recall':>7}  Unmatched expected skills")
        print("  " + "-" * 115)
        for r in worst:
            missed = r.get("unmatched_expected_skills", [])
            missed_str = ", ".join(missed[:4])
            if len(missed) > 4:
                missed_str += f" (+{len(missed)-4} more)"
            print(f"  {r['file']:<50} {r['skill_recall']:>6.0%}   {missed_str}")

    # ── Title mismatches ──────────────────────────────────────────────────────
    title_misses = [r for r in ok if r.get("job_title_match") == 0]
    if title_misses:
        print(f"\n  JOB TITLE MISMATCHES ({len(title_misses)} resumes)")
        print(f"  {'File':<50} {'Expected':<35} Top Matched")
        print("  " + "-" * 120)
        for r in sorted(title_misses, key=lambda x: x["file"])[:20]:
            top = (r.get("top_matched_titles") or [""])[0]
            print(f"  {r['file']:<50} {r['expected_job_title']:<35} {top}")

    # ── Failures ──────────────────────────────────────────────────────────────
    if failed:
        print(f"\n  FAILED RESUMES ({len(failed)})")
        for r in failed:
            print(f"    {r['file']}  —  {r.get('error', 'unknown error')}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local batch evaluator — imports matcher.py directly, no server needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resumes-dir",   default=os.getenv("EVAL_RESUMES_DIR",  "./resumes"),
                   help="Folder containing PDF resumes")
    p.add_argument("--labels",        default=os.getenv("EVAL_LABELS",       "./labels.json"),
                   help="Path to labels.json")
    p.add_argument("--output",        default=os.getenv("EVAL_OUTPUT",       "./eval_results.json"),
                   help="Path for output JSON")
    p.add_argument("--workers",       type=int, default=int(os.getenv("EVAL_WORKERS", "1")),
                   help="Parallel workers (keep at 1 — SBERT/FAISS share global state)")
    p.add_argument("--fuzzy-skill",   type=int, default=int(os.getenv("EVAL_FUZZY_SKILL", "80")),
                   help="Fuzzy threshold for skill matching (0-100)")
    p.add_argument("--fuzzy-title",   type=int, default=int(os.getenv("EVAL_FUZZY_TITLE", "70")),
                   help="Fuzzy threshold for job title match (0-100)")
    p.add_argument("--fuzzy-cert",    type=int, default=int(os.getenv("EVAL_FUZZY_CERT",  "75")),
                   help="Fuzzy threshold for cert matching (0-100)")
    p.add_argument("--resume",        action="store_true",
                   help="Skip resumes already completed in --output file")
    p.add_argument("--dry-run",       action="store_true",
                   help="Validate folder vs labels only — no pipeline calls")
    p.add_argument("--filter-prefix", default=os.getenv("EVAL_FILTER_PREFIX", ""),
                   help="Only evaluate files whose name starts with this prefix")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    resumes_dir = Path(args.resumes_dir)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    # ── Validate paths ────────────────────────────────────────────────────────
    if not resumes_dir.is_dir():
        log.error(f"Resumes directory not found: {resumes_dir}")
        log.error("  Create a 'resumes' folder next to this script and put your PDFs in it.")
        log.error("  Or pass --resumes-dir /path/to/your/pdfs")
        sys.exit(1)

    if not labels_path.is_file():
        log.error(f"Labels file not found: {labels_path}")
        sys.exit(1)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels: dict[str, dict] = json.load(f)

    log.info(f"Loaded {len(labels)} labels from {labels_path}")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        dry_run(resumes_dir, labels, args.filter_prefix)
        return

    # ── Import pipeline (surfaces errors immediately) ─────────────────────────
    log.info("Importing pipeline (matcher.py + firestore_client.py) ...")
    try:
        matcher, firestore_client = _import_pipeline()
    except ImportError as e:
        log.error(str(e))
        sys.exit(1)

    # ── Load Firestore corpus once — shared across all resumes ────────────────
    log.info("Loading jobs corpus from Firestore ...")
    try:
        firestore_skills, jobs_corpus, firestore_certs = firestore_client.load_analyzer_data()
    except Exception as e:
        log.error(f"Failed to load Firestore data: {e}")
        log.error("  Check FIREBASE_CREDENTIALS_PATH in your .env file.")
        sys.exit(1)

    log.info(f"Corpus: {len(jobs_corpus)} jobs  |  {len(firestore_skills)} skills  |  {len(firestore_certs)} certs")

    # ── Collect labeled PDFs ──────────────────────────────────────────────────
    all_pdfs = sorted(resumes_dir.glob("*.pdf"))
    if args.filter_prefix:
        all_pdfs = [p for p in all_pdfs if p.name.startswith(args.filter_prefix)]
        log.info(f"Filter '{args.filter_prefix}' → {len(all_pdfs)} PDFs")

    labeled_pdfs = [p for p in all_pdfs if p.name in labels]
    unlabeled    = [p.name for p in all_pdfs if p.name not in labels]

    if unlabeled:
        log.warning(
            f"{len(unlabeled)} PDFs have no label — skipped. "
            f"Run --dry-run for the full list. "
            f"First few: {unlabeled[:3]}"
        )
    if not labeled_pdfs:
        log.error(
            "No labeled PDFs found. Filenames must exactly match labels.json keys.\n"
            "Run --dry-run to see a full reconciliation."
        )
        sys.exit(1)

    log.info(f"Found {len(labeled_pdfs)} labeled PDFs to evaluate.")

    # ── Resume mode ───────────────────────────────────────────────────────────
    already_done:  set[str]   = set()
    prior_results: list[dict] = []

    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            prior_data = json.load(f)
        prior_results = prior_data.get("per_resume", [])
        already_done  = {r["file"] for r in prior_results if r["status"] == "ok"}
        log.info(f"Resume mode: {len(already_done)} already done, skipping.")

    todo = [p for p in labeled_pdfs if p.name not in already_done]

    if not todo:
        log.info("All resumes already evaluated. Remove --resume or change --output to re-run.")
        sys.exit(0)

    log.info(f"Processing {len(todo)} resumes  (workers={args.workers}) ...")
    log.info(f"Thresholds: skill={args.fuzzy_skill}  title={args.fuzzy_title}  cert={args.fuzzy_cert}")
    log.info(f"Output    : {output_path}")

    # ── Batch execution ───────────────────────────────────────────────────────
    # Default is workers=1 because matcher.py has module-level singletons
    # (_ocr_reader, model, _cached_allowlist) that are not thread-safe.
    # Increase --workers only if you've verified your environment is safe.
    new_results: list[dict] = []

    def _run(pdf: Path) -> dict:
        return evaluate_one(
            pdf_path     = pdf,
            label        = labels[pdf.name],
            jobs_corpus  = jobs_corpus,
            extra_skills = firestore_skills,
            extra_certs  = firestore_certs,
            matcher      = matcher,
            skill_th     = args.fuzzy_skill,
            title_th     = args.fuzzy_title,
            cert_th      = args.fuzzy_cert,
        )

    if args.workers == 1:
        # Sequential — simple, safe, clear progress
        with tqdm(total=len(todo), desc="Evaluating", unit="resume", ncols=90) as pbar:
            for pdf in todo:
                result = _run(pdf)
                new_results.append(result)

                recall     = result.get("skill_recall")
                recall_str = f"{recall:.0%}" if isinstance(recall, float) else "ERR"
                pbar.set_postfix({
                    "file":   (pdf.name[:24] + "…") if len(pdf.name) > 25 else pdf.name,
                    "recall": recall_str,
                    "status": result["status"],
                })
                pbar.update(1)

                # Checkpoint after every resume
                all_so_far = prior_results + new_results
                _write_checkpoint(output_path, all_so_far, args)
    else:
        # Parallel — faster but requires thread-safe pipeline
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run, pdf): pdf.name for pdf in todo}
            with tqdm(total=len(todo), desc="Evaluating", unit="resume", ncols=90) as pbar:
                for future in as_completed(futures):
                    fname = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"file": fname, "status": "error", "error": str(exc)}
                    new_results.append(result)

                    recall     = result.get("skill_recall")
                    recall_str = f"{recall:.0%}" if isinstance(recall, float) else "ERR"
                    pbar.set_postfix({
                        "file":   (fname[:24] + "…") if len(fname) > 25 else fname,
                        "recall": recall_str,
                        "status": result["status"],
                    })
                    pbar.update(1)

                    all_so_far = prior_results + new_results
                    _write_checkpoint(output_path, all_so_far, args)

    # ── Final output ──────────────────────────────────────────────────────────
    all_results = prior_results + new_results
    agg = aggregate(all_results)

    final = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "mode":                  "local",
            "resumes_dir":           str(resumes_dir),
            "labels_file":           str(labels_path),
            "fuzzy_skill_threshold": args.fuzzy_skill,
            "fuzzy_title_threshold": args.fuzzy_title,
            "fuzzy_cert_threshold":  args.fuzzy_cert,
            "workers":               args.workers,
            "filter_prefix":         args.filter_prefix,
        },
        "aggregate":  agg,
        "per_resume": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    log.info(f"Results written to {output_path}")
    print_summary(agg, all_results)


def _write_checkpoint(output_path: Path, all_results: list[dict], args: argparse.Namespace) -> None:
    """Write intermediate results to disk after every resume so --resume can recover."""
    checkpoint = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "mode":        "local",
            "fuzzy_skill": args.fuzzy_skill,
            "fuzzy_title": args.fuzzy_title,
            "fuzzy_cert":  args.fuzzy_cert,
            "workers":     args.workers,
        },
        "aggregate":  aggregate(all_results),
        "per_resume": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()