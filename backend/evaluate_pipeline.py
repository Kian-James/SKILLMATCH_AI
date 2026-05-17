"""
evaluate_pipeline.py
====================
Offline evaluation harness for the full resume-analysis pipeline.

Usage
-----
    # Auto-detect resumes/ folder sitting next to this script:
    python evaluate_pipeline.py

    # Explicit folder of PDFs (e.g. inside the backend directory):
    python evaluate_pipeline.py --resumes resumes/

    # Evaluate a single PDF:
    python evaluate_pipeline.py --resumes resumes/Aaron_Sterling.pdf

    # Check coverage ONLY (no pipeline run) — see which files are
    # in labels but missing from the folder, and vice-versa:
    python evaluate_pipeline.py --check

    # Limit to N resumes (useful for quick sanity checks):
    python evaluate_pipeline.py --resumes resumes/ --limit 20

    # Save a detailed JSON report:
    python evaluate_pipeline.py --resumes resumes/ --out report.json

    # Quiet mode (only print the summary table):
    python evaluate_pipeline.py --resumes resumes/ --quiet

Ground Truth
------------
Labels are loaded from labels.json in the same directory as this script.
Each entry is keyed by the original PDF filename and contains:
    {
        "expected_skills":    ["Python", "SQL", ...],
        "expected_job_title": "Data Scientist",
        "min_match_score":    50,
        "expected_certs":     ["AWS CCP"]   # optional
    }

Skill Matching
--------------
Skills are compared using the same canonicalize() + allowlist the pipeline
uses internally, so aliases like "Node.js" / "NodeJS" / "Node" all map to
the same canonical form and count as a match.

Metrics reported per resume
----------------------------
    skill_recall        - fraction of expected_skills found by the pipeline
    skill_precision     - fraction of discovered_skills that are expected
    skill_f1            - harmonic mean of recall and precision
    job_title_match     - 1 if best-match title ~ expected title (fuzzy), else 0
    score_pass          - 1 if best hybrid score x 100 >= min_match_score
    cert_recall         - fraction of expected_certs found (when label has certs)
    nlp_accuracy        - NLP_ACCURACY from the pipeline's own validator (float)
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
    def tqdm(iterable, **_):  # type: ignore[misc]
        return iterable

try:
    from rapidfuzz import fuzz as _rfuzz
    def _title_similarity(a: str, b: str) -> float:
        return _rfuzz.token_sort_ratio(a.lower(), b.lower()) / 100.0
except ImportError:
    def _title_similarity(a: str, b: str) -> float:  # type: ignore[misc]
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

from core.utils import normalize

log = logging.getLogger("evaluate_pipeline")

# Silence all other loggers so only our output prints during the eval loop
# (Firebase, pipeline stages, OCR — all suppressed unless --log-level DEBUG)
_PIPELINE_LOGGERS = [
    "db.client", "db.jobs", "db.results",
    "core.extraction", "core.skills", "core.certs", "core.scoring",
    "pipeline", "easyocr", "easyocr.easyocr",
]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR          = Path(__file__).parent
LABELS_PATH         = SCRIPT_DIR / "labels.json"
DEFAULT_RESUMES_DIR = SCRIPT_DIR / "resumes"
TITLE_MATCH_THRESHOLD = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Label loader
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(path: Path = LABELS_PATH) -> dict:
    if not path.exists():
        log.error("labels.json not found at %s", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Resume collector
# ─────────────────────────────────────────────────────────────────────────────

def collect_resumes(source: Optional[str] = None) -> list[tuple[str, str]]:
    """Return (filename, disk_path) pairs from a folder or single PDF."""
    source_path = Path(source) if source else DEFAULT_RESUMES_DIR

    if not source_path.exists():
        if source is None:
            log.error(
                "No resumes folder found at %s.\n"
                "Either create a 'resumes/' folder inside the backend directory "
                "or pass --resumes <path>.",
                DEFAULT_RESUMES_DIR,
            )
        else:
            log.error("Source path does not exist: %s", source_path)
        sys.exit(1)

    if source_path.is_file():
        if source_path.suffix.lower() != ".pdf":
            log.error("Single-file mode only supports PDFs, got: %s", source_path.name)
            sys.exit(1)
        return [(source_path.name, str(source_path))]

    if source_path.is_dir():
        pdfs = sorted(source_path.glob("*.pdf"))
        if not pdfs:
            log.error("No PDF files found in %s", source_path)
            sys.exit(1)
        log.info("Found %d PDF(s) in %s", len(pdfs), source_path)
        return [(p.name, str(p)) for p in pdfs]

    log.error("Unsupported source: %s", source_path)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Skill canonicalization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _skills_match(a: str, b: str) -> bool:
    """
    Two skills match if either:
      1. Their normalized forms are identical (e.g. "Node.js" == "Nodejs")
      2. One is a prefix/suffix of the other (e.g. "node" in "nodejs")
      3. Fuzzy token_sort_ratio >= 85 (catches minor OCR/alias variants)
    """
    from rapidfuzz import fuzz as _fuzz
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return _fuzz.token_sort_ratio(na, nb) >= 85


def _match_skill_sets(
        expected:  list[str],
        discovered: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """
    Fuzzy-match expected vs discovered skills.

    Returns (matched_exp, missing_exp, extra_disc) as sets of
    the original string values (not normalized), for readable reporting.
    """
    matched_exp  = set()
    matched_disc = set()

    for exp in expected:
        for disc in discovered:
            if _skills_match(exp, disc):
                matched_exp.add(exp)
                matched_disc.add(disc)
                break   # each expected skill counts once

    missing_exp  = set(expected)  - matched_exp
    extra_disc   = set(discovered) - matched_disc
    return matched_exp, missing_exp, extra_disc


# ─────────────────────────────────────────────────────────────────────────────
# Coverage audit
# ─────────────────────────────────────────────────────────────────────────────

def check_coverage(resumes_source: Optional[str] = None) -> dict:
    """Audit label <-> file coverage WITHOUT running the pipeline."""
    labels      = load_labels()
    all_resumes = collect_resumes(resumes_source)

    label_names  = set(labels.keys())
    folder_names = {fname for fname, _ in all_resumes}

    matched   = sorted(label_names & folder_names)
    missing   = sorted(label_names - folder_names)
    unlabeled = sorted(folder_names - label_names)

    sep = "-" * 62
    print()
    print("=" * 62)
    print("  COVERAGE AUDIT")
    print("=" * 62)
    print(f"  labels.json entries : {len(label_names)}")
    print(f"  PDFs in folder      : {len(folder_names)}")
    print(f"  Matched (runnable)  : {len(matched)}")
    print(f"  Missing from folder : {len(missing)}")
    print(f"  Unlabeled PDFs      : {len(unlabeled)}")
    print()

    if matched:
        print(f"  [OK] MATCHED ({len(matched)}) — will be evaluated:")
        print(f"  {sep}")
        for name in matched:
            label      = labels[name]
            certs_note = (f"  +{len(label.get('expected_certs', []))} certs"
                          if label.get("expected_certs") else "")
            print(f"    {name:<50}  {len(label.get('expected_skills', []))} skills{certs_note}")
        print()

    if missing:
        print(f"  [!] MISSING from folder ({len(missing)}) — skipped during eval:")
        print(f"  {sep}")
        for name in missing:
            print(f"    {name}")
        print()

    if unlabeled:
        print(f"  [?] UNLABELED PDFs ({len(unlabeled)}) — no ground truth, will be skipped:")
        print(f"  {sep}")
        for name in unlabeled:
            print(f"    {name}")
        print()

    return {
        "matched":     matched,
        "missing":     missing,
        "unlabeled":   unlabeled,
        "n_matched":   len(matched),
        "n_missing":   len(missing),
        "n_unlabeled": len(unlabeled),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-resume metrics
# ─────────────────────────────────────────────────────────────────────────────

def _parse_nlp_accuracy(value) -> float:
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 100.0 if v > 1.0 else v
    try:
        return float(str(value).replace("%", "").strip()) / 100.0
    except (ValueError, TypeError):
        return 0.0


def compute_metrics(result: dict, label: dict) -> dict:
    """
    Compute evaluation metrics for one resume against its ground-truth label.

    Skills are matched with fuzzy comparison so aliases like Node / NodeJS /
    Node.js all count as the same skill.
    """
    # ── Skills ────────────────────────────────────────────────────────────────
    expected_skills:   list[str] = label.get("expected_skills", [])
    discovered_skills: list[str] = result.get("discovered_skills", [])

    matched_exp, missing_exp, extra_disc = _match_skill_sets(
        expected_skills, discovered_skills
    )

    n_exp  = len(expected_skills)
    n_disc = len(discovered_skills)
    n_tp   = len(matched_exp)

    recall    = n_tp / n_exp  if n_exp  else 1.0
    precision = n_tp / n_disc if n_disc else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    matched_skills = sorted(matched_exp)
    missing_skills = sorted(missing_exp)
    extra_skills   = sorted(extra_disc)

    # ── Job title ─────────────────────────────────────────────────────────────
    expected_title: str = label.get("expected_job_title", "")
    best_match          = result.get("best", {})
    best_title: str     = best_match.get("title", "")
    title_sim           = _title_similarity(best_title, expected_title) if expected_title else 0.0
    job_title_match     = int(title_sim >= TITLE_MATCH_THRESHOLD)

    # ── Score pass ────────────────────────────────────────────────────────────
    min_score: float = label.get("min_match_score", 50)
    best_score: float = float(best_match.get("score", 0.0)) * 100
    score_pass        = int(best_score >= min_score)

    # ── Certificates ──────────────────────────────────────────────────────────
    expected_certs:  list[str] = label.get("expected_certs", [])
    discovered_certs: list[str] = result.get("discovered_certs", [])

    if expected_certs:
        cert_exp_norms  = {normalize(c) for c in expected_certs}
        cert_disc_norms = {normalize(c) for c in discovered_certs}
        cert_tp         = cert_exp_norms & cert_disc_norms
        cert_recall: Optional[float] = len(cert_tp) / len(cert_exp_norms)
    else:
        cert_recall = None

    # ── NLP accuracy ──────────────────────────────────────────────────────────
    nlp_accuracy = _parse_nlp_accuracy(result.get("nlp_accuracy", "N/A"))

    return {
        "skill_recall":     round(recall,    4),
        "skill_precision":  round(precision, 4),
        "skill_f1":         round(f1,        4),
        "job_title_match":  job_title_match,
        "score_pass":       score_pass,
        "cert_recall":      round(cert_recall, 4) if cert_recall is not None else None,
        "nlp_accuracy":     round(nlp_accuracy, 4),
        "expected_title":   expected_title,
        "best_title":       best_title,
        "title_similarity": round(title_sim, 4),
        "best_score":       round(best_score, 2),
        "min_match_score":  min_score,
        "matched_skills":   matched_skills,
        "missing_skills":   missing_skills,
        "extra_skills":     extra_skills,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(vals):   return round(statistics.mean(vals),   4) if vals else 0.0
def _safe_median(vals): return round(statistics.median(vals), 4) if vals else 0.0
def _safe_std(vals):    return round(statistics.stdev(vals),  4) if len(vals) > 1 else 0.0


def aggregate(per_resume: list[dict]) -> dict:
    def _col(key):
        return [r[key] for r in per_resume if r.get(key) is not None]

    cert_vals = [r["cert_recall"] for r in per_resume if r.get("cert_recall") is not None]
    total     = len(per_resume)
    n_passed  = sum(1 for r in per_resume if r["job_title_match"] and r["score_pass"])

    summary: dict = {}
    for metric in ("skill_recall", "skill_precision", "skill_f1", "nlp_accuracy", "best_score"):
        vals = _col(metric)
        summary[metric] = {
            "mean":   _safe_mean(vals),
            "median": _safe_median(vals),
            "std":    _safe_std(vals),
            "n":      len(vals),
        }

    summary["job_title_match"] = {
        "rate": round(sum(_col("job_title_match")) / total, 4) if total else 0.0,
        "n":    total,
    }
    summary["score_pass"] = {
        "rate": round(sum(_col("score_pass")) / total, 4) if total else 0.0,
        "n":    total,
    }
    summary["cert_recall"] = {
        "mean":   _safe_mean(cert_vals),
        "median": _safe_median(cert_vals),
        "std":    _safe_std(cert_vals),
        "n":      len(cert_vals),
    }
    summary["overall_pass_rate"] = {
        "rate":    round(n_passed / total, 4) if total else 0.0,
        "passed":  n_passed,
        "total":   total,
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"

def _bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return "█" * filled + "░" * (width - filled)

def _display_name(filename: str) -> str:
    """'Com - Aaron Sterling.pdf' -> 'Aaron Sterling'"""
    name = Path(filename).stem          # strip .pdf
    if " - " in name:
        name = name.split(" - ", 1)[1]  # strip 'Com - ' prefix
    return name.strip()


def print_per_resume(results_list: list[dict], quiet: bool = False) -> None:
    if quiet:
        return
    for r in results_list:
        status = "✓" if (r["job_title_match"] and r["score_pass"]) else "✗"
        name   = _display_name(r["filename"])
        print(f"\n  [{status}] {name}")
        print(f"       Matched Skills : {r['matched_skills'] or '—'}")
        print(f"       Missing Skills : {r['missing_skills'] or '—'}")
        print(f"       Extra Skills   : {r['extra_skills']   or '—'}")
        if not r["job_title_match"]:
            print(f"       Title Mismatch : expected {r['expected_title']!r}, "
                  f"got {r['best_title']!r}")
        if r.get("cert_recall") is not None and r["cert_recall"] < 1.0:
            print(f"       Cert Recall    : {_pct(r['cert_recall'])}")
    print()


def print_summary(summary: dict, label_coverage: float, n_total: int, n_labeled: int) -> None:
    sep = "-" * 60
    print()
    print("=" * 60)
    print("  PIPELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Resumes evaluated : {n_labeled}/{n_total} ({_pct(label_coverage)} had labels)")
    print()

    metrics = [
        ("Skill Recall",    summary["skill_recall"]),
        ("Skill Precision", summary["skill_precision"]),
        ("Skill F1",        summary["skill_f1"]),
        ("NLP Accuracy",    summary["nlp_accuracy"]),
    ]
    print(f"  {'Metric':<22}  {'Mean':>7}  {'Median':>7}  {'Std':>7}  Bar (mean)")
    print(f"  {sep}")
    for label, s in metrics:
        bar = _bar(s["mean"])
        print(f"  {label:<22}  {_pct(s['mean']):>7}  {_pct(s['median']):>7}  "
              f"{_pct(s['std']):>7}  {bar}")

    print()
    print(f"  {'Pass-rate Metrics':<22}  {'Rate':>7}  Count")
    print(f"  {sep}")
    jt = summary["job_title_match"]
    sp = summary["score_pass"]
    op = summary["overall_pass_rate"]
    print(f"  {'Job Title Match':<22}  {_pct(jt['rate']):>7}  ({jt['n']} evaluated)")
    print(f"  {'Score >= Threshold':<22}  {_pct(sp['rate']):>7}  ({sp['n']} evaluated)")
    print(f"  {'Both Pass (overall)':<22}  {_pct(op['rate']):>7}  "
          f"({op['passed']}/{op['total']})")

    cr = summary["cert_recall"]
    if cr["n"] > 0:
        cr_n = cr["n"]
        print()
        print(f"  {'Cert Recall (n=' + str(cr_n) + ')':<22}  {_pct(cr['mean']):>7}  "
              f"median {_pct(cr['median'])}")

    bs = summary["best_score"]
    print()
    print(f"  Best Hybrid Score  —  mean {bs['mean']:.1f}  "
          f"median {bs['median']:.1f}  std {bs['std']:.1f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
        source:   Optional[str] = None,
        limit:    Optional[int] = None,
        out_path: Optional[str] = None,
        quiet:    bool          = False,
) -> dict:
    labels      = load_labels()
    all_resumes = collect_resumes(source)
    if limit:
        all_resumes = all_resumes[:limit]

    # ── Load Firebase ONCE before the loop ────────────────────────────────────
    print("  Loading corpus from Firebase...")
    from db import load_analyzer_data
    firestore_skills, jobs_corpus, firestore_certs = load_analyzer_data()
    print(f"  Corpus ready — {len(jobs_corpus)} jobs, "
          f"{len(firestore_skills)} skills, {len(firestore_certs)} certs\n")

    # Patch run_pipeline to accept pre-loaded data so it skips its own DB call
    from pipeline import run_pipeline as _orig_run_pipeline

    def run_pipeline(temp_path: str, file_name: str) -> dict:
        # Patch every Firestore-touching symbol so the pipeline makes
        # ZERO writes during eval — patching both the db module AND
        # pipeline's own namespace (since it does `from db import ...`).
        import db as _db
        import db.jobs as _db_jobs
        import pipeline as _pipeline_mod

        _orig_load         = _db.load_analyzer_data
        _orig_save_db      = _db.save_resume_result
        _orig_cache_jobs   = _db_jobs.cache_job_embeddings
        _orig_cache_pipe   = _pipeline_mod.cache_job_embeddings
        _orig_save_pipe    = _pipeline_mod.save_resume_result

        _noop = lambda *a, **kw: None
        _db.load_analyzer_data          = lambda: (firestore_skills, jobs_corpus, firestore_certs)
        _db.save_resume_result          = _noop
        _db_jobs.cache_job_embeddings   = _noop
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

        # Clean per-resume log line
        name   = _display_name(filename)
        status = "✓" if (metrics["job_title_match"] and metrics["score_pass"]) else "✗"
        print(f"  [{status}] {name:<35}  "
              f"F1={_pct(metrics['skill_f1'])}  "
              f"Score={metrics['best_score']:.1f}")

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

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--resumes", "-r", default=None, metavar="PATH",
                   help="Folder of PDFs or a single PDF. Defaults to backend/resumes/.")
    p.add_argument("--check", "-c", action="store_true",
                   help="Audit label<->file coverage only. Does NOT run the pipeline.")
    p.add_argument("--limit", "-n", type=int, default=None, metavar="N",
                   help="Evaluate only the first N resumes.")
    p.add_argument("--out", "-o", default=None, metavar="FILE",
                   help="Write full JSON report to FILE.")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Skip per-resume skill breakdown; only print the summary table.")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: WARNING — pipeline noise suppressed).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s  %(name)s  %(message)s",
    )

    # Suppress noisy pipeline/library loggers unless DEBUG requested
    if args.log_level != "DEBUG":
        for _name in _PIPELINE_LOGGERS:
            logging.getLogger(_name).setLevel(logging.ERROR)

    if args.check:
        check_coverage(args.resumes)
    else:
        evaluate(
            source=args.resumes,
            limit=args.limit,
            out_path=args.out,
            quiet=args.quiet,
        )