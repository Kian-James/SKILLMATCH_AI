"""
evaluate_pipeline.py
────────────────────
Batch accuracy evaluator for the AI Resume Matching pipeline.

Runs every PDF in a resume folder through Stage 1 (OCR) → Stage 2a (NLP skill
extraction) → Stage 2b (canonicalization) → Stage 2c (cert extraction) and
compares results against ground truth in labels.json.

Produces:
  • Live console output with per-file results
  • eval_results.csv      — full per-file detail
  • eval_summary.json     — aggregate metrics for calibration
  • eval_failures.txt     — files that missed ≥ 1 expected skill (for debugging)

Usage:
    python evaluate_pipeline.py \
        --resumes   ./resumes \
        --labels    ./labels.json \
        --firestore ./mock_jobs.json   # optional: use offline mock if no Firebase

Firestore mode (default, needs .env with FIREBASE_CREDENTIALS_PATH):
    python evaluate_pipeline.py --resumes ./resumes --labels ./labels.json

Offline / mock mode (no Firebase needed):
    python evaluate_pipeline.py --resumes ./resumes --labels ./labels.json \
        --mock-jobs ./mock_jobs.json

Environment variables (loaded from .env — same as main.py):
    FIREBASE_CREDENTIALS_PATH
    FIRESTORE_JOBS_COLLECTION  (default: jobs)
"""

import os
import sys
import json
import time
import argparse
import traceback
import csv
from pathlib import Path
from datetime import datetime

# ── Load .env early so Firebase vars are set before firebase_admin imports ───
from dotenv import load_dotenv
load_dotenv()

from rapidfuzz import fuzz as _fuzz

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch pipeline evaluator")
    p.add_argument("--resumes",   required=True,  help="Folder containing resume PDFs")
    p.add_argument("--labels",    required=True,  help="Path to labels.json")
    p.add_argument("--mock-jobs", default=None,   help="Path to mock_jobs.json (offline mode)")
    p.add_argument("--out-dir",   default=".",    help="Directory to write output files (default: .)")
    p.add_argument("--fuzzy-threshold",  type=int,   default=80,
                   help="Fuzzy threshold for hit detection (default 80, same as main.py NLP accuracy)")
    p.add_argument("--limit",     type=int, default=None,
                   help="Only process first N files (for quick smoke tests)")
    p.add_argument("--resume-filter", default=None,
                   help="Comma-separated list of exact filenames to run (e.g. debugging a subset)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Job corpus loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_jobs_from_firestore(collection: str) -> list:
    """Fetch all job documents from Firestore (requires FIREBASE_CREDENTIALS_PATH)."""
    import firebase_admin
    from firebase_admin import credentials, firestore as fs

    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not cred_path:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_PATH is not set. "
            "Use --mock-jobs for offline mode, or add it to .env."
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db   = fs.client()
    docs = db.collection(collection).stream()
    jobs = [doc.to_dict() for doc in docs]

    if not jobs:
        raise RuntimeError(
            f"No jobs found in Firestore collection '{collection}'. "
            "Run generate_jobs.py to seed it, or use --mock-jobs."
        )
    return jobs


def load_jobs_from_mock(path: str) -> list:
    """Load jobs from a local JSON file for offline evaluation."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either a list or a dict keyed by title/id
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unrecognised mock_jobs format in {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def skill_hit(expected: str, extracted_list: list, threshold: int = 80) -> bool:
    """True if `expected` fuzzy-matches any item in extracted_list."""
    exp_l = expected.lower()
    for ext in extracted_list:
        if _fuzz.partial_ratio(exp_l, ext.lower()) >= threshold:
            return True
    return False


def evaluate_file(
        pdf_path: str,
        label_entry: dict,
        jobs_corpus: list,
        labels_skill_pool: list,
        labels_cert_pool: list,
        fuzzy_threshold: int,
) -> dict:
    """
    Run the full Stage 1 → 2a → 2b → 2c pipeline on one PDF and
    return a structured result dict.
    """
    from matcher import (
        extract_text,
        extract_skills_from_corpus,
        extract_certifications_section,
        extract_certificates_from_section,
        canonicalize_skills_to_corpus,
    )

    filename       = os.path.basename(pdf_path)
    expected_skills = label_entry.get("expected_skills", [])
    expected_certs  = label_entry.get("expected_certs",  [])
    expected_job    = label_entry.get("expected_job_title", "")
    min_score       = label_entry.get("min_match_score", 50)

    result = {
        # Identity
        "filename":          filename,
        "expected_job":      expected_job,
        "min_score":         min_score,

        # Stage 1
        "ocr_quality":       None,
        "raw_text_len":      None,

        # Stage 2a
        "raw_skills_count":  None,
        "raw_skills":        [],

        # Stage 2b
        "verified_skills":   [],
        "verified_count":    None,

        # Stage 2c
        "found_certs":       [],
        "cert_count":        None,

        # Skill accuracy
        "expected_skills":   expected_skills,
        "skill_hits":        0,
        "skill_misses":      [],
        "skill_total":       len(expected_skills),
        "skill_accuracy_pct": None,   # None when no ground truth

        # Cert accuracy
        "expected_certs":    expected_certs,
        "cert_hits":         0,
        "cert_misses":       [],
        "cert_total":        len(expected_certs),
        "cert_accuracy_pct": None,

        # Meta
        "elapsed_sec":       None,
        "error":             None,
    }

    t0 = time.time()
    try:
        # ── Stage 1: OCR ─────────────────────────────────────────────────────
        raw_text, ocr_quality = extract_text(pdf_path)
        result["ocr_quality"]   = ocr_quality
        result["raw_text_len"]  = len(raw_text)

        # ── Stage 2a: NLP extraction ─────────────────────────────────────────
        raw_skills = extract_skills_from_corpus(raw_text, jobs_corpus, labels_skill_pool)
        result["raw_skills"]       = raw_skills
        result["raw_skills_count"] = len(raw_skills)

        # ── Stage 2b: Canonicalization ────────────────────────────────────────
        verified_skills = canonicalize_skills_to_corpus(
            raw_skills,
            jobs_corpus,
            extra_skills=labels_skill_pool,
            semantic_threshold=0.88,
            fuzzy_threshold=86,
        )
        result["verified_skills"]  = verified_skills
        result["verified_count"]   = len(verified_skills)

        # ── Stage 2c: Cert extraction ─────────────────────────────────────────
        cert_section = extract_certifications_section(raw_text)
        found_certs  = extract_certificates_from_section(cert_section, labels_cert_pool)
        result["found_certs"] = found_certs
        result["cert_count"]  = len(found_certs)

        # ── Skill accuracy vs ground truth ────────────────────────────────────
        if expected_skills:
            hits   = []
            misses = []
            for exp in expected_skills:
                if skill_hit(exp, verified_skills, fuzzy_threshold):
                    hits.append(exp)
                else:
                    misses.append(exp)
            result["skill_hits"]         = len(hits)
            result["skill_misses"]        = misses
            result["skill_accuracy_pct"]  = round(len(hits) / len(expected_skills) * 100, 2)

        # ── Cert accuracy vs ground truth ─────────────────────────────────────
        if expected_certs:
            c_hits   = []
            c_misses = []
            for exp in expected_certs:
                if skill_hit(exp, found_certs, fuzzy_threshold):
                    c_hits.append(exp)
                else:
                    c_misses.append(exp)
            result["cert_hits"]         = len(c_hits)
            result["cert_misses"]        = c_misses
            result["cert_accuracy_pct"]  = round(len(c_hits) / len(expected_certs) * 100, 2)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    result["elapsed_sec"] = round(time.time() - t0, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporters
# ─────────────────────────────────────────────────────────────────────────────

PASS_ICON  = "✅"
FAIL_ICON  = "❌"
WARN_ICON  = "⚠️ "
ERROR_ICON = "💥"
SEP        = "─" * 80


def print_result(r: dict, idx: int, total: int):
    """Print a compact per-file summary to stdout."""
    status = ERROR_ICON if r["error"] else (
        PASS_ICON if (r["skill_accuracy_pct"] or 0) >= 80 else FAIL_ICON
    )
    acc_str = (
        f"{r['skill_accuracy_pct']}%"
        if r["skill_accuracy_pct"] is not None
        else "N/A (no GT)"
    )
    cert_str = (
        f"{r['cert_accuracy_pct']}%"
        if r["cert_accuracy_pct"] is not None
        else "N/A"
    )
    print(f"\n[{idx:>3}/{total}] {status} {r['filename']}")
    if r["error"]:
        print(f"       ERROR: {r['error'][:120]}")
        return
    print(f"       OCR Quality : {r['ocr_quality']}%   |   Text length: {r['raw_text_len']} chars")
    print(f"       Skills      : {r['skill_hits']}/{r['skill_total']} expected  →  accuracy {acc_str}   "
          f"(raw={r['raw_skills_count']}, verified={r['verified_count']})")
    print(f"       Certs       : {r['cert_hits']}/{r['cert_total']} expected  →  accuracy {cert_str}")
    if r["skill_misses"]:
        print(f"       MISSED      : {', '.join(r['skill_misses'])}")
    if r.get("cert_misses"):
        print(f"       CERT MISSED : {', '.join(r['cert_misses'])}")
    print(f"       ⏱  {r['elapsed_sec']}s")


def write_csv(results: list, path: str):
    """Write full per-file detail to CSV."""
    fieldnames = [
        "filename", "expected_job", "ocr_quality", "raw_text_len",
        "raw_skills_count", "verified_count", "cert_count",
        "skill_total", "skill_hits", "skill_accuracy_pct",
        "skill_misses",
        "cert_total", "cert_hits", "cert_accuracy_pct",
        "cert_misses",
        "verified_skills",
        "elapsed_sec", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            row["skill_misses"]    = "; ".join(r.get("skill_misses", []))
            row["cert_misses"]     = "; ".join(r.get("cert_misses", []))
            row["verified_skills"] = "; ".join(r.get("verified_skills", []))
            w.writerow(row)
    print(f"\n📄 CSV written → {path}")


def write_failures(results: list, path: str):
    """Write a human-readable failure report — only files with missed skills."""
    lines = []
    for r in results:
        if r["error"]:
            lines.append(f"[ERROR] {r['filename']}\n  {r['error'][:300]}\n")
        elif r["skill_misses"] or r.get("cert_misses"):
            lines.append(f"[MISS] {r['filename']}")
            lines.append(f"  Skill accuracy : {r['skill_accuracy_pct']}%  "
                         f"({r['skill_hits']}/{r['skill_total']})")
            if r["skill_misses"]:
                lines.append(f"  Missed skills  : {', '.join(r['skill_misses'])}")
            if r.get("cert_misses"):
                lines.append(f"  Missed certs   : {', '.join(r['cert_misses'])}")
            lines.append(f"  Verified skills: {', '.join(r['verified_skills'])}")
            lines.append("")
    if not lines:
        lines = ["All files passed with no missed skills! 🎉"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📋 Failure report → {path}")


def write_summary(results: list, path: str):
    """Write aggregate metrics to JSON for calibration analysis."""
    errors    = [r for r in results if r["error"]]
    with_gt   = [r for r in results if r["skill_total"] > 0 and not r["error"]]
    no_gt     = [r for r in results if r["skill_total"] == 0 and not r["error"]]
    with_cert = [r for r in results if r["cert_total"] > 0 and not r["error"]]

    # Per-file skill accuracy distribution
    accs = [r["skill_accuracy_pct"] for r in with_gt]
    avg_skill_acc = round(sum(accs) / len(accs), 2) if accs else None
    pass80  = sum(1 for a in accs if a >= 80)
    pass60  = sum(1 for a in accs if a >= 60)
    fail60  = sum(1 for a in accs if a < 60)

    # Cert accuracy
    cert_accs    = [r["cert_accuracy_pct"] for r in with_cert if r["cert_accuracy_pct"] is not None]
    avg_cert_acc = round(sum(cert_accs) / len(cert_accs), 2) if cert_accs else None

    # OCR quality
    ocr_vals    = [r["ocr_quality"] for r in results if r["ocr_quality"] is not None]
    avg_ocr     = round(sum(ocr_vals) / len(ocr_vals), 2) if ocr_vals else None
    low_ocr     = [r["filename"] for r in results if r["ocr_quality"] is not None and r["ocr_quality"] < 50]

    # Most commonly missed skills
    miss_counter: dict = {}
    for r in with_gt:
        for m in r.get("skill_misses", []):
            miss_counter[m] = miss_counter.get(m, 0) + 1
    top_misses = sorted(miss_counter.items(), key=lambda x: -x[1])[:30]

    # Most commonly missed certs
    cert_miss_counter: dict = {}
    for r in with_cert:
        for m in r.get("cert_misses", []):
            cert_miss_counter[m] = cert_miss_counter.get(m, 0) + 1
    top_cert_misses = sorted(cert_miss_counter.items(), key=lambda x: -x[1])[:20]

    # Skills verified per file stats
    verified_counts = [r["verified_count"] for r in results if r["verified_count"] is not None]
    avg_verified = round(sum(verified_counts) / len(verified_counts), 2) if verified_counts else None

    summary = {
        "generated_at": datetime.now().isoformat(),
        "totals": {
            "files_evaluated":    len(results),
            "with_ground_truth":  len(with_gt),
            "without_ground_truth": len(no_gt),
            "errors":             len(errors),
        },
        "skill_accuracy": {
            "average_pct":        avg_skill_acc,
            "pass_80pct_or_above": pass80,
            "pass_60pct_or_above": pass60,
            "fail_below_60pct":   fail60,
            "pass_rate_80":       f"{round(pass80/len(with_gt)*100,1)}%" if with_gt else "N/A",
        },
        "cert_accuracy": {
            "files_with_cert_gt": len(with_cert),
            "average_pct":        avg_cert_acc,
        },
        "ocr": {
            "average_quality_pct": avg_ocr,
            "files_below_50pct":   len(low_ocr),
            "low_ocr_files":       low_ocr,
        },
        "extraction": {
            "average_verified_skills_per_file": avg_verified,
        },
        "top_missed_skills": [
            {"skill": s, "missed_count": c} for s, c in top_misses
        ],
        "top_missed_certs": [
            {"cert": c, "missed_count": n} for c, n in top_cert_misses
        ],
        "calibration_hints": _calibration_hints(
            avg_skill_acc, avg_ocr, top_misses, low_ocr
        ),
        "error_files": [r["filename"] for r in errors],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary JSON  → {path}")
    return summary


def _calibration_hints(avg_acc, avg_ocr, top_misses, low_ocr_files) -> list:
    """Generate actionable calibration recommendations from aggregate metrics."""
    hints = []

    if avg_acc is not None and avg_acc < 70:
        hints.append(
            "⚠️  Average skill accuracy is below 70%. Consider: "
            "(1) expanding the Firestore required_skills lists for niche roles, "
            "(2) lowering fuzzy_threshold in canonicalize_skills_to_corpus from 86 → 82, "
            "(3) adding missing skills to labels.json skill pool."
        )
    if avg_ocr is not None and avg_ocr < 55:
        hints.append(
            "⚠️  Average OCR quality is low (<55%). Consider: "
            "(1) enabling EasyOCR as primary (not fallback) for low-quality PDFs, "
            "(2) applying image pre-processing (deskew, contrast) before OCR, "
            "(3) checking if PDFs are scanned vs. text-native."
        )
    if low_ocr_files:
        hints.append(
            f"⚠️  {len(low_ocr_files)} file(s) have OCR quality < 50%: "
            + ", ".join(low_ocr_files[:5])
            + (" ..." if len(low_ocr_files) > 5 else "")
        )
    if top_misses:
        top5 = [s for s, _ in top_misses[:5]]
        hints.append(
            f"💡 Most frequently missed skills: {', '.join(top5)}. "
            "Add these to Firestore required_skills or the labels.json pool."
        )
    if not hints:
        hints.append("✅ Pipeline looks healthy — no major calibration issues detected.")
    return hints


def print_summary(summary: dict):
    """Print aggregate summary to console."""
    print(f"\n{'='*80}")
    print("AGGREGATE EVALUATION SUMMARY")
    print(f"{'='*80}")
    t = summary["totals"]
    print(f"  Files evaluated   : {t['files_evaluated']}")
    print(f"  With ground truth : {t['with_ground_truth']}")
    print(f"  Without GT        : {t['without_ground_truth']}")
    print(f"  Errors            : {t['errors']}")

    sa = summary["skill_accuracy"]
    print(f"\n  Skill Accuracy (files with GT only):")
    print(f"    Average          : {sa['average_pct']}%")
    print(f"    Pass ≥80%        : {sa['pass_80pct_or_above']}  ({sa['pass_rate_80']})")
    print(f"    Pass ≥60%        : {sa['pass_60pct_or_above']}")
    print(f"    Fail  <60%       : {sa['fail_below_60pct']}")

    ca = summary["cert_accuracy"]
    print(f"\n  Cert Accuracy (files with cert GT only):")
    print(f"    Files            : {ca['files_with_cert_gt']}")
    print(f"    Average          : {ca['average_pct']}%")

    ocr = summary["ocr"]
    print(f"\n  OCR Quality:")
    print(f"    Average          : {ocr['average_quality_pct']}%")
    print(f"    Files below 50%  : {ocr['files_below_50pct']}")

    print(f"\n  Top Missed Skills:")
    for entry in summary["top_missed_skills"][:10]:
        print(f"    [{entry['missed_count']:>3}x]  {entry['skill']}")

    print(f"\n  Calibration Hints:")
    for hint in summary["calibration_hints"]:
        print(f"    {hint}")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate paths ────────────────────────────────────────────────────────
    resume_dir = Path(args.resumes)
    if not resume_dir.is_dir():
        sys.exit(f"ERROR: --resumes '{args.resumes}' is not a directory.")
    labels_path = Path(args.labels)
    if not labels_path.exists():
        sys.exit(f"ERROR: --labels '{args.labels}' not found.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load labels ───────────────────────────────────────────────────────────
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data: dict = json.load(f)

    # Build global skill and cert pools from all label entries
    labels_skill_pool: set = set()
    labels_cert_pool:  set = set()
    for entry in labels_data.values():
        for s in entry.get("expected_skills", []):
            if s.strip():
                labels_skill_pool.add(s.strip())
        for c in entry.get("expected_certs", []):
            if c.strip():
                labels_cert_pool.add(c.strip())
    labels_skill_pool = list(labels_skill_pool)
    labels_cert_pool  = list(labels_cert_pool)

    print(f"✅ Labels loaded  : {len(labels_data)} entries")
    print(f"   Skill pool     : {len(labels_skill_pool)} unique skills")
    print(f"   Cert pool      : {len(labels_cert_pool)} unique certs")

    # ── Load job corpus ───────────────────────────────────────────────────────
    if args.mock_jobs:
        print(f"📦 Loading jobs from mock file: {args.mock_jobs}")
        jobs_corpus = load_jobs_from_mock(args.mock_jobs)
    else:
        collection = os.getenv("FIRESTORE_JOBS_COLLECTION", "jobs")
        print(f"🔥 Fetching jobs from Firestore collection: '{collection}'")
        jobs_corpus = load_jobs_from_firestore(collection)

    print(f"✅ Jobs loaded    : {len(jobs_corpus)}\n")

    # ── Collect PDFs ──────────────────────────────────────────────────────────
    all_pdfs = sorted(resume_dir.glob("*.pdf"))

    # Optional filename filter
    if args.resume_filter:
        wanted = set(f.strip() for f in args.resume_filter.split(","))
        all_pdfs = [p for p in all_pdfs if p.name in wanted]
        print(f"🔍 Filter applied — {len(all_pdfs)} file(s) selected")

    if args.limit:
        all_pdfs = all_pdfs[:args.limit]
        print(f"⚡ Limit applied  — running first {len(all_pdfs)} file(s)")

    total = len(all_pdfs)
    if total == 0:
        sys.exit("ERROR: No PDF files found in the resumes directory.")

    print(f"📂 Resumes found  : {total}\n{SEP}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results      = []
    unlabeled    = []
    global_start = time.time()

    for idx, pdf_path in enumerate(all_pdfs, start=1):
        filename    = pdf_path.name
        label_entry = labels_data.get(filename, {})

        if not label_entry:
            unlabeled.append(filename)
            # Still run the pipeline so we can see what it extracts
            label_entry = {}

        r = evaluate_file(
            pdf_path      = str(pdf_path),
            label_entry   = label_entry,
            jobs_corpus   = jobs_corpus,
            labels_skill_pool = labels_skill_pool,
            labels_cert_pool  = labels_cert_pool,
            fuzzy_threshold   = args.fuzzy_threshold,
        )
        results.append(r)
        print_result(r, idx, total)

    elapsed_total = round(time.time() - global_start, 1)
    print(f"\n{SEP}")
    print(f"✅ Evaluation complete in {elapsed_total}s  ({total} files)")
    if unlabeled:
        print(f"{WARN_ICON} {len(unlabeled)} file(s) had no labels.json entry: "
              + ", ".join(unlabeled[:5])
              + (" ..." if len(unlabeled) > 5 else ""))

    # ── Write outputs ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path     = out_dir / f"eval_results_{ts}.csv"
    summary_path = out_dir / f"eval_summary_{ts}.json"
    failure_path = out_dir / f"eval_failures_{ts}.txt"

    write_csv(results, str(csv_path))
    summary = write_summary(results, str(summary_path))
    write_failures(results, str(failure_path))
    print_summary(summary)


if __name__ == "__main__":
    main()