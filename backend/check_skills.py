"""
check_skills.py
===============
Audit every skill that appears in report.json against the live Firestore
skills allowlist.

For each skill found in the report (matched, missing, or extra) this script
tells you:

    FOUND     - exact or near-exact match in the allowlist
    FUZZY     - close match found (>= 80 score) — likely an alias or typo
    NOT FOUND - no match at all — skill is not in your Firestore corpus

Usage
-----
    python check_skills.py                          # uses report.json next to this file
    python check_skills.py --report path/to/report.json
    python check_skills.py --report report.json --min-score 85
    python check_skills.py --report report.json --not-found-only
    python check_skills.py --report report.json --out skill_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process as rfprocess

from core.utils import normalize
from db import load_analyzer_data

log = logging.getLogger("check_skills")

SCRIPT_DIR  = Path(__file__).parent
DEFAULT_REPORT = SCRIPT_DIR / "report.json"

# ─────────────────────────────────────────────────────────────────────────────
# Collect all skills from report.json
# ─────────────────────────────────────────────────────────────────────────────

def collect_skills_from_report(report: dict) -> dict[str, set[str]]:
    """
    Walk every resume in per_resume and collect skills by category.

    Returns
    -------
    {
        "matched":  {skill, ...},
        "missing":  {skill, ...},
        "extra":    {skill, ...},
        "all":      {skill, ...},   # union of all three
    }
    """
    matched: set[str] = set()
    missing: set[str] = set()
    extra:   set[str] = set()

    for resume in report.get("per_resume", []):
        matched.update(resume.get("matched_skills", []))
        missing.update(resume.get("missing_skills", []))
        extra.update(resume.get("extra_skills",   []))

    return {
        "matched": matched,
        "missing": missing,
        "extra":   extra,
        "all":     matched | missing | extra,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skill lookup against allowlist
# ─────────────────────────────────────────────────────────────────────────────

def _best_match(skill: str, allowlist_norms: list[str], min_score: int) -> tuple[str, int]:
    """
    Return (best_matching_norm, score) for a skill against the allowlist.
    Uses token_sort_ratio + ratio, takes the max.
    """
    sn = normalize(skill)

    # Fast exact check
    if sn in allowlist_norms:
        return sn, 100

    # Substring check (e.g. "node" in "nodejs")
    for an in allowlist_norms:
        if (sn in an or an in sn) and len(sn) >= 3:
            score = 95
            if score >= min_score:
                return an, score

    # Fuzzy
    result = rfprocess.extractOne(
        sn,
        allowlist_norms,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=min_score,
    )
    if result:
        return result[0], result[1]

    return "", 0


def audit_skills(
        skills:          set[str],
        allowlist:       list[str],
        allowlist_norms: list[str],
        norm_to_canon:   dict[str, str],
        min_score:       int = 80,
) -> list[dict]:
    """
    For each skill, determine if it exists in the Firestore allowlist.

    Returns a list of result dicts sorted by status (NOT FOUND first) then name.
    """
    results = []

    for skill in sorted(skills):
        best_norm, score = _best_match(skill, allowlist_norms, min_score)

        if score == 100:
            status      = "FOUND"
            canon       = norm_to_canon.get(best_norm, best_norm)
            match_label = canon
        elif score >= min_score:
            status      = "FUZZY"
            canon       = norm_to_canon.get(best_norm, best_norm)
            match_label = f"{canon}  (score {score})"
        else:
            status      = "NOT FOUND"
            match_label = ""

        results.append({
            "skill":       skill,
            "status":      status,
            "best_match":  match_label,
            "score":       score,
        })

    # Sort: NOT FOUND first, then FUZZY, then FOUND; alpha within each group
    order = {"NOT FOUND": 0, "FUZZY": 1, "FOUND": 2}
    results.sort(key=lambda r: (order[r["status"]], r["skill"].lower()))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {"FOUND": "✓", "FUZZY": "~", "NOT FOUND": "✗"}
_STATUS_LABEL = {
    "FOUND":     "FOUND     ",
    "FUZZY":     "FUZZY     ",
    "NOT FOUND": "NOT FOUND ",
}

def print_audit(
        results:       list[dict],
        category:      str,
        not_found_only: bool = False,
) -> None:
    filtered = [r for r in results if not not_found_only or r["status"] == "NOT FOUND"]
    if not filtered:
        return

    n_found     = sum(1 for r in results if r["status"] == "FOUND")
    n_fuzzy     = sum(1 for r in results if r["status"] == "FUZZY")
    n_not_found = sum(1 for r in results if r["status"] == "NOT FOUND")

    print(f"\n  {'─' * 62}")
    print(f"  {category.upper()}  "
          f"({len(results)} unique skills — "
          f"✓ {n_found} found, ~ {n_fuzzy} fuzzy, ✗ {n_not_found} not found)")
    print(f"  {'─' * 62}")

    for r in filtered:
        icon  = _STATUS_ICON[r["status"]]
        label = _STATUS_LABEL[r["status"]]
        line  = f"  [{icon}] {label}  {r['skill']:<45}"
        if r["best_match"]:
            line += f"  -> {r['best_match']}"
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(category_results: dict[str, list[dict]]) -> None:
    print(f"\n  {'=' * 62}")
    print("  SKILL AUDIT SUMMARY")
    print(f"  {'=' * 62}")
    print(f"  {'Category':<15}  {'Total':>6}  {'Found':>6}  {'Fuzzy':>6}  {'Not Found':>10}")
    print(f"  {'-' * 60}")

    totals = {"total": 0, "found": 0, "fuzzy": 0, "not_found": 0}

    for cat, results in category_results.items():
        n_found     = sum(1 for r in results if r["status"] == "FOUND")
        n_fuzzy     = sum(1 for r in results if r["status"] == "FUZZY")
        n_not_found = sum(1 for r in results if r["status"] == "NOT FOUND")
        n_total     = len(results)

        print(f"  {cat:<15}  {n_total:>6}  {n_found:>6}  {n_fuzzy:>6}  {n_not_found:>10}")

        totals["total"]     += n_total
        totals["found"]     += n_found
        totals["fuzzy"]     += n_fuzzy
        totals["not_found"] += n_not_found

    print(f"  {'-' * 60}")
    print(f"  {'TOTAL':<15}  {totals['total']:>6}  {totals['found']:>6}  "
          f"{totals['fuzzy']:>6}  {totals['not_found']:>10}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
        report_path:    str,
        min_score:      int  = 80,
        not_found_only: bool = False,
        out_path:       Optional[str] = None,
        categories:     list[str] = None,
) -> dict:
    # ── Load report ───────────────────────────────────────────────────────────
    rp = Path(report_path)
    if not rp.exists():
        log.error("Report file not found: %s", rp)
        sys.exit(1)

    with open(rp, "r", encoding="utf-8") as f:
        report = json.load(f)

    skills_by_cat = collect_skills_from_report(report)
    total_unique  = len(skills_by_cat["all"])
    print(f"\n  Report       : {rp.name}")
    print(f"  Resumes      : {report.get('n_labeled', '?')}")
    print(f"  Unique skills: {total_unique}  "
          f"(matched={len(skills_by_cat['matched'])}  "
          f"missing={len(skills_by_cat['missing'])}  "
          f"extra={len(skills_by_cat['extra'])})")

    # ── Load Firestore allowlist once ─────────────────────────────────────────
    print("\n  Loading allowlist from Firestore...")
    firestore_skills, _, _ = load_analyzer_data()
    allowlist_norms = [normalize(s) for s in firestore_skills]
    norm_to_canon   = {normalize(s): s for s in firestore_skills}
    print(f"  Allowlist    : {len(firestore_skills)} skills\n")

    # ── Audit each category ───────────────────────────────────────────────────
    cats_to_run = categories or ["matched", "missing", "extra"]
    category_results: dict[str, list[dict]] = {}

    for cat in cats_to_run:
        skills = skills_by_cat.get(cat, set())
        results = audit_skills(
            skills, firestore_skills, allowlist_norms, norm_to_canon, min_score
        )
        category_results[cat] = results
        print_audit(results, cat, not_found_only=not_found_only)

    print_summary(category_results)

    # ── Save JSON output ──────────────────────────────────────────────────────
    audit_report = {
        "report_file":     str(rp),
        "firestore_skills": len(firestore_skills),
        "min_score":       min_score,
        "categories":      category_results,
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2, default=str)
        print(f"  Audit saved -> {out_path}")

    return audit_report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--report", "-r",
                   default=str(DEFAULT_REPORT),
                   metavar="FILE",
                   help="Path to report.json (default: backend/report.json).")
    p.add_argument("--min-score", "-s",
                   type=int, default=80, metavar="N",
                   help="Fuzzy match threshold 0-100 (default: 80).")
    p.add_argument("--not-found-only", "-x",
                   action="store_true",
                   help="Only print skills that are NOT in the allowlist.")
    p.add_argument("--category", "-c",
                   choices=["matched", "missing", "extra", "all"],
                   default=None,
                   help="Audit only one category (default: all three).")
    p.add_argument("--out", "-o",
                   default=None, metavar="FILE",
                   help="Save full audit as JSON to FILE.")
    p.add_argument("--log-level",
                   default="ERROR",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: ERROR).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s  %(name)s  %(message)s",
    )

    cats = None
    if args.category == "all":
        cats = ["matched", "missing", "extra"]
    elif args.category:
        cats = [args.category]

    run(
        report_path=args.report,
        min_score=args.min_score,
        not_found_only=args.not_found_only,
        out_path=args.out,
        categories=cats,
    )