"""
evaluate_pipeline.py
────────────────────
Batch accuracy evaluator for the AI Resume Matching pipeline.
Calls matcher.py functions DIRECTLY — no running server required.

Stages run per resume:
  Stage 1   — OCR               (extract_text)
  Stage 2a  — NLP skill extract (discover_skills_dynamically)
  Stage 2b  — Cert extraction   (extract_certificates)

Compares against ground truth in labels.json.

Output files (timestamped):
  eval_results_<ts>.csv        full per-file detail
  eval_summary_<ts>.json       aggregate metrics + calibration hints
  eval_failures_<ts>.txt       files that missed ≥1 expected skill/cert
  eval_ocr_report_<ts>.txt     OCR quality breakdown

──────────────────────────────────────────────────────────────────────────────
QUICKSTART
──────────────────────────────────────────────────────────────────────────────

1.  Create labels.json  (one entry per PDF filename):

    {
      "John_Doe_Resume.pdf": {
        "expected_skills": ["Python", "Docker", "PostgreSQL"],
        "expected_certs":  ["AWS Certified Developer"],
        "expected_job_title": "Backend Developer"
      },
      "Jane_Smith.pdf": {
        "expected_skills": ["React", "TypeScript"],
        "expected_certs":  []
      }
    }

    Keys:
      expected_skills     list of skills you can SEE in the PDF  (required)
      expected_certs      list of cert names visible in the PDF  (optional)
      expected_job_title  free text note                         (optional)

2.  Run with Firestore (same as main.py):

    python evaluate_pipeline.py --resumes ./resumes --labels ./labels.json

3.  Run OFFLINE with a local mock_jobs.json:

    python evaluate_pipeline.py \\
        --resumes   ./resumes \\
        --labels    ./labels.json \\
        --mock-jobs ./mock_jobs.json

    mock_jobs.json format — list of job objects exactly as Firestore returns:
    [
      {
        "id": 1,
        "title": "IT Support",
        "industry": "Computing",
        "description": "...",
        "required_skills": ["Zendesk", "AnyDesk", "Ansible", "macOS"],
        "required_certifications": []
      },
      ...
    ]

4.  Other flags:
    --out-dir ./results        where to write output files (default: .)
    --fuzzy-threshold 80       match threshold 0-100       (default: 80)
    --limit 10                 only process first N files  (smoke test)
    --resume-filter a.pdf,b.pdf  run specific files only
    --no-soft-filter           disable soft-skill pre-filtering of labels
                               (use to audit your labels.json quality)
"""

import os
import sys
import json
import re
import csv
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from rapidfuzz import fuzz as _fuzz


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch pipeline evaluator — no running server required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--resumes",         required=True,
                   help="Folder containing resume PDFs")
    p.add_argument("--labels",          required=True,
                   help="Path to labels.json (ground truth)")
    p.add_argument("--mock-jobs",       default=None,
                   help="Path to mock_jobs.json for offline mode (skips Firestore)")
    p.add_argument("--out-dir",         default=".",
                   help="Directory to write output files (default: .)")
    p.add_argument("--fuzzy-threshold", type=int, default=80,
                   help="Fuzzy match threshold for hit detection (default: 80)")
    p.add_argument("--limit",           type=int, default=None,
                   help="Process only first N files (quick smoke test)")
    p.add_argument("--resume-filter",   default=None,
                   help="Comma-separated exact filenames to run")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Job corpus loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_jobs_firestore(collection: str) -> list:
    import firebase_admin
    from firebase_admin import credentials, firestore as fs

    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not cred_path:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_PATH not set. "
            "Add it to .env, or use --mock-jobs for offline mode."
        )
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db   = fs.client()
    docs = db.collection(collection).stream()
    jobs = [doc.to_dict() for doc in docs]
    if not jobs:
        raise RuntimeError(
            f"No jobs in Firestore collection '{collection}'. "
            "Seed it with generate_jobs.py, or use --mock-jobs."
        )
    jobs.sort(key=lambda j: j.get("id", 0))
    return jobs


def load_jobs_mock(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unrecognised mock_jobs format in {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS-MEMBERSHIP FILTER FOR LABELS
#
# The evaluator uses the same rule as matcher.py: a label skill only counts
# toward accuracy if it exists in the Firestore corpus (jobs required_skills).
# Skills in labels.json that are NOT in the corpus (soft skills, custom phrases,
# etc.) are excluded from the denominator so the pipeline isn't penalised for
# correctly not finding them.
#
# corpus_norm is built once in main() and passed into evaluate_file().
# ══════════════════════════════════════════════════════════════════════════════

def filter_labels_to_corpus(
        expected_skills: list[str],
        corpus_norm:     dict[str, str],
        filename:        str = "",
) -> tuple[list[str], list[str]]:
    """
    Split expected_skills into (corpus_skills, non_corpus_skills).

    corpus_skills   — labels that exist in Firestore required_skills; these
                      are what the pipeline is designed to find, so accuracy
                      is scored against them.
    non_corpus_skills — labels absent from the corpus; excluded from scoring
                        since the pipeline would never emit them.

    Uses the same normalize() logic as matcher.py so "REST APIs" and
    "REST API Development" resolve consistently.
    """
    from matcher import normalize as _norm

    in_corpus, not_in_corpus = [], []
    for skill in expected_skills:
        if _norm(skill) in corpus_norm:
            in_corpus.append(skill)
        else:
            not_in_corpus.append(skill)

    if not_in_corpus:
        print(
            f"       [corpus-filter] {filename}: {len(not_in_corpus)} label(s) "
            f"not in corpus (excluded from scoring): "
            f"{', '.join(not_in_corpus[:6])}"
            + (" ..." if len(not_in_corpus) > 6 else "")
        )
    return in_corpus, not_in_corpus


# ══════════════════════════════════════════════════════════════════════════════
# Fuzzy hit check
# ══════════════════════════════════════════════════════════════════════════════

def skill_hit(expected: str, extracted: list, threshold: int = 80) -> bool:
    exp_l = expected.lower()
    for ext in extracted:
        if _fuzz.partial_ratio(exp_l, ext.lower()) >= threshold:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# OCR diagnostics
# ══════════════════════════════════════════════════════════════════════════════

_GARBLE_RE = re.compile(
    r"^[^a-zA-Z]{3,}$"
    r"|[bcdfghjklmnpqrstvwxz]{5,}"
    r"|[^\x00-\x7F]{2,}",
    re.IGNORECASE,
)
_SKILLS_KEYWORDS = {
    "skills", "technical", "competencies", "expertise",
    "proficiencies", "qualifications", "technologies",
}


def _ocr_diagnostics(raw_text: str, ocr_quality: float) -> dict:
    if   ocr_quality >= 85: grade = "A"
    elif ocr_quality >= 70: grade = "B"
    elif ocr_quality >= 50: grade = "C"
    elif ocr_quality >= 30: grade = "D"
    else:                   grade = "F"

    words         = raw_text.split()
    garbled       = [w for w in words if _GARBLE_RE.search(w)]
    garbled_ratio = round(len(garbled) / max(len(words), 1), 4)
    text_lower    = raw_text.lower()
    has_header    = any(kw in text_lower for kw in _SKILLS_KEYWORDS)

    if   len(raw_text) > 200 and garbled_ratio < 0.02: engine = "pdfminer (text-native)"
    elif garbled_ratio > 0.10:                          engine = "easyocr (scanned — high garble)"
    else:                                               engine = "easyocr (scanned)"

    issues = []
    if   ocr_quality < 30: issues.append(f"CRITICAL: OCR {ocr_quality}% — extraction nearly failed")
    elif ocr_quality < 50: issues.append(f"LOW OCR quality ({ocr_quality}%) — significant errors likely")
    elif ocr_quality < 70: issues.append(f"MODERATE OCR quality ({ocr_quality}%) — some tokens garbled")
    if len(raw_text) < 200:    issues.append(f"Very short text ({len(raw_text)} chars)")
    if garbled_ratio > 0.15:   issues.append(f"High garble {garbled_ratio:.1%} — examples: {garbled[:5]}")
    elif garbled_ratio > 0.05: issues.append(f"Moderate garble {garbled_ratio:.1%}")
    if not has_header:         issues.append("No Skills section header detected — full-doc fallback active")
    if words:
        isolated = sum(1 for w in words if len(w) == 1 and not w.isalpha())
        if isolated / len(words) > 0.08:
            issues.append(f"Many isolated punctuation tokens ({isolated}) — complex PDF layout")

    if   ocr_quality < 30 or len(raw_text) < 200:
        rec = "Re-export as text-native PDF. If scanned: deskew, CLAHE contrast, 300 DPI."
    elif ocr_quality < 50: rec = "Apply image pre-processing: deskew + adaptive contrast + 300 DPI."
    elif ocr_quality < 70: rec = "Try EasyOCR as primary engine. Verify PDF is not a low-res scan."
    elif not has_header:   rec = "Add a clear 'Skills' heading to the resume."
    else:                  rec = "No action needed."

    return {
        "grade":          grade,
        "engine":         engine,
        "char_count":     len(raw_text),
        "word_count":     len(words),
        "garbled_ratio":  garbled_ratio,
        "has_header":     has_header,
        "issues":         issues,
        "recommendation": rec,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-file evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_file(
        pdf_path:         str,
        label_entry:      dict,
        jobs_corpus:      list,
        label_skill_pool: list,
        label_cert_pool:  list,
        fuzzy_threshold:  int,
        corpus_norm:      dict | None = None,
) -> dict:
    """
    Run Stage 1 → 2a → 2b on one PDF using matcher.py directly.

    Ground truth labels are filtered to only skills present in the Firestore
    corpus before accuracy is computed. This matches exactly what the pipeline
    can ever emit — it only outputs skills from known_skills (the corpus), so
    labels outside the corpus would always be misses regardless of pipeline
    quality and should not count against accuracy.

    corpus_norm — pre-built {normalize(skill): canonical_skill} dict built
                  from all jobs required_skills. Passed in from main() to
                  avoid rebuilding it per file.
    """
    from matcher import (
        extract_text,
        discover_skills_dynamically,
        extract_certificates,
    )

    if corpus_norm is None:
        from matcher import normalize as _norm
        corpus_norm = {}
        for job in jobs_corpus:
            for s in job.get("required_skills", []):
                corpus_norm[_norm(s)] = s

    filename       = os.path.basename(pdf_path)
    raw_skills     = label_entry.get("expected_skills", [])
    expected_certs = label_entry.get("expected_certs",  [])
    expected_job   = label_entry.get("expected_job_title", "")

    # ── Corpus-membership filter on ground truth ──────────────────────────────
    expected_skills, non_corpus_skills = filter_labels_to_corpus(
        raw_skills, corpus_norm, filename=filename,
    )

    r = {
        # identity
        "filename":     filename,
        "expected_job": expected_job,
        # stage 1
        "ocr_quality":  None,
        "ocr":          {},
        # stage 2a
        "discovered_skills": [],
        "skill_count":       None,
        # stage 2b
        "discovered_certs":   [],
        "cert_count":         None,
        "cert_section_found": False,
        # skill accuracy (against corpus-filtered labels only)
        "expected_skills_raw":  raw_skills,
        "expected_skills":      expected_skills,
        "non_corpus_skills":    non_corpus_skills,
        "skill_hits":           0,
        "skill_misses":         [],
        "skill_total":          len(expected_skills),
        "skill_total_raw":      len(raw_skills),
        "skill_accuracy_pct":   None,
        # cert accuracy
        "expected_certs":    expected_certs,
        "cert_hits":         0,
        "cert_misses":       [],
        "cert_total":        len(expected_certs),
        "cert_accuracy_pct": None,
        # meta
        "elapsed_sec": None,
        "error":       None,
    }

    t0 = time.time()
    try:
        # ── Stage 1: OCR ─────────────────────────────────────────────────────
        raw_text, ocr_quality = extract_text(pdf_path)
        r["ocr_quality"] = ocr_quality
        r["ocr"]         = _ocr_diagnostics(raw_text, ocr_quality)

        # ── Stage 2a: NLP skill extraction ───────────────────────────────────
        discovered_skills = discover_skills_dynamically(
            resume_text  = raw_text,
            jobs_corpus  = jobs_corpus,
            extra_skills = label_skill_pool,
        )
        r["discovered_skills"] = discovered_skills
        r["skill_count"]       = len(discovered_skills)

        # ── Stage 2b: cert extraction ─────────────────────────────────────────
        discovered_certs = extract_certificates(raw_text)
        r["discovered_certs"]   = discovered_certs
        r["cert_count"]         = len(discovered_certs)
        r["cert_section_found"] = bool(discovered_certs) or bool(
            re.search(
                r"\b(certif|licens|credential|accredit)\w*",
                raw_text,
                re.IGNORECASE,
            )
        )

        # ── Skill accuracy vs corpus-filtered ground truth ────────────────────
        if expected_skills:
            hits, misses = [], []
            for exp in expected_skills:
                (hits if skill_hit(exp, discovered_skills, fuzzy_threshold) else misses).append(exp)
            r["skill_hits"]         = len(hits)
            r["skill_misses"]       = misses
            r["skill_accuracy_pct"] = round(len(hits) / len(expected_skills) * 100, 2)
        elif raw_skills:
            # All labels were outside the corpus — mark as N/A (not a failure)
            r["skill_accuracy_pct"] = None

        # ── Cert accuracy vs ground truth ─────────────────────────────────────
        if expected_certs:
            c_hits, c_misses = [], []
            for exp in expected_certs:
                (c_hits if skill_hit(exp, discovered_certs, fuzzy_threshold) else c_misses).append(exp)
            r["cert_hits"]         = len(c_hits)
            r["cert_misses"]       = c_misses
            r["cert_accuracy_pct"] = round(len(c_hits) / len(expected_certs) * 100, 2)

    except Exception as exc:
        r["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    r["elapsed_sec"] = round(time.time() - t0, 2)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Console output
# ══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 80


def _ocr_badge(quality) -> str:
    if quality is None: return "OCR: N/A"
    if   quality >= 85: return f"OCR: {quality}% [A] ✅"
    elif quality >= 70: return f"OCR: {quality}% [B] 🟡"
    elif quality >= 50: return f"OCR: {quality}% [C] 🟠"
    elif quality >= 30: return f"OCR: {quality}% [D] 🔴"
    else:               return f"OCR: {quality}% [F] 💀"


def print_result(r: dict, idx: int, total: int):
    sa = r["skill_accuracy_pct"]
    ca = r["cert_accuracy_pct"]

    if   r["error"]:                icon = "💥"
    elif sa is None and ca is None: icon = "⚠️ "
    elif (sa or 0) >= 80:           icon = "✅"
    else:                           icon = "❌"

    ocr = r["ocr"]
    print(f"\n[{idx:>3}/{total}] {icon}  {r['filename']}")

    if r["error"]:
        print(f"       ERROR: {r['error'][:160]}")
        return

    print(f"       {_ocr_badge(r['ocr_quality'])}   "
          f"words={ocr.get('word_count','?')}   "
          f"garble={ocr.get('garbled_ratio',0):.1%}   "
          f"header={'yes' if ocr.get('has_header') else 'NO ⚠'}")

    for issue in ocr.get("issues", []):
        print(f"       ⚠  {issue}")

    raw_total    = r.get("skill_total_raw", r["skill_total"])
    excluded_n   = len(r.get("non_corpus_skills", []))
    excluded_note = f" ({excluded_n} not in corpus)" if excluded_n else ""
    skill_str = f"{sa}%" if sa is not None else "N/A (no corpus labels)"
    cert_str  = f"{ca}%" if ca is not None else "N/A"
    cert_icon = "✅" if r["cert_section_found"] else "⚠ NOT FOUND"

    print(f"       Skills  : {r['skill_hits']}/{r['skill_total']} corpus skills → {skill_str}"
          f"   (raw labels={raw_total}{excluded_note}, discovered={r['skill_count']})")
    print(f"       Certs   : {r['cert_hits']}/{r['cert_total']} → {cert_str}   "
          f"cert_section={cert_icon}   found={r['discovered_certs']}")

    if r["skill_misses"]:
        print(f"       MISSED  : {', '.join(r['skill_misses'])}")
    if r.get("non_corpus_skills"):
        print(f"       ¬CORPUS : {', '.join(r['non_corpus_skills'])}")
    if r["cert_misses"]:
        print(f"       CERT ✗  : {', '.join(r['cert_misses'])}")

    print(f"       ⏱  {r['elapsed_sec']}s")


# ══════════════════════════════════════════════════════════════════════════════
# File writers
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(results: list, path: str):
    fields = [
        "filename", "expected_job",
        "ocr_quality", "ocr_grade", "ocr_engine", "ocr_word_count",
        "ocr_garbled_ratio", "ocr_has_header", "ocr_issues", "ocr_recommendation",
        "skill_count", "skill_total", "skill_total_raw", "skill_hits",
        "skill_accuracy_pct", "skill_misses", "non_corpus_skills",
        "discovered_skills",
        "cert_count", "cert_total", "cert_hits", "cert_accuracy_pct",
        "cert_misses", "discovered_certs", "cert_section_found",
        "elapsed_sec", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            ocr = r.get("ocr", {})
            w.writerow({
                **r,
                "ocr_grade":          ocr.get("grade"),
                "ocr_engine":         ocr.get("engine"),
                "ocr_word_count":     ocr.get("word_count"),
                "ocr_garbled_ratio":  ocr.get("garbled_ratio"),
                "ocr_has_header":     ocr.get("has_header"),
                "ocr_issues":         " | ".join(ocr.get("issues", [])),
                "ocr_recommendation": ocr.get("recommendation"),
                "skill_misses":       "; ".join(r.get("skill_misses",       [])),
                "non_corpus_skills":  "; ".join(r.get("non_corpus_skills",   [])),
                "discovered_skills":  "; ".join(r.get("discovered_skills",   [])),
                "cert_misses":        "; ".join(r.get("cert_misses",         [])),
                "discovered_certs":   "; ".join(r.get("discovered_certs",    [])),
            })
    print(f"\n📄 CSV              → {path}")


def write_failures(results: list, path: str):
    lines = []
    for r in results:
        if r["error"]:
            lines += [f"[ERROR] {r['filename']}", f"  {r['error'][:300]}", ""]
            continue
        if r["skill_misses"] or r["cert_misses"]:
            lines.append(f"[MISS] {r['filename']}")
            if r["skill_misses"]:
                lines.append(f"  Skill accuracy   : {r['skill_accuracy_pct']}%  "
                             f"({r['skill_hits']}/{r['skill_total']} hard skills)")
                lines.append(f"  Missed skills    : {', '.join(r['skill_misses'])}")
                lines.append(f"  Discovered       : {', '.join(r['discovered_skills'])}")
            non_corpus = r.get("non_corpus_skills", [])
            if non_corpus:
                lines.append(f"  Not in corpus    : {', '.join(non_corpus)}")
            if r["cert_misses"]:
                lines.append(f"  Cert accuracy    : {r['cert_accuracy_pct']}%  "
                             f"({r['cert_hits']}/{r['cert_total']})")
                lines.append(f"  Missed certs     : {', '.join(r['cert_misses'])}")
                lines.append(f"  Discovered certs : {', '.join(r['discovered_certs'])}")
                lines.append(f"  Cert section     : {'found' if r['cert_section_found'] else 'NOT FOUND'}")
            lines.append("")
    if not lines:
        lines = ["All files passed — no missed skills or certs! 🎉"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📋 Failures         → {path}")


def write_corpus_filter_report(results: list, path: str):
    """
    Lists every label that was excluded per file because it wasn't found
    in the Firestore corpus. Helps you audit labels.json — if a skill
    appears here often, consider adding it to the Firestore job corpus.
    """
    total_excluded = sum(len(r.get("non_corpus_skills", [])) for r in results)
    counter: dict = {}
    for r in results:
        for s in r.get("non_corpus_skills", []):
            counter[s] = counter.get(s, 0) + 1

    lines = [
        "CORPUS FILTER REPORT",
        "=" * 72,
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Files      : {len(results)}",
        f"Total labels excluded (not in Firestore corpus): {total_excluded}",
        "",
        "These skills were in labels.json but do NOT appear in any job's",
        "required_skills in Firestore. They are excluded from accuracy scoring",
        "because the pipeline can only emit skills that exist in the corpus.",
        "",
        "If any of these should be findable, add them to the relevant job's",
        "required_skills in Firestore and re-run.",
        "",
        "TOP EXCLUDED LABELS (by frequency across all files)",
        "-" * 40,
        ]
    for skill, count in sorted(counter.items(), key=lambda x: -x[1])[:40]:
        lines.append(f"  [{count:>3}×]  {skill}")

    lines += ["", "PER-FILE BREAKDOWN", "-" * 40]
    for r in results:
        excluded = r.get("non_corpus_skills", [])
        if excluded:
            lines.append(f"\n  {r['filename']}")
            lines.append(f"    Excluded  : {', '.join(excluded)}")
            lines.append(f"    Scored on : {', '.join(r['expected_skills'])}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"🗂  Corpus filter rpt → {path}")


def write_ocr_report(results: list, path: str):
    grade_order = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}
    sorted_r = sorted(
        results,
        key=lambda r: (
            grade_order.get((r.get("ocr") or {}).get("grade"), 5),
            -(r.get("ocr_quality") or 0),
        ),
    )

    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in results:
        g = (r.get("ocr") or {}).get("grade")
        if g in grade_counts:
            grade_counts[g] += 1

    lines = [
        "OCR QUALITY REPORT",
        "=" * 72,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Files     : {len(results)}",
        "",
        "Grade bands:  A ≥85%  |  B ≥70%  |  C ≥50%  |  D ≥30%  |  F <30%",
        "=" * 72,
        "",
        "GRADE DISTRIBUTION",
        "-" * 40,
        ]
    for grade in ("A", "B", "C", "D", "F"):
        count = grade_counts[grade]
        bar   = "█" * count
        pct   = round(count / len(results) * 100) if results else 0
        lines.append(f"  [{grade}]  {bar:<35}  {count:>3} files  ({pct}%)")

    lines += ["", "PER-FILE DETAIL  (sorted worst OCR first)", "=" * 72]

    for r in sorted_r:
        ocr = r.get("ocr") or {}
        g   = ocr.get("grade", "?")
        q   = r.get("ocr_quality", "?")
        if g == "A" and not ocr.get("issues"):
            continue

        sa = r.get("skill_accuracy_pct")
        ca = r.get("cert_accuracy_pct")
        lines += [
            "",
            f"[{g}]  {r['filename']}",
            f"     OCR score     : {q}%",
            f"     Engine        : {ocr.get('engine','?')}",
            f"     Words         : {ocr.get('word_count','?')}",
            f"     Garble ratio  : {ocr.get('garbled_ratio',0):.1%}",
            f"     Skills header : {'yes' if ocr.get('has_header') else 'NOT FOUND'}",
            (f"     Skill accuracy: {sa}%" if sa is not None else "     Skill accuracy: N/A"),
            (f"     Cert accuracy : {ca}%" if ca is not None else "     Cert accuracy : N/A"),
        ]
        for issue in ocr.get("issues", []):
            lines.append(f"     ⚠  {issue}")
        rec = ocr.get("recommendation", "")
        if rec and "No action" not in rec:
            lines.append(f"     ✏  FIX: {rec}")

    needs_action = [
        r for r in results
        if (r.get("ocr") or {}).get("grade") in ("D", "F")
           or not (r.get("ocr") or {}).get("has_header")
           or (r.get("ocr") or {}).get("garbled_ratio", 0) > 0.10
    ]
    lines += [
        "",
        "=" * 72,
        f"\nFILES NEEDING OCR ATTENTION ({len(needs_action)}):",
        ]
    for r in sorted(needs_action, key=lambda x: x.get("ocr_quality") or 0):
        g = (r.get("ocr") or {}).get("grade", "?")
        lines.append(f"  [{g}] {r['filename']}  —  {r.get('ocr_quality','?')}%")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"🔬 OCR report       → {path}")


def write_summary(results: list, path: str) -> dict:
    errors    = [r for r in results if r["error"]]
    with_gt   = [r for r in results if r["skill_total"] > 0 and not r["error"]]
    no_gt     = [r for r in results if r["skill_total"] == 0 and not r["error"]]
    with_cert = [r for r in results if r["cert_total"]  > 0 and not r["error"]]

    # ── Skill stats ───────────────────────────────────────────────────────────
    accs   = [r["skill_accuracy_pct"] for r in with_gt if r["skill_accuracy_pct"] is not None]
    avg_s  = round(sum(accs) / len(accs), 2) if accs else None
    pass80 = sum(1 for a in accs if a >= 80)
    pass60 = sum(1 for a in accs if a >= 60)
    fail60 = sum(1 for a in accs if a < 60)

    # ── Soft-filter stats ─────────────────────────────────────────────────────
    total_raw_labels   = sum(r.get("skill_total_raw", r["skill_total"]) for r in results)
    total_non_corpus   = sum(len(r.get("non_corpus_skills", [])) for r in results)
    total_corpus_labels = sum(r["skill_total"] for r in results)

    # ── Cert stats ────────────────────────────────────────────────────────────
    cert_accs      = [r["cert_accuracy_pct"] for r in with_cert if r["cert_accuracy_pct"] is not None]
    avg_cert       = round(sum(cert_accs) / len(cert_accs), 2) if cert_accs else None
    cert_sec_found = sum(1 for r in results if r.get("cert_section_found"))

    # ── OCR stats ─────────────────────────────────────────────────────────────
    ocr_vals  = [r["ocr_quality"] for r in results if r["ocr_quality"] is not None]
    avg_ocr   = round(sum(ocr_vals) / len(ocr_vals), 2) if ocr_vals else None
    garb_vals = [(r.get("ocr") or {}).get("garbled_ratio", 0) for r in results]
    avg_garb  = round(sum(garb_vals) / len(garb_vals), 4) if garb_vals else None

    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in results:
        g = (r.get("ocr") or {}).get("grade")
        if g in grade_counts:
            grade_counts[g] += 1

    low_ocr     = [r["filename"] for r in results if (r.get("ocr_quality") or 101) < 50]
    no_header   = [r["filename"] for r in results if not (r.get("ocr") or {}).get("has_header")]
    high_garble = [r["filename"] for r in results if (r.get("ocr") or {}).get("garbled_ratio", 0) > 0.10]

    ocr_corr: dict = {}
    for grade in ("A", "B", "C", "D", "F"):
        gf = [r for r in results
              if (r.get("ocr") or {}).get("grade") == grade
              and r.get("skill_accuracy_pct") is not None]
        if gf:
            ocr_corr[grade] = {
                "files": len(gf),
                "avg_skill_accuracy": round(
                    sum(r["skill_accuracy_pct"] for r in gf) / len(gf), 2
                ),
            }

    miss_counter: dict = {}
    for r in with_gt:
        for m in r.get("skill_misses", []):
            miss_counter[m] = miss_counter.get(m, 0) + 1
    top_skill_misses = sorted(miss_counter.items(), key=lambda x: -x[1])[:30]

    cert_miss_counter: dict = {}
    for r in with_cert:
        for m in r.get("cert_misses", []):
            cert_miss_counter[m] = cert_miss_counter.get(m, 0) + 1
    top_cert_misses = sorted(cert_miss_counter.items(), key=lambda x: -x[1])[:20]

    hints = _calibration_hints(
        avg_skill     = avg_s,
        avg_cert      = avg_cert,
        avg_ocr       = avg_ocr,
        avg_garble    = avg_garb,
        top_s_misses  = top_skill_misses,
        top_c_misses  = top_cert_misses,
        low_ocr       = low_ocr,
        no_header     = no_header,
        high_garble   = high_garble,
        cert_found    = cert_sec_found,
        cert_gt_total = len(with_cert),
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "totals": {
            "files_evaluated":         len(results),
            "with_ground_truth":       len(with_gt),
            "without_ground_truth":    len(no_gt),
            "errors":                  len(errors),
        },
        "label_quality": {
            "total_raw_labels":       total_raw_labels,
            "total_non_corpus":       total_non_corpus,
            "total_corpus_labels":    total_corpus_labels,
            "non_corpus_pct":         round(
                total_non_corpus / max(total_raw_labels, 1) * 100, 1
            ),
            "note": (
                "non_corpus labels were in labels.json but absent from Firestore "
                "required_skills. Accuracy is scored against corpus_labels only."
            ),
        },
        "skill_accuracy": {
            "average_pct":       avg_s,
            "pass_rate_80pct":   f"{round(pass80/len(with_gt)*100,1)}%" if with_gt else "N/A",
            "pass_80_or_above":  pass80,
            "pass_60_or_above":  pass60,
            "fail_below_60":     fail60,
        },
        "cert_accuracy": {
            "files_with_cert_gt":       len(with_cert),
            "files_cert_section_found": cert_sec_found,
            "average_pct":              avg_cert,
        },
        "ocr": {
            "average_quality_pct":    avg_ocr,
            "average_garbled_ratio":  avg_garb,
            "grade_distribution":     grade_counts,
            "files_grade_D_or_F":     grade_counts["D"] + grade_counts["F"],
            "files_below_50pct":      len(low_ocr),
            "files_no_skills_header": len(no_header),
            "files_high_garble":      len(high_garble),
            "low_ocr_files":          low_ocr,
            "no_header_files":        no_header,
            "high_garble_files":      high_garble,
            "ocr_vs_skill_accuracy":  ocr_corr,
        },
        "top_missed_skills": [{"skill": s, "count": c} for s, c in top_skill_misses],
        "top_missed_certs":  [{"cert": c,  "count": n} for c, n in top_cert_misses],
        "calibration_hints": hints,
        "error_files":       [r["filename"] for r in errors],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 Summary JSON     → {path}")
    return summary


def _calibration_hints(
        avg_skill, avg_cert, avg_ocr, avg_garble,
        top_s_misses, top_c_misses,
        low_ocr, no_header, high_garble,
        cert_found, cert_gt_total,
) -> list:
    hints = []
    if avg_skill is not None and avg_skill < 70:
        hints.append(
            f"⚠️  Average skill accuracy {avg_skill}% (<70%). "
            "Check required_skills in Firestore/mock_jobs and labels.json skill pool."
        )
    if avg_cert is not None and avg_cert < 70:
        hints.append(
            f"⚠️  Average cert accuracy {avg_cert}% (<70%). "
            "Expand _CERT_REGEX vendor prefixes in matcher.py."
        )
    if cert_gt_total > 0 and cert_found < cert_gt_total:
        hints.append(
            f"💡 {cert_gt_total - cert_found} resume(s) with cert ground truth had no detectable "
            "Certifications section. Expand _CERT_HEADERS in matcher.py."
        )
    if avg_ocr is not None and avg_ocr < 55:
        hints.append(
            "⚠️  Average OCR quality <55%. Enable EasyOCR as primary engine "
            "and apply image pre-processing (deskew + CLAHE + 300 DPI)."
        )
    if low_ocr:
        hints.append(
            f"⚠️  {len(low_ocr)} file(s) below 50% OCR: " +
            ", ".join(low_ocr[:5]) + (" ..." if len(low_ocr) > 5 else "")
        )
    if no_header:
        hints.append(
            f"💡 {len(no_header)} resume(s) missing a Skills section header: " +
            ", ".join(no_header[:5])
        )
    if avg_garble is not None and avg_garble > 0.08:
        hints.append(f"⚠️  Average garble ratio {avg_garble:.1%} (>8%) — OCR producing noise.")
    if high_garble:
        hints.append(
            f"🔴 {len(high_garble)} file(s) with >10% garbled tokens: " +
            ", ".join(high_garble[:5])
        )
    if top_s_misses:
        top5 = [s for s, _ in top_s_misses[:5]]
        hints.append(
            f"💡 Top missed skills: {', '.join(top5)}. "
            "Verify these appear in Firestore/mock required_skills or labels skill pool."
        )
    if top_c_misses:
        top3 = [c for c, _ in top_c_misses[:3]]
        hints.append(
            f"💡 Top missed certs: {', '.join(top3)}. "
            "Add vendor prefixes to _CERT_REGEX in matcher.py."
        )
    if not hints:
        hints.append("✅ Pipeline looks healthy — no major calibration issues detected.")
    return hints


def print_summary(summary: dict):
    t  = summary["totals"]
    sa = summary["skill_accuracy"]
    ca = summary["cert_accuracy"]
    oc = summary["ocr"]
    gd = oc["grade_distribution"]
    lq = summary.get("label_quality", {})

    print(f"\n{'='*80}")
    print("AGGREGATE EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Files evaluated    : {t['files_evaluated']}")
    print(f"  With ground truth  : {t['with_ground_truth']}")
    print(f"  Without GT         : {t['without_ground_truth']}")
    print(f"  Errors             : {t['errors']}")

    if lq:
        print(f"\n  ── Label Quality (corpus filter) ────────────────────────────────────")
        print(f"     Raw labels total  : {lq.get('total_raw_labels')}")
        print(f"     Not in corpus     : {lq.get('total_non_corpus')}  "
              f"({lq.get('non_corpus_pct')}% of labels)")
        print(f"     Corpus labels used: {lq.get('total_corpus_labels')}  (scored against these only)")

    print(f"\n  ── Skill Accuracy (hard skills only) ────────────────────────────────")
    print(f"     Average           : {sa['average_pct']}%")
    print(f"     Pass ≥80%         : {sa['pass_80_or_above']}  ({sa['pass_rate_80pct']})")
    print(f"     Pass ≥60%         : {sa['pass_60_or_above']}")
    print(f"     Fail  <60%        : {sa['fail_below_60']}")

    print(f"\n  ── Cert Accuracy ────────────────────────────────────────────────────")
    print(f"     Files with cert GT: {ca['files_with_cert_gt']}")
    print(f"     Cert section found: {ca['files_cert_section_found']}")
    print(f"     Average           : {ca['average_pct']}%")

    print(f"\n  ── OCR Quality ──────────────────────────────────────────────────────")
    print(f"     Average score     : {oc['average_quality_pct']}%")
    print(f"     Avg garble ratio  : {(oc['average_garbled_ratio'] or 0):.1%}")
    print(f"     No skills header  : {oc['files_no_skills_header']} file(s)")
    print(f"     High garble       : {oc['files_high_garble']} file(s)")
    print(f"     Grades            : A={gd['A']}  B={gd['B']}  C={gd['C']}  D={gd['D']}  F={gd['F']}")

    corr = oc.get("ocr_vs_skill_accuracy", {})
    if corr:
        print(f"\n  ── OCR Grade → Avg Skill Accuracy ───────────────────────────────────")
        for grade in ("A", "B", "C", "D", "F"):
            if grade in corr:
                c = corr[grade]
                print(f"     [{grade}]  {c['avg_skill_accuracy']}%  ({c['files']} files)")

    print(f"\n  ── Top Missed Skills ────────────────────────────────────────────────")
    for e in summary["top_missed_skills"][:10]:
        print(f"     [{e['count']:>3}×]  {e['skill']}")

    print(f"\n  ── Top Missed Certs ─────────────────────────────────────────────────")
    for e in summary["top_missed_certs"][:10]:
        print(f"     [{e['count']:>3}×]  {e['cert']}")

    print(f"\n  ── Calibration Hints ────────────────────────────────────────────────")
    for h in summary["calibration_hints"]:
        print(f"     {h}")

    print(f"{'='*80}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    resume_dir  = Path(args.resumes)
    labels_path = Path(args.labels)
    out_dir     = Path(args.out_dir)

    if not resume_dir.is_dir():
        sys.exit(f"ERROR: --resumes '{args.resumes}' is not a directory.")
    if not labels_path.exists():
        sys.exit(f"ERROR: --labels '{args.labels}' not found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data: dict = json.load(f)

    label_skill_pool: set = set()
    label_cert_pool:  set = set()
    for entry in labels_data.values():
        for s in entry.get("expected_skills", []):
            if s.strip(): label_skill_pool.add(s.strip())
        for c in entry.get("expected_certs", []):
            if c.strip(): label_cert_pool.add(c.strip())
    label_skill_pool = list(label_skill_pool)
    label_cert_pool  = list(label_cert_pool)

    print(f"✅ Labels loaded   : {len(labels_data)} entries")
    print(f"   Skill pool      : {len(label_skill_pool)} unique skills")
    print(f"   Cert pool       : {len(label_cert_pool)} unique certs")

    if args.mock_jobs:
        print(f"📦 Loading jobs from mock file: {args.mock_jobs}")
        jobs_corpus = load_jobs_mock(args.mock_jobs)
    else:
        collection  = os.getenv("FIRESTORE_JOBS_COLLECTION", "jobs")
        print(f"🔥 Fetching jobs from Firestore collection: '{collection}'")
        jobs_corpus = load_jobs_firestore(collection)
    print(f"✅ Jobs loaded     : {len(jobs_corpus)}")

    # Build corpus_norm once so every evaluate_file call uses the same dict
    from matcher import normalize as _norm
    corpus_norm: dict[str, str] = {}
    for job in jobs_corpus:
        for s in job.get("required_skills", []):
            corpus_norm[_norm(s.strip())] = s.strip()
    print(f"   Corpus skills   : {len(corpus_norm)} unique (normalised)\n")

    all_pdfs = sorted(resume_dir.glob("*.pdf"))
    if args.resume_filter:
        wanted   = {f.strip() for f in args.resume_filter.split(",")}
        all_pdfs = [p for p in all_pdfs if p.name in wanted]
        print(f"🔍 Filter applied  — {len(all_pdfs)} file(s) selected")
    if args.limit:
        all_pdfs = all_pdfs[:args.limit]
        print(f"⚡ Limit applied   — first {len(all_pdfs)} file(s)")
    if not all_pdfs:
        sys.exit("ERROR: No PDF files found matching the given criteria.")

    total = len(all_pdfs)
    print(f"📂 Resumes found   : {total}\n{SEP}")

    results      = []
    unlabeled    = []
    global_start = time.time()

    for idx, pdf_path in enumerate(all_pdfs, start=1):
        fname       = pdf_path.name
        label_entry = labels_data.get(fname, {})
        if not label_entry:
            unlabeled.append(fname)

        r = evaluate_file(
            pdf_path         = str(pdf_path),
            label_entry      = label_entry,
            jobs_corpus      = jobs_corpus,
            label_skill_pool = label_skill_pool,
            label_cert_pool  = label_cert_pool,
            fuzzy_threshold  = args.fuzzy_threshold,
            corpus_norm      = corpus_norm,
        )
        results.append(r)
        print_result(r, idx, total)

    elapsed_total = round(time.time() - global_start, 1)
    print(f"\n{SEP}")
    print(f"✅ Done in {elapsed_total}s  ({total} files)")
    if unlabeled:
        print(f"⚠️  {len(unlabeled)} file(s) had no labels.json entry: "
              + ", ".join(unlabeled[:5])
              + (" ..." if len(unlabeled) > 5 else ""))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_csv(                results, str(out_dir / f"eval_results_{ts}.csv"))
    summary = write_summary(  results, str(out_dir / f"eval_summary_{ts}.json"))
    write_failures(           results, str(out_dir / f"eval_failures_{ts}.txt"))
    write_ocr_report(         results, str(out_dir / f"eval_ocr_report_{ts}.txt"))
    write_corpus_filter_report(results, str(out_dir / f"eval_corpus_filter_{ts}.txt"))
    print_summary(summary)


if __name__ == "__main__":
    main()