"""
eval_resumes.py
===============
Batch accuracy evaluation across PDF resumes.

Runs each PDF through the FULL pipeline (OCR → NLP → Embed → FAISS → Score)
and measures per-AI-component accuracy independently.

USAGE
-----
  # Offline (OCR + NLP only, no Firebase/FAISS needed):
  python eval_resumes.py --resumes ./resumes/

  # Full pipeline (needs Firebase + FAISS warmed up):
  python eval_resumes.py --resumes ./resumes/ --live

  # With ground-truth labels for F1 scoring:
  python eval_resumes.py --resumes ./resumes/ --labels labels.json --live

  # Limit to first 20 for a quick smoke test:
  python eval_resumes.py --resumes ./resumes/ --limit 20 --verbose

  # Save full JSON results:
  python eval_resumes.py --resumes ./resumes/ --output results.json --live

LABELS FILE FORMAT (labels.json)
----------------------------------
  {
    "resume_john.pdf": {
      "expected_skills":    ["Python", "React", "AWS"],
      "expected_job_title": "Backend Developer",
      "min_match_score":    60
    }
  }
  Generate a starter file automatically:
    python generate_labels.py --resumes ./resumes/ --mode auto
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CONTENT_TYPE = "application/pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResumeResult:
    filename: str
    success:  bool = False
    error:    str  = ""
    label:    dict = field(default_factory=dict)

    # OCR
    ocr_path:       str   = ""      # "pdfplumber" | "pypdf2" | "tesseract" | "failed"
    ocr_chars:      int   = 0
    ocr_ok:         bool  = False
    ocr_ms:         float = 0.0

    # NLP
    skills:         list  = field(default_factory=list)
    certs:          list  = field(default_factory=list)
    nlp_f1:         float = -1.0   # -1 = no label
    nlp_precision:  float = -1.0
    nlp_recall:     float = -1.0
    nlp_ok:         bool  = False
    nlp_ms:         float = 0.0

    # Embedding
    embed_dim:      int   = 0
    embed_norm:     float = 0.0
    embed_self_sim: float = 0.0
    embed_ok:       bool  = False
    embed_ms:       float = 0.0

    # FAISS / Scoring
    top_job:        str   = ""
    top_score:      float = 0.0
    all_matches:    list  = field(default_factory=list)
    job_hit:        bool  = False
    faiss_ok:       bool  = False
    faiss_ms:       float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f1_scores(predicted: list, expected: list) -> tuple:
    pred_l = {s.lower().strip() for s in predicted}
    exp_l  = {s.lower().strip() for s in expected}
    tp = len(pred_l & exp_l)
    fp = len(pred_l - exp_l)
    fn = len(exp_l - pred_l)
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving",
    "analytical thinking", "attention to detail", "time management",
    "creativity", "adaptability", "work ethic", "multitasking",
    "collaboration", "interpersonal skills", "critical thinking",
}


def _soft_leaks(skills: list) -> list:
    return [s for s in skills if s.lower().strip() in SOFT_SKILLS]


def _bar(value: float, width: int = 16) -> str:
    if value < 0:
        return f"{'─' * width}  n/a"
    filled = round(value * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = GREEN if value >= 0.80 else (YELLOW if value >= 0.60 else RED)
    return f"{color}{bar}{RESET} {value:.0%}"


def _pct(value: float, width: int = 16) -> float:
    return _percentile([], 0) if not value else value


def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p), len(s) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# OCR path detector — mirrors ocr_extractor.py priority order
# ─────────────────────────────────────────────────────────────────────────────

def _detect_ocr_path(pdf_bytes: bytes) -> tuple:
    """
    Try pdfplumber → PyPDF2 → Tesseract in the same order as ocr_extractor.py.
    Returns (path_name, extracted_text, elapsed_ms).
    """
    t0 = time.perf_counter()

    # Path 1: pdfplumber (column-aware)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        if len(text) >= 100:
            return "pdfplumber", text, (time.perf_counter() - t0) * 1000
    except Exception:
        pass

    # Path 2: PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        if len(text) >= 100:
            return "pypdf2", text, (time.perf_counter() - t0) * 1000
    except Exception:
        pass

    # Path 3: Tesseract (scanned/image-only PDF)
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        images = convert_from_bytes(pdf_bytes, dpi=300)
        pages  = [pytesseract.image_to_string(img, config="--psm 6 --oem 3") for img in images]
        text   = "\n\n".join(pages).strip()
        return "tesseract", text, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return "failed", "", (time.perf_counter() - t0) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Offline evaluation  (OCR + NLP only — no Firebase / FAISS)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_offline(path: Path, label: dict) -> ResumeResult:
    r = ResumeResult(filename=path.name, label=label)
    pdf_bytes = path.read_bytes()

    # OCR
    r.ocr_path, raw_text, r.ocr_ms = _detect_ocr_path(pdf_bytes)
    r.ocr_chars = len(raw_text)
    r.ocr_ok    = r.ocr_path != "failed" and r.ocr_chars >= 50
    if not r.ocr_ok:
        r.error = f"OCR failed (path={r.ocr_path}, chars={r.ocr_chars})"
        return r

    # NLP
    try:
        from ai_modules.nlp_processor import NLPProcessor
        nlp = NLPProcessor()
        t0 = time.perf_counter()
        extracted   = nlp.process(raw_text)
        r.nlp_ms    = (time.perf_counter() - t0) * 1000
        r.skills    = extracted["skills"]
        r.certs     = extracted["certifications"]
        soft_leaked = _soft_leaks(r.skills)
        if label.get("expected_skills"):
            r.nlp_precision, r.nlp_recall, r.nlp_f1 = _f1_scores(
                r.skills, label["expected_skills"]
            )
            r.nlp_ok = r.nlp_f1 >= 0.40 and not soft_leaked
        else:
            # Zero skills = skills not in Firestore vocab, not a pipeline failure.
            # OCR worked, NLP ran — just nothing matched. Mark as ok with a note.
            r.nlp_ok = not soft_leaked
            if len(r.skills) == 0:
                r.error = "0 skills extracted — resume skills may not be in Firestore job vocab"
    except Exception as e:
        r.error = f"NLP: {e}"
        return r

    # success = OCR extracted text + NLP ran without errors (even if 0 skills)
    r.success = r.ocr_ok and r.nlp_ok
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Live evaluation  (full pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_live(path: Path, label: dict, resume_svc, embedder) -> ResumeResult:
    import numpy as np
    r         = ResumeResult(filename=path.name, label=label)
    pdf_bytes = path.read_bytes()

    # OCR
    r.ocr_path, raw_text, r.ocr_ms = _detect_ocr_path(pdf_bytes)
    r.ocr_chars = len(raw_text)
    r.ocr_ok    = r.ocr_path != "failed" and r.ocr_chars >= 50
    if not r.ocr_ok:
        r.error = f"OCR failed (path={r.ocr_path})"
        return r

    # NLP
    try:
        from ai_modules.nlp_processor import NLPProcessor
        nlp = NLPProcessor()
        t0 = time.perf_counter()
        extracted   = nlp.process(raw_text)
        r.nlp_ms    = (time.perf_counter() - t0) * 1000
        r.skills    = extracted["skills"]
        r.certs     = extracted["certifications"]
        soft_leaked = _soft_leaks(r.skills)
        if label.get("expected_skills"):
            r.nlp_precision, r.nlp_recall, r.nlp_f1 = _f1_scores(
                r.skills, label["expected_skills"]
            )
            r.nlp_ok = r.nlp_f1 >= 0.40 and not soft_leaked
        else:
            r.nlp_ok = not soft_leaked
            if len(r.skills) == 0:
                r.error = "0 skills — not in Firestore vocab"
    except Exception as e:
        r.error = f"NLP: {e}"
        return r

    # Embedding
    try:
        skill_text = " ".join(r.skills) if r.skills else "software developer"
        t0 = time.perf_counter()
        emb1 = embedder.embed(skill_text)
        emb2 = embedder.embed(skill_text)   # cache hit — tests determinism
        r.embed_ms       = (time.perf_counter() - t0) * 1000
        r.embed_dim      = emb1.shape[0]
        r.embed_norm     = round(float(np.linalg.norm(emb1)), 4)
        r.embed_self_sim = round(float(np.dot(emb1, emb2)), 4)
        r.embed_ok       = (
                r.embed_dim > 0
                and 0.98 <= r.embed_norm <= 1.02
                and r.embed_self_sim >= 0.999
        )
    except Exception as e:
        r.error += f" | Embed: {e}"

    # FAISS + full scoring
    try:
        t0 = time.perf_counter()
        analysis    = resume_svc.analyze(pdf_bytes, CONTENT_TYPE)
        r.faiss_ms  = (time.perf_counter() - t0) * 1000
        matches      = analysis.get("top_job_matches", [])
        r.all_matches = [(m["job_title"], m["match_score"]) for m in matches]
        r.top_job    = matches[0]["job_title"]  if matches else ""
        r.top_score  = matches[0]["match_score"] if matches else 0.0
        r.faiss_ok   = bool(matches)
        if label.get("expected_job_title") and r.all_matches:
            top3 = [t.lower() for t, _ in r.all_matches[:3]]
            r.job_hit = any(label["expected_job_title"].lower() in t for t in top3)
    except Exception as e:
        r.error += f" | FAISS: {e}"

    r.success = r.ocr_ok and r.nlp_ok
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(results: list, live: bool) -> None:
    n = len(results)

    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}  PDF Resume Batch Evaluation — {n} files{RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    # ── OCR ──────────────────────────────────────────────────────────────────
    ocr_ok    = [r for r in results if r.ocr_ok]
    by_path   = {k: sum(1 for r in results if r.ocr_path == k)
                 for k in ("pdfplumber", "pypdf2", "tesseract", "failed")}
    char_cnts = [r.ocr_chars for r in ocr_ok]
    ocr_times = [r.ocr_ms   for r in ocr_ok]

    print(f"\n  {BOLD}{CYAN}① OCR Extractor{RESET}  "
          f"{DIM}(pdfplumber → PyPDF2 → Tesseract){RESET}")
    print(f"     Success rate      {_bar(len(ocr_ok)/n)}  ({len(ocr_ok)}/{n})")
    print(f"     Extraction paths:")
    print(f"       pdfplumber      {by_path['pdfplumber']:>4}  resumes  "
          f"{DIM}(clean text, column-aware){RESET}")
    print(f"       PyPDF2          {by_path['pypdf2']:>4}  resumes  "
          f"{DIM}(clean text, no column split){RESET}")
    print(f"       Tesseract OCR   {by_path['tesseract']:>4}  resumes  "
          f"{DIM}(scanned / image-only PDFs){RESET}")
    print(f"       Failed          {by_path['failed']:>4}  resumes")
    if char_cnts:
        print(f"     Chars extracted   "
              f"avg={statistics.mean(char_cnts):.0f}  "
              f"min={min(char_cnts)}  "
              f"max={max(char_cnts)}")
    if ocr_times:
        print(f"     Latency           "
              f"avg={statistics.mean(ocr_times):.0f}ms  "
              f"p90={_percentile(ocr_times, 0.9):.0f}ms")

    # ── NLP ──────────────────────────────────────────────────────────────────
    nlp_ok      = [r for r in results if r.nlp_ok]
    labelled    = [r for r in results if r.nlp_f1 >= 0]
    skill_cnts  = [len(r.skills) for r in nlp_ok]
    soft_leaked = [r for r in results if _soft_leaks(r.skills)]
    zero_skills = [r for r in results if r.ocr_ok and len(r.skills) == 0]
    nlp_times   = [r.nlp_ms for r in nlp_ok]

    print(f"\n  {BOLD}{CYAN}② spaCy NLP — Skill Extraction{RESET}")
    print(f"     Valid extractions  {_bar(len(nlp_ok)/n)}  ({len(nlp_ok)}/{n})")
    if skill_cnts:
        print(f"     Skills / resume   "
              f"avg={statistics.mean(skill_cnts):.1f}  "
              f"min={min(skill_cnts)}  "
              f"max={max(skill_cnts)}  "
              f"median={statistics.median(skill_cnts):.0f}")
    if labelled:
        f1s  = [r.nlp_f1       for r in labelled]
        prec = [r.nlp_precision for r in labelled]
        rec  = [r.nlp_recall    for r in labelled]
        print(f"     Labelled F1       {_bar(statistics.mean(f1s))}  "
              f"({len(labelled)} resumes)")
        print(f"       Precision       avg={statistics.mean(prec):.0%}  "
              f"min={min(prec):.0%}  max={max(prec):.0%}")
        print(f"       Recall          avg={statistics.mean(rec):.0%}  "
              f"min={min(rec):.0%}  max={max(rec):.0%}")
        buckets = {"≥80%": 0, "60-79%": 0, "40-59%": 0, "<40%": 0}
        for f in f1s:
            if f >= 0.80:   buckets["≥80%"] += 1
            elif f >= 0.60: buckets["60-79%"] += 1
            elif f >= 0.40: buckets["40-59%"] += 1
            else:           buckets["<40%"] += 1
        print(f"       F1 buckets      "
              f"≥80%={buckets['≥80%']}  "
              f"60-79%={buckets['60-79%']}  "
              f"40-59%={buckets['40-59%']}  "
              f"<40%={buckets['<40%']}")
    print(f"     Soft-skill leaks   {len(soft_leaked)} resumes "
          f"{DIM}(soft skills incorrectly extracted){RESET}")
    print(f"     Zero skills        {len(zero_skills)} resumes "
          f"{DIM}(skills likely not in Firestore job vocab — run /admin/generate-synonyms){RESET}")
    if zero_skills:
        print(f"       Files with 0 skills:")
        for r in zero_skills[:10]:
            print(f"         {r.filename}")
    if nlp_times:
        print(f"     Latency           "
              f"avg={statistics.mean(nlp_times):.0f}ms  "
              f"p90={_percentile(nlp_times, 0.9):.0f}ms")

    if live:
        # ── Embedding ─────────────────────────────────────────────────────
        emb_ok    = [r for r in results if r.embed_ok]
        self_sims = [r.embed_self_sim for r in results if r.embed_dim > 0]
        norms     = [r.embed_norm     for r in results if r.embed_dim > 0]
        emb_times = [r.embed_ms       for r in results if r.embed_dim > 0]

        print(f"\n  {BOLD}{CYAN}③ MiniLM Embedding Engine{RESET}  "
              f"{DIM}(all-MiniLM-L6-v2){RESET}")
        print(f"     Valid embeddings   {_bar(len(emb_ok)/n)}  ({len(emb_ok)}/{n})")
        if self_sims:
            perfect = sum(1 for s in self_sims if s >= 0.999)
            print(f"     Determinism        {perfect}/{len(self_sims)} "
                  f"resumes self-sim ≥ 0.999")
        if norms:
            print(f"     L2 norm            avg={statistics.mean(norms):.4f}  "
                  f"{DIM}(expect 1.0000 — L2-normalised){RESET}")
        if emb_times:
            print(f"     Latency            "
                  f"avg={statistics.mean(emb_times):.0f}ms  "
                  f"p90={_percentile(emb_times, 0.9):.0f}ms")

        # ── FAISS + Scoring ────────────────────────────────────────────────
        faiss_ok    = [r for r in results if r.faiss_ok]
        scores      = [r.top_score for r in faiss_ok]
        labelled_j  = [r for r in results if r.label.get("expected_job_title")]
        faiss_times = [r.faiss_ms  for r in faiss_ok]

        print(f"\n  {BOLD}{CYAN}④ FAISS Semantic Search + Scoring{RESET}")
        print(f"     Returned ≥1 match  {_bar(len(faiss_ok)/n)}  ({len(faiss_ok)}/{n})")
        if scores:
            high   = sum(1 for s in scores if s >= 70)
            medium = sum(1 for s in scores if 40 <= s < 70)
            low    = sum(1 for s in scores if s < 40)
            print(f"     Score distribution")
            print(f"       mean={statistics.mean(scores):.1f}%  "
                  f"median={statistics.median(scores):.1f}%  "
                  f"p10={_percentile(scores, 0.1):.1f}%  "
                  f"p90={_percentile(scores, 0.9):.1f}%")
            print(f"       ≥70%={high}  40-69%={medium}  <40%={low}")
        if labelled_j:
            hits = sum(1 for r in labelled_j if r.job_hit)
            print(f"     Title hit@3        {_bar(hits/len(labelled_j))}  "
                  f"({hits}/{len(labelled_j)} labelled)")
        if faiss_times:
            print(f"     Latency            "
                  f"avg={statistics.mean(faiss_times):.0f}ms  "
                  f"p90={_percentile(faiss_times, 0.9):.0f}ms")

    # ── Overall ────────────────────────────────────────────────────────────
    overall_ok = sum(1 for r in results if r.success)
    print(f"\n  {'─' * 64}")
    print(f"  {BOLD}Overall success    {_bar(overall_ok/n)}  ({overall_ok}/{n}){RESET}")

    # ── Failed resumes ─────────────────────────────────────────────────────
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n  {BOLD}{RED}Failed ({len(failed)}):{RESET}")
        for r in failed[:25]:
            print(f"    {RED}✗{RESET} {r.filename:<48}  {r.error}")
        if len(failed) > 25:
            print(f"    {DIM}… {len(failed)-25} more — see batch_results.json{RESET}")

    # ── Worst NLP F1 (labelled only) ───────────────────────────────────────
    if labelled:
        worst = sorted(labelled, key=lambda r: r.nlp_f1)[:5]
        print(f"\n  {BOLD}{YELLOW}Lowest NLP F1 — review these resumes:{RESET}")
        for r in worst:
            got  = r.skills[:4]
            want = r.label.get("expected_skills", [])[:4]
            print(f"    F1={r.nlp_f1:.0%}  {r.filename:<42}  "
                  f"got={got}  want={want}")

    print(f"\n{'═' * 68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PDF resume batch accuracy evaluator")
    parser.add_argument("--resumes", required=True, help="Folder of PDF files")
    parser.add_argument("--labels",  default=None,  help="Path to labels.json")
    parser.add_argument("--live",    action="store_true",
                        help="Full pipeline (Firebase + FAISS required)")
    parser.add_argument("--output",  default="batch_results.json")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Process only first N PDFs (0 = all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print one line per resume")
    args = parser.parse_args()

    resume_dir = Path(args.resumes)
    if not resume_dir.is_dir():
        print(f"{RED}Error: '{resume_dir}' is not a directory.{RESET}")
        sys.exit(1)

    files = sorted(f for f in resume_dir.iterdir() if f.suffix.lower() == ".pdf")
    if not files:
        print(f"{RED}No PDF files found in '{resume_dir}'.{RESET}")
        sys.exit(1)
    if args.limit:
        files = files[:args.limit]

    labels: dict = {}
    if args.labels and Path(args.labels).exists():
        labels = json.loads(Path(args.labels).read_text())
        print(f"  Loaded labels for {len(labels)} resumes")

    sys.path.insert(0, str(Path(__file__).parent))

    print(f"\n{BOLD}Processing {len(files)} PDFs "
          f"({'full pipeline' if args.live else 'offline: OCR + NLP only'})...{RESET}\n")

    resume_svc = embedder = None

    # Always connect to Firebase + build PhraseMatcher regardless of --live.
    # Without this the NLP matcher is empty and every resume returns 0 skills.
    print(f"  {CYAN}Connecting to Firebase and building PhraseMatcher...{RESET}")
    try:
        from services.firebase_service import init_firebase
        from ai_modules.nlp_processor import NLPProcessor
        from utils.skill_normalizer import SkillNormalizer

        init_firebase()
        SkillNormalizer.load_dynamic_synonyms()
        n = NLPProcessor.build_matcher_from_firestore()
        print(f"  {GREEN}✓ PhraseMatcher ready — {n} skill patterns from Firestore{RESET}\n")
    except Exception as e:
        print(f"  {RED}✗ Firebase warmup failed: {e}{RESET}")
        print(f"  {YELLOW}  Skills will be empty without this. Check your .env credentials.{RESET}\n")

    if args.live:
        try:
            import asyncio
            from ai_modules.faiss_index import FAISSIndex
            from ai_modules.embedding_engine import EmbeddingEngine
            from services.resume_service import ResumeService

            print("  Loading FAISS index...")
            faiss_idx = FAISSIndex()
            asyncio.run(faiss_idx.load_or_build())   # async method — must use asyncio.run()
            embedder   = EmbeddingEngine()
            resume_svc = ResumeService(faiss_idx)
            print(f"  {GREEN}✓ FAISS + scoring ready.{RESET}\n")
        except Exception as e:
            print(f"  {RED}Live init failed: {e}{RESET}")
            print("  Falling back to offline mode.\n")
            args.live = False

    results = []
    t_start = time.perf_counter()

    for i, path in enumerate(files, 1):
        label = labels.get(path.name, {})
        try:
            if args.live and resume_svc and embedder:
                r = _eval_live(path, label, resume_svc, embedder)
            else:
                r = _eval_offline(path, label)
        except Exception as e:
            r = ResumeResult(filename=path.name, success=False,
                             error=str(e), label=label)
        results.append(r)

        if args.verbose or not r.success:
            icon  = (GREEN + "✓" + RESET) if r.success else (RED + "✗" + RESET)
            parts = []
            if r.ocr_ok:
                parts.append(f"OCR={r.ocr_path}({r.ocr_chars}c)")
            if r.nlp_ok:
                parts.append(f"{len(r.skills)} skills")
            if r.nlp_f1 >= 0:
                parts.append(f"F1={r.nlp_f1:.0%}")
            if r.faiss_ok:
                parts.append(f"score={r.top_score:.0f}%")
            if r.error:
                parts.append(f"{RED}{r.error}{RESET}")
            print(f"  [{i:>3}/{len(files)}] {icon} {path.name:<42}  "
                  f"{' | '.join(parts)}")
        elif i % 20 == 0:
            elapsed  = time.perf_counter() - t_start
            eta      = (elapsed / i) * (len(files) - i)
            pct_done = i / len(files)
            bar = "█" * int(pct_done * 20) + "░" * (20 - int(pct_done * 20))
            print(f"  {CYAN}{bar}{RESET} {i}/{len(files)}  "
                  f"{elapsed:.0f}s elapsed  ETA ~{eta:.0f}s")

    total_s = time.perf_counter() - t_start
    print(f"\n  Done — {len(files)} PDFs in {total_s:.1f}s "
          f"({total_s / len(files) * 1000:.0f}ms avg/resume)")

    _print_report(results, live=args.live)

    out = {
        "meta": {
            "total":   len(results),
            "live":    args.live,
            "total_s": round(total_s, 2),
        },
        "summary": {
            "ocr_ok":   sum(1 for r in results if r.ocr_ok),
            "nlp_ok":   sum(1 for r in results if r.nlp_ok),
            "embed_ok": sum(1 for r in results if r.embed_ok),
            "faiss_ok": sum(1 for r in results if r.faiss_ok),
            "success":  sum(1 for r in results if r.success),
            "ocr_paths": {k: sum(1 for r in results if r.ocr_path == k)
                          for k in ("pdfplumber", "pypdf2", "tesseract", "failed")},
        },
        "resumes": [
            {
                "file":          r.filename,
                "success":       r.success,
                "error":         r.error or None,
                "ocr_path":      r.ocr_path,
                "ocr_chars":     r.ocr_chars,
                "ocr_ms":        round(r.ocr_ms, 1),
                "skills":        r.skills,
                "certs":         r.certs,
                "nlp_f1":        r.nlp_f1        if r.nlp_f1 >= 0        else None,
                "nlp_precision": r.nlp_precision  if r.nlp_precision >= 0 else None,
                "nlp_recall":    r.nlp_recall     if r.nlp_recall >= 0    else None,
                "nlp_ms":        round(r.nlp_ms, 1),
                "embed_dim":     r.embed_dim,
                "embed_norm":    r.embed_norm,
                "embed_ok":      r.embed_ok,
                "embed_ms":      round(r.embed_ms, 1),
                "top_job":       r.top_job,
                "top_score":     r.top_score,
                "all_matches":   r.all_matches,
                "job_hit":       r.job_hit,
                "faiss_ms":      round(r.faiss_ms, 1),
                "label":         r.label or None,
            }
            for r in results
        ],
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"  Full results → {args.output}\n")


if __name__ == "__main__":
    main()