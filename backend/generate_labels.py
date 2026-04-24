"""
generate_labels.py
==================
Bootstraps a labels.json file from your 200 real PDF resumes.

Connects to Firebase, builds the PhraseMatcher (same as /warmup does),
then runs OCR + NLP on each resume so extracted skills are real — not empty.

USAGE
-----
  # Auto mode — extracts real skills via Firebase + NLP (recommended)
  python generate_labels.py --resumes ./resumes/ --mode auto

  # Template mode — creates empty entries for you to fill manually
  python generate_labels.py --resumes ./resumes/ --mode template

  # Limit to first 10 to test quickly
  python generate_labels.py --resumes ./resumes/ --mode auto --limit 10

OUTPUT: labels.json
  {
    "resume_john.pdf": {
      "expected_skills":    ["Python", "React", "AWS"],   <- verify/correct these
      "expected_job_title": "",                            <- fill this in manually
      "min_match_score":    50
    }
  }

AFTER GENERATION
  1. Open labels.json
  2. Spot-check the extracted skills — correct obvious errors
  3. Fill in expected_job_title where you know the right role
  4. Run the evaluator:
       python eval_resumes.py --resumes ./resumes/ --labels labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _warmup_services() -> bool:
    """
    Run the same startup sequence as main.py /warmup:
      1. init_firebase()
      2. SkillNormalizer.load_dynamic_synonyms()
      3. NLPProcessor.build_matcher_from_firestore()

    Returns True if warmup succeeded, False if Firebase is unavailable.
    """
    print(f"\n  {CYAN}Connecting to Firebase and building PhraseMatcher...{RESET}")

    try:
        from services.firebase_service import init_firebase
        init_firebase()
        print(f"  {GREEN}✓{RESET} Firebase connected")
    except Exception as e:
        print(f"  {RED}✗ Firebase failed: {e}{RESET}")
        print(f"\n  {YELLOW}Tip: make sure your .env has correct credentials:{RESET}")
        print(f"       FIREBASE_SERVICE_ACCOUNT_PATH=../firebase_service_account.json")
        print(f"       FIREBASE_PROJECT_ID=your-project-id")
        return False

    try:
        from utils.skill_normalizer import SkillNormalizer
        SkillNormalizer.load_dynamic_synonyms()
        print(f"  {GREEN}✓{RESET} Skill synonyms loaded")
    except Exception as e:
        print(f"  {YELLOW}~ Synonyms load skipped (non-fatal): {e}{RESET}")

    try:
        from ai_modules.nlp_processor import NLPProcessor
        n = NLPProcessor.build_matcher_from_firestore()
        print(f"  {GREEN}✓{RESET} PhraseMatcher built — {n} skill patterns from Firestore")
        if n == 0:
            print(f"  {YELLOW}  Warning: 0 patterns loaded. Check your Firestore 'jobs' collection has data.{RESET}")
    except Exception as e:
        print(f"  {RED}✗ PhraseMatcher build failed: {e}{RESET}")
        return False

    print()
    return True


def _process_resume(path: Path, nlp) -> tuple[list, list, str]:
    """
    Run OCR + NLP on one PDF. Returns (skills, certs, error_msg).
    error_msg is empty string on success.
    """
    try:
        from ai_modules.ocr_extractor import extract_text_flagged
        pdf_bytes    = path.read_bytes()
        raw_text, _  = extract_text_flagged(pdf_bytes, "application/pdf")

        if not raw_text or len(raw_text.strip()) < 30:
            return [], [], f"OCR returned < 30 chars"

        extracted = nlp.process(raw_text)
        return extracted["skills"], extracted["certifications"], ""

    except Exception as e:
        return [], [], str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labels.json from real PDF resumes"
    )
    parser.add_argument("--resumes", required=True,
                        help="Folder containing PDF resume files")
    parser.add_argument("--mode",    choices=["auto", "template"], default="auto",
                        help="'auto' runs NLP (needs Firebase); 'template' creates empty entries")
    parser.add_argument("--output",  default="labels.json",
                        help="Output file name (default: labels.json)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Only process first N resumes (0 = all)")
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

    print(f"\n{BOLD}generate_labels.py{RESET}")
    print(f"  Found {len(files)} PDFs in '{resume_dir}'")
    print(f"  Mode: {args.mode}  →  output: {args.output}")

    # ── Template mode — no Firebase needed ────────────────────────────────────
    if args.mode == "template":
        labels = {
            f.name: {
                "expected_skills":    [],
                "expected_job_title": "",
                "min_match_score":    50,
            }
            for f in files
        }
        Path(args.output).write_text(json.dumps(labels, indent=2))
        print(f"\n  {GREEN}✓{RESET} Created {len(labels)} empty label entries → {args.output}")
        print(f"  Open {args.output} and fill in expected_skills and expected_job_title.")
        return

    # ── Auto mode — needs Firebase ─────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))

    if not _warmup_services():
        print(f"\n{RED}Cannot proceed without Firebase. Use --mode template to create empty labels.{RESET}")
        sys.exit(1)

    # NLPProcessor is now warmed up — create one instance (reused for all resumes)
    try:
        from ai_modules.nlp_processor import NLPProcessor
        nlp = NLPProcessor()
    except Exception as e:
        print(f"{RED}Failed to create NLPProcessor: {e}{RESET}")
        sys.exit(1)

    print(f"{BOLD}Processing {len(files)} PDFs...{RESET}\n")

    labels:  dict = {}
    ok_count = 0
    err_count = 0
    zero_count = 0
    t_start = time.perf_counter()

    for i, path in enumerate(files, 1):
        t0 = time.perf_counter()
        skills, certs, err = _process_resume(path, nlp)
        elapsed = (time.perf_counter() - t0) * 1000

        labels[path.name] = {
            "expected_skills":    skills,
            "expected_job_title": "",
            "min_match_score":    50,
        }
        if certs:
            labels[path.name]["expected_certs"] = certs

        if err:
            err_count += 1
            icon   = f"{RED}✗{RESET}"
            detail = f"{RED}{err}{RESET}"
        elif len(skills) == 0:
            zero_count += 1
            icon   = f"{YELLOW}~{RESET}"
            detail = f"{YELLOW}0 skills — may be scanned PDF or unusual format{RESET}"
        else:
            ok_count += 1
            icon   = f"{GREEN}✓{RESET}"
            detail = f"{len(skills)} skills  {', '.join(skills[:5])}{'…' if len(skills) > 5 else ''}"

        print(f"  [{i:>3}/{len(files)}] {icon}  {path.name:<45}  {detail}  ({elapsed:.0f}ms)")

        # Save incrementally every 20 resumes so you don't lose progress
        if i % 20 == 0:
            Path(args.output).write_text(json.dumps(labels, indent=2))
            elapsed_total = time.perf_counter() - t_start
            eta = (elapsed_total / i) * (len(files) - i)
            print(f"\n  {CYAN}[saved {i}/{len(files)} — ETA ~{eta:.0f}s]{RESET}\n")

    # Final save
    Path(args.output).write_text(json.dumps(labels, indent=2))
    total_s = time.perf_counter() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  {BOLD}Results{RESET}")
    print(f"  {GREEN}✓ {ok_count}{RESET}  resumes with skills extracted")
    print(f"  {YELLOW}~ {zero_count}{RESET}  resumes returned 0 skills (scanned/unusual format)")
    print(f"  {RED}✗ {err_count}{RESET}  resumes failed (OCR error)")
    print(f"  Total time: {total_s:.1f}s  ({total_s/len(files)*1000:.0f}ms avg/resume)")
    print(f"{'─' * 62}")
    print(f"\n  {GREEN}Saved → {args.output}{RESET}")

    print(f"""
{BOLD}Next steps:{RESET}
  1. Open {args.output} in any text editor
  2. Spot-check the 'expected_skills' lists — fix obvious errors
  3. Fill in 'expected_job_title' for resumes where you know the role
     e.g. "expected_job_title": "Software Engineer"
  4. Run the full evaluator:
       python eval_resumes.py --resumes {args.resumes} --labels {args.output} --verbose
""")


if __name__ == "__main__":
    main()