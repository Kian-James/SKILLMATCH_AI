"""
debug_list_skills.py
=====================
Standalone diagnostic — dumps every unique `required_skills` entry across
the whole Firestore jobs corpus (the "allowlist" that Stage 2 SBERT-matches
resume-extracted candidates against), and optionally scores specific
candidate strings against it so you can see exactly why a term did or
didn't clear the 0.80 cosine threshold.

Uses the project's own db/ and core/ modules, so it reflects production
behaviour exactly (same corpus fetch, same SBERT model, same threshold).

USAGE
-----
  # 1. Just dump the full allowlist:
  python debug_list_skills.py

  # 2. Also score specific candidate strings against it (SBERT + fuzzy):
  python debug_list_skills.py "MySQL" "Node js" "Angular" "Git" "Github" "VS Code" "Figma"

Requires the same environment as the rest of the backend: Firebase
credentials (FIREBASE_CREDENTIALS_JSON or FIREBASE_CREDENTIALS_PATH) and
whatever SBERT model core/ocr_correction.py normally loads.
"""

from __future__ import annotations

import sys

from db.jobs import fetch_jobs_corpus, derive_skills_from_corpus, derive_certs_from_corpus


def dump_allowlist() -> list[str]:
    print("Fetching jobs corpus from Firestore...")
    jobs_corpus = fetch_jobs_corpus()
    print(f"Fetched {len(jobs_corpus)} job(s).\n")

    skills = derive_skills_from_corpus(jobs_corpus)
    certs  = derive_certs_from_corpus(jobs_corpus)

    print(f"=== {len(skills)} unique required_skills across the corpus ===")
    for i, s in enumerate(skills):
        print(f"  [{i:>3}] {s}")

    print(f"\n=== {len(certs)} unique required_certifications across the corpus ===")
    for i, c in enumerate(certs):
        print(f"  [{i:>3}] {c}")

    # Per-job breakdown — useful to see which specific job postings do/don't
    # list a given skill, rather than just the flattened union.
    print("\n=== Per-job required_skills ===")
    for job in jobs_corpus:
        title = job.get("title") or job.get("id")
        req   = job.get("required_skills", [])
        print(f"  {title!r} ({job.get('id')}): {req}")

    return skills


def score_candidates(candidates: list[str], allowlist: list[str]) -> None:
    from rapidfuzz import fuzz

    print(f"\n=== Scoring {len(candidates)} candidate string(s) against the allowlist ===")

    # Fuzzy scores first — cheap, always available, and a decent proxy for
    # "is this even spelled anything like an allowlist entry".
    print("\n--- Fuzzy (rapidfuzz) top-3 matches per candidate ---")
    for cand in candidates:
        scored = []
        for item in allowlist:
            score = max(
                fuzz.ratio(cand.lower(), item.lower()),
                fuzz.token_sort_ratio(cand.lower(), item.lower()),
                fuzz.token_set_ratio(cand.lower(), item.lower()),
            )
            scored.append((score, item))
        scored.sort(reverse=True)
        top3 = ", ".join(f"{item!r} ({score})" for score, item in scored[:3])
        print(f"  {cand!r:25} -> {top3}")

    # Real SBERT cosine, matching production's exact threshold (0.80), if
    # the model is available in this environment.
    try:
        from core.ocr_correction import get_sbert
        import numpy as np

        print("\n--- SBERT cosine (production threshold = 0.80) ---")
        model = get_sbert()
        allow_emb = model.encode(allowlist, normalize_embeddings=True)
        cand_emb  = model.encode(candidates, normalize_embeddings=True)

        for i, cand in enumerate(candidates):
            sims = allow_emb @ cand_emb[i]
            order = np.argsort(-sims)[:3]
            top3 = ", ".join(f"{allowlist[j]!r} ({sims[j]:.3f})" for j in order)
            best = sims[order[0]]
            verdict = "MATCH" if best >= 0.80 else "no match"
            print(f"  {cand!r:25} -> {top3}   [{verdict}]")
    except Exception as e:
        print(f"\n(Skipping SBERT scoring — model unavailable here: {e})")


if __name__ == "__main__":
    allowlist = dump_allowlist()
    candidates = sys.argv[1:]
    if candidates:
        score_candidates(candidates, allowlist)
    else:
        print(
            "\nTip: pass candidate strings as CLI args to score them against "
            "the allowlist, e.g.:\n"
            '  python debug_list_skills.py "MySQL" "Node js" "Angular" "Git" "Github" "VS Code" "Figma"'
        )