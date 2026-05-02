import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import fitz
import easyocr
import torch
import re
import numpy as np
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# ── Hardware ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── SBERT ─────────────────────────────────────────────────────────────────────
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# ── EasyOCR ───────────────────────────────────────────────────────────────────
reader = easyocr.Reader(["en"], gpu=(device == "cuda"))

# ── Poppler ───────────────────────────────────────────────────────────────────
POPPLER_PATH = None


def extract_text(file_path: str):
    """Stage 1: OCR Extraction with Quality Metric."""
    text = ""
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        text = " ".join([page.get_text("text") for page in doc])
        doc.close()

    if len(text.strip()) < 50:
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        text = " ".join(
            [" ".join(reader.readtext(np.array(img), detail=0)) for img in images]
        )

    clean_text = re.sub(r"\s+", " ", text)
    alnum_ratio = (
        len([c for c in clean_text if c.isalnum()]) / len(clean_text)
        if clean_text else 0
    )
    return clean_text, round(alnum_ratio * 100, 2)


# ── Known resume section headers ─────────────────────────────────────────────
SECTION_END_HEADERS = [
    'certifications', 'certificates', 'education', 'work experience',
    'professional experience', 'experience', 'employment', 'projects',
    'awards', 'references', 'publications', 'languages', 'summary',
    'objective', 'profile', 'about',
]


def extract_skills_section(text: str) -> str:
    """
    Option A: Find the Skills section boundary, then return ONLY the
    isolated listed token lines — stripping all prose entirely.

    OCR collapses PDF line-breaks into spaces, so a resume skills column:
        Zendesk
        AnyDesk
        SolarWinds Network Performance Monitor
        Malwarebytes
    becomes one flat string:
        "Zendesk AnyDesk SolarWinds Network Performance Monitor Malwarebytes"

    The listed tokens are bare, capitalized, and NOT preceded by a verb
    connector. Prose sentences inside the same section block always start
    with a verb or have a verb within the first two words:
        "Used Ansible for...", "Conducted data analysis using..."

    Steps:
      1. Fuzzy-detect the SKILLS header and the next section header to get
         the raw section text (same boundary logic as before).
      2. Walk the section word-by-word and collect only tokens that:
           (a) start with a capital letter
           (b) are NOT immediately preceded (<=2 words back) by a verb /
               preposition connector
           (c) are not pure noise (dates, numbers, single chars, company
               suffixes, etc.)
      3. Greedily grow each qualifying token into a multi-word phrase
         (up to 6 words) as long as subsequent words also look like listed
         items, stopping at a verb connector or noise boundary.
      4. Return the collected phrases joined by spaces — this is the clean
         token-only string passed to extract_skills_from_corpus().

    Falls back to the full raw section text if no tokens are isolated,
    and to the full document text if no skills header is found.
    """
    from difflib import SequenceMatcher

    def _r(a: str, b: str) -> int:
        return round(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)

    SKILLS_HEADERS = [
        'technical skills', 'key skills', 'core skills',
        'skills summary', 'skills and expertise', 'skills',
    ]

    # ── Step 1: Locate the raw skills section via fuzzy header detection ──────
    words        = text.split()
    n            = len(words)
    skills_start = None
    skills_end   = None

    for i in range(n):
        for ws in (2, 1):
            if i + ws > n:
                continue
            window = ' '.join(words[i:i + ws])
            for header in SKILLS_HEADERS:
                if len(header.split()) != ws:
                    continue
                if _r(window, header) >= 85:
                    skills_start = i + ws
                    break
            if skills_start is not None:
                break
        if skills_start is not None:
            break

    if skills_start is None:
        return text  # fallback: no skills header found

    for i in range(skills_start, n):
        for ws in (3, 2, 1):
            if i + ws > n:
                continue
            window = ' '.join(words[i:i + ws])
            for header in SECTION_END_HEADERS:
                if len(header.split()) != ws:
                    continue
                if _r(window, header) >= 88:
                    skills_end = i
                    break
            if skills_end is not None:
                break
        if skills_end is not None:
            break

    section_words = words[skills_start:skills_end] if skills_end else words[skills_start:]
    if not section_words:
        return text  # fallback: empty section

    # ── Step 2 & 3: Extract only isolated listed tokens from the section ──────
    # Verb/preposition connectors that signal prose context.
    # A capitalized word preceded by any of these within 2 positions is a
    # prose mention of a skill, not a listed item.
    VERB_CONNECTORS = {
        'using', 'used', 'use', 'leveraged', 'leverage', 'implemented',
        'implement', 'with', 'via', 'through', 'by', 'for', 'from',
        'deployed', 'deploy', 'conducted', 'conduct', 'provided', 'provide',
        'assisted', 'assist', 'collaborated', 'monitored', 'monitor',
        'worked', 'work', 'offered', 'offer', 'achieved', 'achieve',
        'ensuring', 'ensure', 'reducing', 'reduce', 'resulting', 'result',
        'including', 'include', 'maintaining', 'maintain',
    }

    # Tokens that are noise even when capitalized (company suffixes, stop
    # words, location abbreviations, social links, etc.)
    NOISE_TOKENS = {
        'inc', 'llc', 'ltd', 'corp', 'ma', 'ca', 'ny', 'tx', 'current',
        'present', 'linkedin', 'github', 'the', 'and', 'or', 'of', 'in',
        'to', 'a', 'an', 'is', 'are', 'was', 'were',
    }

    sw = section_words
    m  = len(sw)

    def _clean(w: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', w)

    def _is_noise(w: str) -> bool:
        c = _clean(w).lower()
        if not c or len(c) < 2:
            return True
        if c in NOISE_TOKENS:
            return True
        if re.fullmatch(r'\d+', c):
            return True
        return False

    def _starts_capital(w: str) -> bool:
        c = _clean(w)
        return bool(c) and c[0].isupper()

    def _preceded_by_verb(pos: int) -> bool:
        for lookback in range(1, 3):
            idx = pos - lookback
            if idx < 0:
                break
            if _clean(sw[idx]).lower() in VERB_CONNECTORS:
                return True
        return False

    candidates: list[str] = []
    visited:    set[int]  = set()

    for i in range(m):
        if i in visited:
            continue
        w = sw[i]
        if _is_noise(w):
            continue
        if not _starts_capital(w):
            continue
        if _preceded_by_verb(i):
            continue

        # Greedily grow into a multi-word listed phrase (max 6 words).
        # Stop at a verb connector, noise, or a lowercase prose word (4+ chars).
        MAX_PHRASE = 6
        phrase_words = [w]
        visited.add(i)

        j = i + 1
        while j < m and len(phrase_words) < MAX_PHRASE:
            nw  = sw[j]
            nwc = _clean(nw).lower()

            if nwc in VERB_CONNECTORS:
                break
            if _is_noise(nw):
                # Allow short joiners ("and", "of", "&") only when the very
                # next word is also capitalized, e.g. "Security and Compliance"
                if nwc in ('and', 'of', '&') and j + 1 < m and _starts_capital(sw[j + 1]):
                    phrase_words.append(nw)
                    visited.add(j)
                    j += 1
                    continue
                break
            if not _starts_capital(nw) and len(nwc) >= 4:
                break

            phrase_words.append(nw)
            visited.add(j)
            j += 1

        phrase       = ' '.join(phrase_words).strip()
        phrase_clean = _clean(phrase.replace(' ', ''))
        if len(phrase_clean) >= 3:
            candidates.append(phrase)

    # ── Step 4: Return clean token string or fallback to raw section ──────────
    if candidates:
        return ' '.join(candidates)

    return ' '.join(section_words)  # fallback: raw section if nothing isolated


def extract_certifications_section(text: str) -> str:
    """
    Find the Certifications section of a resume using fuzzy header detection.
    Returns only that section text, or empty string if none found.
    Falls back to empty string (unlike skills, certs may not exist).
    """
    from difflib import SequenceMatcher

    def _r(a: str, b: str) -> int:
        return round(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)

    CERT_HEADERS = [
        'certifications and awards', 'certifications & awards',
        'certifications and licenses', 'licenses and certifications',
        'professional certifications', 'certifications', 'certificates',
        'courses and certifications', 'training and certifications',
        'awards and certifications',
    ]

    CERT_END_HEADERS = [
        'education', 'work experience', 'professional experience',
        'experience', 'employment', 'projects', 'awards', 'references',
        'publications', 'languages', 'summary', 'objective', 'profile',
        'skills', 'technical skills', 'key skills', 'about',
    ]

    words       = text.split()
    n           = len(words)
    cert_start  = None
    cert_end    = None

    for i in range(n):
        for ws in (4, 3, 2, 1):
            if i + ws > n:
                continue
            window = ' '.join(words[i:i + ws])
            for header in CERT_HEADERS:
                if len(header.split()) != ws:
                    continue
                if _r(window, header) >= 85:
                    cert_start = i + ws
                    break
            if cert_start is not None:
                break
        if cert_start is not None:
            break

    if cert_start is None:
        return ''

    for i in range(cert_start, n):
        for ws in (3, 2, 1):
            if i + ws > n:
                continue
            window = ' '.join(words[i:i + ws])
            for header in CERT_END_HEADERS:
                if len(header.split()) != ws:
                    continue
                if _r(window, header) >= 88:
                    cert_end = i
                    break
            if cert_end is not None:
                break
        if cert_end is not None:
            break

    cert_text = ' '.join(
        words[cert_start:cert_end] if cert_end else words[cert_start:]
    ).strip()

    return cert_text


def extract_certificates_from_section(cert_text: str, known_certs: list) -> list:
    """
    Match known certificate names against the certifications section text.

    Pass 1 — exact/normalised substring match
    Pass 2 — OCR fuzzy using sliding window with three gates:
              (a) first-word anchor >= 80
              (b) second-word anchor >= 80 (if cert has 2+ words)
              (c) full phrase >= 82
    Pass 3 — partial prefix match: match just the first 3 words of a cert
              against windows. Handles cases like:
              "Microsoft Certified Azure Fundamentals" where the last word
              is replaced by a year "2013" in the OCR output, so the full
              4-word window never scores high enough.
    """
    from difflib import SequenceMatcher

    if not cert_text or not known_certs:
        return []

    def _r(a: str, b: str) -> int:
        return round(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)

    def normalize(s: str) -> str:
        return re.sub(r"[\s.\-_/]", "", s).lower()

    cert_lower      = cert_text.lower()
    cert_normalized = normalize(cert_text)
    cert_words      = cert_text.split()

    known_certs_sorted = sorted(known_certs, key=len, reverse=True)

    exact_matches = set()
    remaining     = []

    # Pass 1: exact / normalised substring
    for cert in known_certs_sorted:
        cert_l    = cert.lower()
        cert_norm = normalize(cert)
        if len(cert_norm) <= 1:
            continue
        if cert_l in cert_lower or cert_norm in cert_normalized:
            exact_matches.add(cert)
        else:
            remaining.append(cert)

    # Pass 2: full OCR fuzzy (first + second word anchors + full phrase)
    ocr_matches = set()
    for cert in remaining:
        cert_ws = cert.split()
        cert_wc = len(cert_ws)
        cert_fw = cert_ws[0]
        cert_sw = cert_ws[1] if cert_wc >= 2 else None

        if cert_wc == 1 and len(cert_ws[0]) < 11:
            continue

        for start in range(len(cert_words) - cert_wc + 1):
            ww     = cert_words[start:start + cert_wc]
            window = ' '.join(ww)
            if _r(cert_fw, ww[0]) < 80:
                continue
            if cert_sw and len(ww) >= 2:
                if _r(cert_sw, ww[1]) < 80:
                    continue
            if _r(cert, window) >= 82:
                ocr_matches.add(cert)
                break

    # Pass 3: prefix match for multi-word certs
    already = exact_matches | ocr_matches
    prefix_matches = set()
    for cert in remaining:
        if cert in already:
            continue
        cert_ws  = cert.split()
        cert_wc  = len(cert_ws)
        if cert_wc < 2:
            continue
        prefix_len = min(3, cert_wc - 1)
        prefix     = cert_ws[:prefix_len]
        prefix_str = ' '.join(prefix)

        for start in range(len(cert_words) - prefix_len + 1):
            ww     = cert_words[start:start + prefix_len]
            window = ' '.join(ww)
            if _r(prefix[0], ww[0]) < 80:
                continue
            if prefix_len >= 2 and _r(prefix[1], ww[1]) < 80:
                continue
            if _r(prefix_str, window) >= 85:
                prefix_matches.add(cert)
                break

    all_certs = exact_matches | ocr_matches | prefix_matches

    # Remove fragment if longer form also matched
    to_remove = set()
    certs_list = list(all_certs)
    for ca in certs_list:
        for cb in certs_list:
            if ca != cb and ca.lower() in cb.lower():
                to_remove.add(ca)
    all_certs -= to_remove

    return sorted(all_certs)


def extract_skills_from_corpus(
        resume_text: str,
        jobs_corpus: list,
        extra_skills: list | None = None,
) -> list:
    """
    Stage 2: Skill Extraction matched against the live Firestore job corpus
    PLUS any extra_skills (e.g. the full pool from labels.json).

    extract_skills_section() (Option A) already returns only the isolated
    listed tokens — all prose has been stripped before we get here.
    So both passes below operate on a clean token-only string.

    Pass 1 — exact / normalised substring match (case-insensitive).
             Skills are sorted longest-first so multi-word phrases like
             "Microsoft Azure" or "SolarWinds Network Performance Monitor"
             are matched before shorter fragments like "Azure".
    Pass 2 — OCR-fuzzy sliding window for skills not caught by Pass 1.
             Handles garbled OCR tokens like "Malwarcbytcs" → "Malwarebytes",
             "Microson Azuie" → "Microsoft Azure", etc.
             Uses first-word + second-word anchors to avoid false positives.

    Post-processing:
    - If a shorter skill is a substring of a longer matched skill, keep only
      the longer one (e.g. "Azure" is dropped when "Microsoft Azure" matched).
    - Case-variant deduplication keeps the canonical corpus form.
    """
    # ── Collect all unique skills: Firestore corpus + labels pool ─────────────
    corpus_skills: set[str] = {
        skill.strip()
        for job in jobs_corpus
        for skill in job.get("required_skills", [])
    }
    if extra_skills:
        for s in extra_skills:
            corpus_skills.add(s.strip())

    all_known_skills: list[str] = list(corpus_skills)

    if not all_known_skills:
        return []

    # Sort longest-first so multi-word phrases are matched before fragments
    all_known_skills.sort(key=len, reverse=True)

    # ── Option A: extract_skills_section returns clean listed tokens only ─────
    skills_section    = extract_skills_section(resume_text)
    resume_lower      = skills_section.lower()

    def normalize(s: str) -> str:
        return re.sub(r'[\s.\-_/]', '', s).lower()

    resume_normalized = normalize(skills_section)

    # ── Pass 1: exact / normalised substring match ────────────────────────────
    exact_matches:    set[str] = set()
    remaining_skills: list[str] = []

    for skill in all_known_skills:
        skill_lower = skill.lower()
        skill_norm  = normalize(skill)

        if len(skill_norm) <= 1:
            continue

        if len(skill_norm) <= 3:
            pattern = rf'\b{re.escape(skill_lower)}\b'
            if re.search(pattern, resume_lower):
                exact_matches.add(skill)
            else:
                remaining_skills.append(skill)
            continue

        if skill_lower in resume_lower or skill_norm in resume_normalized:
            exact_matches.add(skill)
        else:
            remaining_skills.append(skill)

    # ── Pass 2: OCR-fuzzy sliding window ─────────────────────────────────────
    # Operates on the already-clean token string from extract_skills_section(),
    # so the window only ever sees listed items, never prose verbs.
    ocr_fuzzy_matches: set[str] = set()
    resume_words = skills_section.split()

    def _word_ratio(a: str, b: str) -> int:
        from difflib import SequenceMatcher
        return round(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)

    for skill in remaining_skills:
        skill_words      = skill.split()
        skill_word_count = len(skill_words)

        # Single-word skills need >= 6 chars to enter fuzzy matching
        if skill_word_count == 1 and len(skill_words[0]) < 6:
            continue

        skill_first_word  = skill_words[0]
        skill_second_word = skill_words[1] if skill_word_count >= 2 else None

        for start in range(len(resume_words) - skill_word_count + 1):
            window_words  = resume_words[start:start + skill_word_count]
            window        = " ".join(window_words)
            window_first  = window_words[0]

            if _word_ratio(skill_first_word, window_first) < 80:
                continue

            if skill_second_word and skill_word_count >= 2:
                window_second = window_words[1] if len(window_words) >= 2 else ""
                if _word_ratio(skill_second_word, window_second) < 80:
                    continue

            if _word_ratio(skill, window) >= 82:
                ocr_fuzzy_matches.add(skill)
                break

    all_matches = exact_matches | ocr_fuzzy_matches

    # ── Post-process: prefer longer match over fragment ────────────────────────
    to_remove: set[str] = set()
    matches_list = list(all_matches)
    for skill_a in matches_list:
        for skill_b in matches_list:
            if skill_a is skill_b:
                continue
            if skill_a.lower() in skill_b.lower() and skill_a.lower() != skill_b.lower():
                to_remove.add(skill_a)

    all_matches -= to_remove

    # ── Post-process: deduplicate case variants, keep corpus form ─────────────
    seen_normalized: dict[str, str] = {}
    for skill in sorted(all_matches):
        key = normalize(skill)
        if key not in seen_normalized:
            seen_normalized[key] = skill

    return sorted(seen_normalized.values())


def canonicalize_skills_to_corpus(
        extracted_skills: list,
        jobs_corpus: list,
        extra_skills: list | None = None,
        semantic_threshold: float = 0.88,
        fuzzy_threshold: int = 86,
) -> list:
    """
    Stage 2b: NLP Canonicalization Pass.

    Maps each raw extracted skill to its canonical Firestore/corpus form.
    Crucially, this function ONLY renames skills — it never introduces skills
    that were not already found in the resume. If no confident mapping is found,
    the raw extracted skill is kept as-is and dropped from the final output
    (since it has no matching corpus entry and should not appear in scoring).

    Two hard guards prevent hallucination:

      Guard 1 — Length-ratio gate (anti-expansion):
        A short token like "Mass" must not be mapped to a long corpus phrase
        like "Mass and Energy Balance". The candidate is rejected if:
            len(raw_token) / len(corpus_candidate) < LENGTH_RATIO_MIN
        This blocks partial_ratio false positives where a common word in the
        raw skill scores 100 as a substring of a long corpus phrase.

      Guard 2 — Bidirectional fuzzy gate (Tier 1):
        Both partial_ratio AND token_set_ratio must exceed the threshold.
        partial_ratio alone is too permissive for short tokens.

    Tier 1 — RapidFuzz (bidirectional, length-gated):
        Handles genuine OCR garbles, e.g.:
          "Malwarcbytcs"  → "Malwarebytes"       (both ratios high, lengths similar)
          "SolarWinds"    → "SolarWinds Network Performance Monitor"
                            (partial=100 but length-ratio passes since "SolarWinds"
                             is the distinctive first token)

    Tier 2 — SBERT cosine similarity (length-gated, high threshold):
        Handles brand+product expansions where fuzzy character overlap is low:
          "Azure"         → "Microsoft Azure"
          "Excel"         → "Microsoft Excel"
        Only fires when the raw skill is NOT a common English word (min 5 chars
        or is multi-word), preventing noise tokens from getting semantic matches.

    If neither tier produces a confident, length-safe mapping the skill is
    dropped — it was an OCR noise token or generic word with no corpus entry.

    Args:
        extracted_skills:   Raw skills list from extract_skills_from_corpus().
        jobs_corpus:        Firestore job documents (dicts with required_skills).
        extra_skills:       Additional skill pool (e.g. from labels.json).
        semantic_threshold: Min SBERT cosine score (raised to 0.88 vs old 0.82).
        fuzzy_threshold:    Min RapidFuzz score (raised to 86 vs old 80).

    Returns:
        Sorted, deduplicated list of canonical skill names from the corpus.
        Skills with no confident corpus mapping are excluded.
    """
    # ── Build the full corpus skill pool ──────────────────────────────────────
    corpus_skills: set[str] = {
        skill.strip()
        for job in jobs_corpus
        for skill in job.get("required_skills", [])
    }
    if extra_skills:
        corpus_skills.update(s.strip() for s in extra_skills)

    # Longest-first so deduplication later picks the fuller canonical form
    corpus_list: list[str] = sorted(corpus_skills, key=len, reverse=True)

    if not corpus_list or not extracted_skills:
        return extracted_skills

    # ── Precompute corpus embeddings once (batch for speed) ───────────────────
    corpus_embeddings = model.encode(corpus_list, convert_to_tensor=True)

    # Minimum length ratio: raw skill length vs corpus candidate length.
    # Prevents "Mass" (4 chars) from mapping to "Mass and Energy Balance" (26 chars).
    # 4/26 = 0.15, well below the 0.45 floor.
    # "SolarWinds" (10) vs "SolarWinds Network Performance Monitor" (38): 10/38 = 0.26
    # — this would be blocked too, so for known first-token anchors we use a
    # separate anchor check in Tier 1 rather than relying on length ratio alone.
    LENGTH_RATIO_MIN = 0.45

    def _length_ok(raw: str, candidate: str) -> bool:
        """
        Accept the mapping if:
          (a) length ratio is above the floor, OR
          (b) the raw skill is a clean prefix/anchor of the candidate
              (first word of candidate starts with the raw skill, case-insensitive).
        Rule (b) handles "SolarWinds" → "SolarWinds Network Performance Monitor"
        and "Malwarebytes" OCR garbles without needing a high length ratio.
        """
        ratio = len(raw) / max(len(candidate), 1)
        if ratio >= LENGTH_RATIO_MIN:
            return True
        # Anchor check: raw is a substantial first-token match
        candidate_first = candidate.split()[0].lower()
        raw_clean = raw.lower().strip()
        if len(raw_clean) >= 5 and candidate_first.startswith(raw_clean[:5]):
            return True
        # Allow if raw skill (multi-word) shares ALL its words with candidate
        raw_words = set(raw_clean.split())
        cand_words = set(candidate.lower().split())
        if len(raw_words) > 1 and raw_words.issubset(cand_words):
            return True
        return False

    canonical: set[str] = set()

    for raw_skill in extracted_skills:
        raw_lower = raw_skill.lower().strip()
        raw_words = raw_lower.split()
        best_match: str | None = None

        # ── Tier 1: Bidirectional fuzzy match (length-gated) ─────────────────
        # Uses BOTH partial_ratio (substring-aware) AND token_set_ratio
        # (word-order-insensitive). Both must exceed the threshold.
        # This double-gate kills cases like:
        #   "Mass" → partial_ratio=100 vs "Mass and Energy Balance" (substring hit)
        #   but token_set_ratio("mass", "mass and energy balance") ≈ 44 → rejected.
        best_fuzz_score = 0
        best_fuzz_skill: str | None = None

        for cs in corpus_list:
            cs_lower = cs.lower()

            # Fast pre-filter: skip corpus entries that share no word-start chars
            # with the raw skill to avoid scoring every entry for every skill.
            if raw_words and not any(
                    cs_lower.startswith(w[:3]) or w[:3] in cs_lower
                    for w in raw_words if len(w) >= 3
            ):
                continue

            p_score  = fuzz.partial_ratio(raw_lower, cs_lower)
            ts_score = fuzz.token_set_ratio(raw_lower, cs_lower)
            combined = min(p_score, ts_score)   # both must be high

            if combined > best_fuzz_score and _length_ok(raw_skill, cs):
                best_fuzz_score = combined
                best_fuzz_skill = cs

        if best_fuzz_score >= fuzzy_threshold and best_fuzz_skill:
            best_match = best_fuzz_skill

        # ── Tier 2: Semantic match via SBERT (length-gated, high threshold) ───
        # Only runs when Tier 1 found nothing above threshold.
        # Guards:
        #   - raw skill must be >= 5 chars or multi-word (blocks noise tokens)
        #   - cosine threshold raised to 0.88 (was 0.82) for tighter control
        #   - length ratio gate applied to semantic candidate too
        if best_match is None:
            raw_char_len = len(raw_skill.replace(" ", ""))
            is_multiword = len(raw_words) > 1

            if raw_char_len >= 5 or is_multiword:
                raw_vec    = model.encode(raw_skill, convert_to_tensor=True)
                cos_scores = util.cos_sim(raw_vec, corpus_embeddings)[0]

                # Check top-3 candidates in case the best by score fails length gate
                top_indices = cos_scores.argsort(descending=True)[:3].tolist()
                for idx in top_indices:
                    top_score     = float(cos_scores[idx])
                    top_candidate = corpus_list[idx]
                    if top_score >= semantic_threshold and _length_ok(raw_skill, top_candidate):
                        best_match = top_candidate
                        break

        # ── Decision ─────────────────────────────────────────────────────────
        # Only add to canonical set if a confident corpus mapping was found.
        # Skills with no mapping are DROPPED — they are OCR noise or generic
        # words that don't belong in the scored skill profile.
        if best_match is not None:
            canonical.add(best_match)

    # ── Remove fragment skills subsumed by a longer canonical match ───────────
    # e.g. once "Microsoft Azure" is in the set, bare "Azure" is redundant.
    result      = list(canonical)
    to_drop: set[str] = set()
    for skill_a in result:
        for skill_b in result:
            if skill_a != skill_b and skill_a.lower() in skill_b.lower():
                to_drop.add(skill_a)

    return sorted(canonical - to_drop)


def fuzzy_skill_verify(discovered_list: list, ground_truth_list: list, threshold: int = 80):
    """Stage 2 validation against expected skills."""
    matches = []
    for truth in ground_truth_list:
        for discovered in discovered_list:
            if fuzz.partial_ratio(truth.lower(), discovered.lower()) >= threshold:
                matches.append(truth)
                break
    return matches


def calculate_cosine_similarity(resume_text: str, resume_skills: list, jobs_corpus: list):
    """
    Stage 3: Hybrid Scoring — uses pre-computed SBERT embeddings from Firestore
    when available, falls back to live encoding for manually inserted jobs.
    """
    resume_profile = f"Skills: {' '.join(resume_skills)}. Resume: {resume_text}"
    resume_vec = model.encode(resume_profile, convert_to_tensor=True)

    job_vecs_list      = []
    jobs_to_encode_idx = []

    for i, job in enumerate(jobs_corpus):
        if "embedding" in job and job["embedding"]:
            vec = torch.tensor(job["embedding"], dtype=torch.float32).to(device)
            job_vecs_list.append(vec)
        else:
            job_vecs_list.append(None)
            jobs_to_encode_idx.append(i)

    if jobs_to_encode_idx:
        profiles = [
            f"Job: {jobs_corpus[i]['title']}. "
            f"Core Skills: {' '.join(jobs_corpus[i]['required_skills'] * 3)}. "
            f"Desc: {jobs_corpus[i]['description']}"
            for i in jobs_to_encode_idx
        ]
        live_vecs = model.encode(profiles, convert_to_tensor=True)
        for idx, vec in zip(jobs_to_encode_idx, live_vecs):
            job_vecs_list[idx] = vec

    job_vecs      = torch.stack(job_vecs_list).to(device)
    cosine_scores = util.cos_sim(resume_vec, job_vecs)[0]

    def _skill_matched(job_skill: str, resume_skill_set: set, resume_skills_raw: list) -> bool:
        """
        Returns True if a job_skill is covered by the resume.
        Three-tier check:
        1. Exact lowercase match  ("ansible" == "ansible")
        2. Substring check        ("azure" in "microsoft azure" or vice-versa)
        3. Fuzzy token match      (partial_ratio >= 88, catches OCR/spacing variants)
        """
        jsl = job_skill.lower().strip()

        if jsl in resume_skill_set:
            return True

        for rs in resume_skills_raw:
            rsl = rs.lower().strip()
            if jsl in rsl or rsl in jsl:
                return True

        if len(jsl) >= 4:
            for rs in resume_skills_raw:
                if fuzz.partial_ratio(jsl, rs.lower().strip()) >= 88:
                    return True

        return False

    results = []
    for i, score in enumerate(cosine_scores):
        raw_semantic_score = float(score)
        job_skills = jobs_corpus[i]["required_skills"]

        resume_set = {s.lower() for s in resume_skills}
        matched    = [s for s in job_skills if _skill_matched(s, resume_set, resume_skills)]
        missing    = [s for s in job_skills if not _skill_matched(s, resume_set, resume_skills)]
        gap_count  = len(missing)
        match_ratio = len(matched) / len(job_skills) if job_skills else 0
        final_score = (raw_semantic_score * 0.6) + (match_ratio * 0.4)

        if gap_count == 0:
            final_score = min(0.985, final_score * 1.25)

        results.append({
            "job_id":          jobs_corpus[i].get("id"),
            "title":           jobs_corpus[i]["title"],
            "industry":        jobs_corpus[i].get("industry", "Computing"),
            "score":           f"{round(final_score * 100, 2)}%",
            "gap_count":       gap_count,
            "required_skills": job_skills,
            "matched_skills":  matched,
            "missing_skills":  missing,
        })

    return sorted(results, key=lambda x: float(x["score"].replace("%", "")), reverse=True)[:5]


def identify_skill_gap(resume_skills: list, job_skills: list):
    """Stage 4: Gap Analysis."""
    resume_set = {s.strip().lower() for s in resume_skills}
    return [s for s in job_skills if s.lower() not in resume_set]