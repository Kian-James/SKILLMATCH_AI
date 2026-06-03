"""
core/skills.py
==============
Stage 2 — Skill extraction.

Changes (v6 backport): added is_native_pdf to fix early-exit guard truncation.
Changes (v7): fixed split_ocr_blob greedy consumption bug.

           Problem: longest-first-hit algorithm committed to win=2 "Zendosk AnyDesk"
           matching a compound allowlist entry (e.g. "Zendesk AnyDesk") at score 93,
           consuming both tokens and silently dropping "AnyDesk" as its own skill.

           Fix: check win=1 first. If score >= _SINGLE_TOKEN_CONFIDENCE (85),
           commit immediately without trying longer windows. Only fall through to
           multi-token search when the single token is ambiguous (score < 85).

Changes (v8): fixed emoji/symbol prefix stripping in _SKILL_HEADING_RE and
           _SKILLS_END_RE so headers like "⭐ SKILLS" or "👤 WORK EXPERIENCE"
           are correctly detected. Previously the ^ anchor failed to match
           because non-word characters (emojis, bullets) preceded the keyword.

Changes (v9): fixed _SKILL_HEADING_RE false-positive on words like "Skilled",
           "Skillful", "Skilling", "Skillset" by adding a negative lookahead
           (?!ed|ing|ful|set) after skills? so only bare "Skill" / "Skills"
           headings are matched, not substrings inside prose words.

Changes (v10): rewrote _SKILL_HEADING_RE as a full-line match (keyword + EOL).
           The $ anchor means the heading keyword must be the ONLY meaningful
           content on the line (optional leading emoji/bullet, optional trailing
           colon). This blocks prose lines like "Expertise includes Ethereum..."
           or "Technologies used include React and Node" because they continue
           past the keyword. "expertise" and bare "technologies" are removed
           from the alternation — they appear too often in prose. Only compound
           forms ("Technical Expertise", "Tools & Technologies") are kept.

           Accepted headings (case-insensitive, optional leading symbol/colon):
             Skill / Skills
             Key/Core/Technical/Professional Skills
             Skills & Tools / Skills and Tools
             Skills & Technologies / Skills and Technologies
             Competencies / Competency
             Technical/Core/Key Competencies
             Technical Expertise / Areas of Technical Expertise
             Tools & Technologies / Tools and Technologies
             Technical Proficiencies / Proficiencies / Proficiency

Changes (v11): fixed over-consumption of skills section in PDFs where the
           candidate's name and job title (e.g. "QUENTIN RAMIREZ",
           "LINUX SYSTEMS ADMINISTRATOR") appear as all-caps banner lines
           between the skills list and the WORK EXPERIENCE heading.
           These lines don't match _SKILLS_END_RE (no known section keyword)
           so they were silently included in the extracted skills block.

           Fix: introduced _is_banner_line() helper. When a line is encountered
           after at least one skill has been captured, it is tested against
           three layered checks (structural pattern -> word-shape -> allowlist
           guard). Applied in both is_native_pdf and OCR branches.

Changes (v12): fixed _is_banner_line() false-positive on single-word skill
           acronyms (NLTK, AWS, GIT, DOCKER) and two-word skill phrases
           (Hugging Face, Bash Scripting, REST APIs, Server Administration).

           Problem (single-word): the v11 regex matched any all-caps string
           >= 4 chars with no digits. "NLTK" matched -> loop stopped after
           "Hugging Face", dropping NLTK, Transformers, REST APIs, Git.

           Problem (two-word skill phrases): "Hugging Face" and "Bash Scripting"
           are both Title Case two-word strings -- structurally identical to
           a person's name like "Camille Reyes".

           Fix (layer 1 -- structural): require at least one internal space
           (2+ words). Eliminates single-word acronyms.

           Fix (layer 2 -- word-shape): require every word to be either pure
           Title Case (Xxx) or pure ALL CAPS (XXX). Eliminates tokens like
           "APIs" (mixed), "PyTorch" (camelCase), "scikit-learn" (lowercase).

           Fix (layer 3 -- allowlist guard): if the line fuzzy-matches any
           known skill in the allowlist (score >= 80), treat it as a skill
           and do NOT stop. Catches two-word skill phrases like "Hugging Face"
           and "Server Administration" that pass the structural/shape checks.

Changes (v13): added _CONTACT_LINE_RE to stop kept_lines when a contact-info
           line appears inside the skills block. Affects PDFs where the
           candidate's email, phone, and location (with emoji icons) are
           rendered after the skills list instead of in a sidebar column.
           Without this guard the comma-branch fired on "Cebu City, Philippines"
           and split the entire remainder of the document on commas.
           Triggers: @ symbol, phone-like digit run (7+ chars), contact emoji
           codepoints, or www./http prefix. Applied before _is_banner_line in
           both the is_native_pdf and OCR branches. _EXPERIENCE_LINE_RE also
           promoted to module level (was redundantly re-compiled per call).

Changes (v14): fixed _SKILLS_END_RE false-positive on skill names that begin
           with a section keyword. "Objective-C" starts with "objective" which
           is an alternation in _SKILLS_END_RE. The regex matched at position 0
           of rest, producing an empty block and skills_section_rule_based = "".
           Fix: added negative lookahead (?![\w\-]) after "objective" so it
           only matches the bare heading word, not "Objective-C" or similar.

Changes (v15): removed "projects?" from _SKILLS_END_RE. It was too ambiguous --
           fires on skill-context phrases and is not a reliable hard section
           boundary. Work Experience / Education / Employment are sufficient;
           actual project content is caught by _EXPERIENCE_LINE_RE downstream.

Changes (v16): fixed excess/duplicate skills produced when the LLM splits a
           compound skill like "CAD Software (SolidWorks, AutoCAD)" into its
           parts ("CAD Software", "SolidWorks", "AutoCAD"). Added _is_subsumed()
           cleanup step before return to drop any skill whose lowercase form is
           a substring of a longer already-accepted canonical skill.

Changes (v17): fixed comma-split incorrectly fragmenting parenthetical skills
           like "Process Simulation Software (Aspen HYSYS, Aspen Plus)".
           The comma inside the parentheses was treated as a skill delimiter,
           producing broken fragments ("Process Simulation Software (Aspen",
           "Aspen Plus)") that the LLM and canonicalizer could not match.

           Fix: introduced _protect_parens() and _restore_parens() helpers.
           Before comma-splitting, all parenthesised substrings have their
           internal commas replaced with a unique placeholder. After splitting,
           the placeholder is restored so the full skill phrase is preserved.
           Applied only to the comma-branch of extract_skills_section since
           semicolon, pipe, and bullet branches are not affected by this issue.

Changes (v18): fixed _is_subsumed() incorrectly dropping distinct skills that
           share words with a longer skill via mere substring overlap.

           Problem: "C Programming" was dropped because "c programming" is a
           substring of "embedded c programming" — but these are two separate,
           valid skills. The v16 implementation used a naive `s in other.lower()`
           check which matched on any substring regardless of context.

           Fix: _is_subsumed() now only drops a skill when it appears inside a
           longer skill as a parenthetical fragment — i.e. the matched substring
           is immediately preceded by '(' and followed by ',' or ')'. This is
           the only case where a skill is a true split artifact (e.g. "SolidWorks"
           split out of "CAD Software (SolidWorks, AutoCAD)").

           Word-overlap between semantically distinct skills (e.g. "C Programming"
           and "Embedded C programming") no longer triggers a drop. Both skills
           are preserved in discovered_skills, matching skills_section_rule_based.

Changes (v19): fixed _is_subsumed() missing skills that appear after a comma
           inside parentheses (e.g. "AutoCAD" in "CAD Software (SolidWorks, AutoCAD)").

           Problem: "AutoCAD" was NOT dropped because its preceding character
           was ' ' (space after the comma separator), not '(', so the v18
           parenthetical-fragment check did not fire.

           Fix: extended the subsumed check to also match when the skill is
           preceded by any character (including space) AND followed by ',' or ')',
           provided that the text before the match contains more '(' than ')' —
           i.e. the match is genuinely inside an unclosed parenthetical group.
           This catches all comma-separated items inside parens regardless of
           their position (first, middle, or last).

           Examples:
             "SolidWorks" in "CAD Software (SolidWorks, AutoCAD)"  -> drop (after '(')
             "AutoCAD"    in "CAD Software (SolidWorks, AutoCAD)"  -> drop (after ', ', inside parens)
             "C Programming" in "Embedded C programming"           -> KEEP (no parens at all)

Changes (v20): fixed drop guard in discover_skills_dynamically silently dropping
           skills whose names contain punctuation meaningful to their identity
           (e.g. ".NET", "C#", "C++", "F#", "ASP.NET", "Node.js").

           Root cause: the guard relied on `normalize(canon) not in allow_norm`
           as a proxy for "not in allowlist". Because normalize() strips all
           punctuation, normalize(".NET") -> "net" and normalize("C#") -> "c",
           which either collide with unrelated keys or produce keys too short
           to match, causing the guard to fire and drop valid skills.

           Fix: replaced the brittle normalized-key lookup with a direct fuzzy
           score check against the full allow_norm key set. A skill is only
           dropped when canonicalize() returned it unchanged (i.e. found nothing
           better) AND no allowlist entry scores >= 75 against it. This is purely
           a change to the drop-guard heuristic; normalize() itself is untouched
           and the allowlist remains the Firestore source of truth.

Changes (v21): fixed deduplication collisions for punctuation-significant skills.

           Problem: seen set used normalize(canon) as the dedup key. normalize()
           strips all punctuation so "C#" -> "c", ".NET" -> "net", "C++" -> "c".
           "C#" and "C++" both produced the key "c" and the second one was
           silently dropped as a duplicate of the first.

           Fix: replaced normalize(canon) dedup key with canon.lower().strip()
           in both the LLM loop and the direct_scan_section loop. This preserves
           meaningful punctuation in the key ("c#", ".net", "c++") so each skill
           gets its own unique slot in seen. normalize() is still used elsewhere
           in the file where punctuation stripping is genuinely needed.

Changes (v22): added prose-sentence guard in the comma-branch of
           extract_skills_section. When a comma-delimited token contains more
           than 5 words it is treated as a prose sentence (e.g. "Managed
           procurement of IT hardware") rather than a skill, and extraction
           stops immediately at that point. This prevents experience bullet
           points that bleed past _SKILLS_END_RE from polluting the skills list
           and the downstream LLM prompt.

           Example (Riven Sarmiento resume):
             Before v22 — skills_section_rule_based included:
               "...Git, Managed procurement of IT hardware, software licenses,
                and infrastructure, equipment for enterprise clients., ..."
             After v22 — extraction stops at the first >5-word token:
               "Procurement Management, Vendor Coordination, Inventory
                Management, Microsoft Excel, Asset Tracking, Git"

Changes (v23): fixed direct_scan_section hits not being canonicalized before
           being added to discovered_list.

           Root cause: the LLM path calls canonicalize() on every item it
           returns, mapping raw/OCR forms like "Node js" to their allowlist
           canonical form "Node.js". The direct_scan_section path did NOT call
           canonicalize() — it appended hits as-is. So "Node js" found by the
           direct scan was added to discovered_list as "Node js" instead of
           "Node.js", causing a mismatch in Stage 3 NLP validation where the
           ground truth "Node.js" failed to match "Node js".

           Fix: canonicalize() is now called on every hit from
           direct_scan_section before dedup and append, identical to how the
           LLM path handles its output. No hardcoding, no normalize() changes —
           every discovered skill goes through the same canonicalization step
           regardless of which path found it.

Changes (v24): fixed OCR-split compound skill names being treated as separate
           single-word skills (e.g. "Healthcare" / "Technology" on separate
           lines instead of "Healthcare Technology").

           Root cause: some OCR layouts break compound skill names across
           consecutive lines. The multi-line branch of extract_skills_section
           (elif len(kept_lines) > 1) treats each line as a separate entry,
           so "Healthcare" and "Technology" become two distinct (invalid) skills.

           Fix: introduced _merge_split_lines() helper, called on kept_lines
           before the comma/bullet/line branching logic. For each single-word
           line, it checks whether joining it with the next line produces a
           string that scores >= 80 against the allowlist AND the single word
           alone scores < 80. If both conditions hold, the two lines are merged
           into one entry. This is conservative — it never merges two valid
           standalone skills that happen to be adjacent.

           Example:
             Before: kept_lines = ["Healthcare", "Technology", "Calibration"]
             After:  kept_lines = ["Healthcare Technology", "Calibration"]

Changes (v25): fixed OCR-split parenthetical skill names that span multiple
           lines (e.g. "Process Simulation Software (Aspen" / "HYSYS;" / "Plus)")
           being fragmented before _protect_parens could act on them.

           Root cause: OCR breaks the closing ')' onto a later line, so each
           line has mismatched parenthesis counts. The semicolon or comma
           branching logic then fires on the fragmented text rather than on
           the intact compound skill.

           Fix 1 — _rejoin_open_parens(): new helper called on kept_lines
           (after _merge_split_lines) that buffers lines until open parentheses
           are closed. Lines are concatenated with a space until the paren count
           balances, producing a single intact line like:
             "Process Simulation Software (Aspen HYSYS; Plus)"
           This lets _protect_parens (comma-branch) and the semicolon-branch
           work on the full, balanced string.

           Fix 2 — _effective_threshold(): new helper used inside
           discover_skills_dynamically that lowers the direct_scan_section
           threshold by 4 points (floor 75) when the candidate string contains
           parentheses. Compound parenthetical skill names naturally score lower
           on fuzzy ratio due to punctuation weight, so a blanket threshold of
           82 rejects valid matches like "Process Simulation Software
           (Aspen HYSYS, Aspen Plus)" at score 80.95. The reduction is scoped
           only to parenthetical candidates to avoid introducing noise for
           simple single-phrase skills.

Changes (v28): split compound skills joined by ' & ' or ' and ' into separate
           entries when both halves are independently recognised in the allowlist.

           Problem: skills like "HTML5 & CSS3" and "Jest & Enzyme" were emitted
           as single tokens. The allowlist has "HTML5", "CSS3", "Jest", "Enzyme"
           as separate entries so the compound token either matched only one or
           was dropped entirely, causing unmatched_skills in Stage 3.

           Fix: added _expand_ampersand() helper called on the final entries
           list just before the join. For each entry containing ' & ' or ' and ',
           it splits into two halves and checks:
             - If the full compound scores >= 80 against the allowlist (it IS
               a known compound skill), keep it as-is.
             - Otherwise if both halves individually score >= 75, replace the
               entry with the two separate skills.
           This is conservative — legitimate compound allowlist entries like
           "Figma & Adobe XD" that score >= 80 as a unit are never split.

           Before: entries = ["HTML5 & CSS3", "Jest & Enzyme", ...]
                   HTML5 & CSS3 unmatched in Stage 3
           After:  entries = ["HTML5", "CSS3", "Jest", "Enzyme", ...]
                   all four matched individually ✓

Changes (v27): fixed _merge_split_lines() missing two-word skills whose parts
           OCR emitted in reverse order on separate lines (e.g. "Tracking" /
           "Bug" instead of "Bug Tracking").

           Root cause: the forward join "Tracking Bug" scores low (~60) and
           the merge was rejected. The correct form "Bug Tracking" was never
           tried because _merge_split_lines only attempted current + nxt.

           Fix: after computing forward_score (current + nxt), also compute
           reversed_score (nxt + current). Use whichever scores higher as the
           candidate. If best_score >= 80 and neither word is a valid standalone
           skill (the v26 guard), commit the best-order merge.

           Also extracted a shared _score_against_allowlist() helper to avoid
           repeating the max(fuzz.ratio, fuzz.token_sort_ratio) loop four times.

           Before: kept_lines = ["Tracking", "Bug", "Usability Testing", ...]
                   -> "Tracking" and "Bug" emitted separately, LLM drops both
                   -> "Bug Tracking" never reconstructed  (unmatched skill)
           After:  kept_lines = ["Bug Tracking", "Usability Testing", ...]
                   -> "Bug Tracking" matched normally  ✓

Changes (v26): fixed _merge_split_lines() incorrectly merging an OCR-garbled
           single-word skill with the following valid standalone skill.

           Problem: "SAP2OOO" (OCR garbling of "SAP2000") scores < 80 on its
           own because it is unrecognised. "Structural Analysis" on the next
           line scores ~100 as a standalone skill. The v24 merge conditions
           (combined >= 80 AND current < 80) both passed because token_sort_ratio
           of "SAP2OOO Structural Analysis" against "Structural Analysis" is
           high, so the two lines were fused into one token. The fused token
           matched "Structural Analysis" and "SAP2OOO" was silently lost —
           it never reached apply_ocr_corrections / _digit_letter_candidates
           where it would have been corrected to "SAP2000".

           Fix: added a 4th guard — also compute nxt_score (how well the next
           line scores as a standalone skill). If nxt_score >= 80 the next line
           is already a valid independent skill; do NOT merge. Both lines are
           kept separate so "SAP2OOO" flows through OCR correction downstream.

           Before: kept_lines = ["SAP2OOO Structural Analysis", "AutoCAD", ...]
                   discovered  = ["Structural Analysis", "AutoCAD", ...]   (SAP2000 lost)
           After:  kept_lines = ["SAP2OOO", "Structural Analysis", "AutoCAD", ...]
                   discovered  = ["SAP2000", "Structural Analysis", "AutoCAD", ...]

Changes (v30): replaced direct_scan_section + fuzzy_scan pipeline in
           discover_skills_dynamically with an SBERT-based discovery step.

           Previously discovered_skills was produced by:
             1. direct_scan_section (fuzzy string match against allowlist)
             2. fuzzy_scan fallback (if direct scan found nothing)

           Now the pipeline is:
             1. extract_skills_section (unchanged — produces skills_section_rule_based)
             2. SBERT encodes every comma-delimited entry from the rule-based section
             3. SBERT encodes every skill in the Firestore-sourced allowlist
             4. For each entry, cosine similarity is computed against all allowlist
                embeddings; any allowlist skill scoring >= 0.80 is accepted
             5. The matched canonical allowlist skill name is used (not the raw entry),
                so OCR variants like "Networking Troubleshooting" correctly resolve
                to the canonical "Network Troubleshooting"
             6. direct_scan_section is retained as a pre-pass for exact/near-exact
                hits (fast path) before SBERT is invoked on remaining unmatched entries
             7. fuzzy_scan is retained as a final safety net if both passes produce
                nothing

           Rationale: SBERT semantic similarity catches skill synonyms and OCR
           variants that fuzzy string matching misses. For example:
             "Systems Administration" -> cosine 0.91 -> "System Administration" ✓
             "Networking"             -> cosine 0.83 -> "Network Troubleshooting" ✗
                                     (below 0.80 threshold — correct rejection)
           The 0.80 cosine threshold is strict enough to avoid false positives
           while catching genuine semantic matches.

Changes (v29 — this change): fixed _merge_split_lines() v26 guard firing on
           weak partial-word matches, preventing valid two-word skill merges.

           Problem: "Administration" scored 80 against "Server Administration"
           via token_sort_ratio (partial substring match). The v26 guard
           condition `current_score >= 80` fired, causing the loop to bail out
           before attempting the reversed join "System Administration".
           Result: "Administration" and "System" were emitted as separate tokens
           and both failed direct_scan_section individually, leaving
           "System Administration" as an unmatched skill in Stage 3.

           Root cause: the v26 guard threshold of 80 is too permissive. It was
           designed to protect *valid standalone skills* from being consumed
           into a merge. A score of 80 via token_sort_ratio on a single word
           against a two-word skill is a partial match, not a standalone skill.

           Fix: raised the v26 guard threshold from >= 80 to >= 85 in both
           the current_score and nxt_score checks. This ensures only genuinely
           high-confidence standalone skills (e.g. "PowerShell" scoring ~100
           against "PowerShell") block the merge, while weak partial matches
           like "Administration" -> "Server Administration" at score 80 are
           ignored and the merge is still attempted.

           Before: current_score("Administration") = 80 >= 80  -> bail, no merge
           After:  current_score("Administration") = 80 <  85  -> continue to merge
                   reversed join "System Administration" scores ~100 -> merged ✓

Changes (v31): fixed _is_banner_line() false-positive on skill phrases like
           "CRUD Application Development" that pass all three banner layers
           because they are structurally identical to name/title banners.

           Root cause: Layer 3 (allowlist guard) used only fuzz.ratio and
           fuzz.token_sort_ratio with a threshold of 80. "CRUD Application
           Development" scores below 80 against any single allowlist entry
           because it is a compound phrase not present verbatim. However,
           token_set_ratio computes similarity on the token intersection, so
           "CRUD Application Development" scores high against any allowlist
           entry that shares the token "CRUD" (e.g. "CRUD Operations").

           Fix: added fuzz.token_set_ratio to the score calculation and lowered
           the threshold from 80 to 65. token_set_ratio handles subset matches,
           catching legitimate skill phrases that contain a recognisable
           allowlist keyword even when the full string doesn't match verbatim.
           Threshold 65 is still high enough to reject real name banners like
           "Maria Santos" or "JOHN DELA CRUZ" which share no tokens with skills.

Changes (v36): fixed MongoDB being missed when "MERN Stack (MongoDB, Express.js,
           React, Node.js)" is emitted as a single entry after _rejoin_open_parens
           reassembles the multi-line parenthetical.

           Root cause: SBERT encodes the full string "MERN Stack (MongoDB,
           Express.js, React, Node.js)" as a single vector. Its cosine similarity
           against any single allowlist entry (e.g. "MongoDB") is only ~0.66
           because the embedding represents the entire compound phrase, not the
           individual technologies inside it. So MongoDB, Express.js, React, and
           Node.js were all missed in the SBERT pass.

           Fix: introduced _expand_parentheticals() helper. For any entry
           containing a parenthetical, it emits:
             - the base phrase (text before the opening paren)
             - each comma/semicolon-separated item inside the parens
           as separate candidates. This runs in discover_skills_dynamically
           after _expand_slash and before the SBERT pass, and also inside
           _sbert_discover_skills before encoding so that every component of
           a compound parenthetical entry gets its own SBERT comparison.

           Example:
             "MERN Stack (MongoDB, Express.js, React, Node.js)"
             -> ["MERN Stack", "MongoDB", "Express.js", "React", "Node.js"]
             -> MongoDB cosine=1.0 ✓, Express.js cosine=1.0 ✓, etc.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
from rapidfuzz import fuzz

from config import BLOB_FUZZY_THRESHOLD
from core.utils import (
    normalize,
    build_allow_norm_sorted,
    canonicalize,
    direct_scan_section,
    fuzzy_scan,
)
from core.ocr_correction import (
    OcrCorrector,
    build_ocr_corrector,
    apply_ocr_corrections,
    get_sbert,
)

log = logging.getLogger(__name__)

_cached_allowlist: list[str] = []

# If win=1 scores >= this, commit immediately without trying longer windows.
_SINGLE_TOKEN_CONFIDENCE: int = 85

# v30: SBERT cosine threshold for skill discovery.
# Only allowlist skills whose cosine similarity to a candidate entry
# meets or exceeds this value are accepted as discovered skills.
_SBERT_COSINE_THRESHOLD: float = 0.80

_SKILL_HEADING_RE = re.compile(
    r"^[^\w]*"                                                   # leading emoji/bullet/whitespace
    r"(?:"
    r"(?:key\s+|core\s+|technical\s+|professional\s+)?skills?"  # Skill, Skills, Key Skills, Technical Skills
    r"(?:\s*(?:&|and)\s*(?:tools?|technologies?))?"             #   optionally suffixed with & Tools / & Technologies
    r"|(?:technical\s+|core\s+|key\s+)?competenc(?:ies|y)"     # Competencies, Technical Competencies, Core Competencies
    r"|(?:areas?\s+of\s+)?technical\s+expertise"                # Technical Expertise, Areas of Technical Expertise
    r"|tools?\s*(?:&|and)\s*technologies?"                      # Tools & Technologies, Tools and Technologies
    r"|(?:technical\s+)?proficienc(?:ies|y)"                    # Proficiencies, Proficiency, Technical Proficiencies
    r")"
    r"\s*:?\s*$",                                               # optional colon -> must end the line here
    re.I | re.M,
    )

# v15: removed "projects?" -- too ambiguous as a section boundary.
# v14: added (?![\w\-]) after "objective" to avoid matching "Objective-C".
_SKILLS_END_RE = re.compile(
    r"^\s*[^\w]*(certifications?|licen[cs]es?\s*(&|and)?\s*certifications?|"
    r"professional\s+certifications?|credentials?|accreditations?|"
    r"work\s+experience|experience|education|employment|awards?|publications?|"
    r"references?|volunteer|languages?|interests?|activities|summary|objective(?![\w\-]))",
    re.I | re.M,
    )

_BANNER_STRUCT_RE = re.compile(
    r"^[A-Za-záéíóúñüÁÉÍÓÚÑÜ][A-Za-záéíóúñüÁÉÍÓÚÑÜ\s\-]{4,59}$"
)

_CONTACT_LINE_RE = re.compile(
    r"@"
    r"|(?:\+?[\d\s\-\(\)]{7,})"
    r"|[\U00002709\U0000260E\U0001F4DE\U0001F4E7\U0001F4E9\U0001F4CC\U0001F4CD\U0001F4F1]"
    r"|\b(?:www\.|http)",
    re.I,
)

_EXPERIENCE_LINE_RE = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|present|current"
    r"|\b(developed?|managed?|led|built|designed?|implemented?|integrated?|"
    r"optimized?|maintained?|participated?|created?|delivered?|deployed?|"
    r"automated?|reduced?|improved?|increased?|collaborated?|coordinated?)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Parenthesis protection helpers (v17)
# ---------------------------------------------------------------------------

def _protect_parens(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace commas inside parentheses with a unique placeholder so they are
    not treated as skill delimiters during comma-splitting.

    Example:
        "Process Simulation Software (Aspen HYSYS, Aspen Plus), AutoCAD"
        ->
        "Process Simulation Software __PAREN_0__, AutoCAD"
        pmap = {"__PAREN_0__": "(Aspen HYSYS, Aspen Plus)"}

    Returns the protected text and the placeholder->original mapping.
    """
    placeholder_map: dict[str, str] = {}

    def replacer(m: re.Match) -> str:
        key = f"__PAREN_{len(placeholder_map)}__"
        placeholder_map[key] = m.group(0)
        return key

    protected = re.sub(r"\([^)]*\)", replacer, text)
    return protected, placeholder_map


def _restore_parens(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore placeholders back to their original parenthesised strings."""
    for key, original in placeholder_map.items():
        text = text.replace(key, original)
    return text


# ---------------------------------------------------------------------------
# Banner-line detection
# ---------------------------------------------------------------------------

def _is_subsumed(skill: str, others: list[str]) -> bool:
    """
    Return True only when *skill* is a parenthetical fragment that was
    literally split out of a longer skill in *others*.

    v18: only drops skills that appear as parenthetical fragments, not
    skills that merely share words with a longer skill (e.g. "C Programming"
    and "Embedded C programming" are distinct and both kept).

    v19: extended to catch skills that appear after a comma inside
    parentheses (e.g. "AutoCAD" in "CAD Software (SolidWorks, AutoCAD)").
    """
    s_lower   = skill.lower()
    s_escaped = re.escape(s_lower)
    pattern   = re.compile(r'(?<![a-z0-9])' + s_escaped + r'(?![a-z0-9])')

    for other in others:
        if other == skill or other.lower() == s_lower:
            continue
        other_l = other.lower()
        if len(other_l) <= len(s_lower):
            continue
        m = pattern.search(other_l)
        if not m:
            continue

        following = other_l[m.end()] if m.end() < len(other_l) else ''

        if following not in (',', ')'):
            continue

        before_match = other_l[:m.start()]
        if before_match.count('(') > before_match.count(')'):
            return True

    return False


def _is_banner_line(line: str, allowlist: list[str]) -> bool:
    """
    Return True if *line* looks like a personal name or job-title banner
    injected by certain PDF layouts between the skills list and the next
    section heading.

    Three layered checks must ALL pass:
      Layer 1 -- structural: only letters/spaces/hyphens, 2+ words, length 5-60.
      Layer 2 -- word shape: every word is pure Title Case or pure ALL CAPS.
      Layer 3 -- allowlist guard: fuzzy score < 65 against all known skills
                 (lowered from 80, and now includes token_set_ratio so that
                 partial-word matches like "CRUD Application Development"
                 against "CRUD Operations" are caught).

    Changes (v31): lowered allowlist guard threshold from 80 to 65 and added
    fuzz.token_set_ratio to the score calculation. token_set_ratio handles
    subset matches — e.g. "CRUD Application Development" scores high against
    "CRUD Operations" because "CRUD" is a shared token. This prevents the
    banner guard from firing on legitimate skill phrases that contain a
    recognisable allowlist keyword even when the full string doesn't match.
    """
    if not _BANNER_STRUCT_RE.match(line):
        return False
    if " " not in line:
        return False
    if any(ch.isdigit() for ch in line):
        return False

    for word in line.split():
        w = word.strip("-")
        if not w:
            continue
        is_title_case = w[0].isupper() and w[1:].islower()
        is_all_caps   = w.isupper()
        if not (is_title_case or is_all_caps):
            return False

    if allowlist:
        line_norm = line.lower().strip()
        for skill in allowlist:
            score = max(
                fuzz.ratio(line_norm, skill.lower()),
                fuzz.token_sort_ratio(line_norm, skill.lower()),
                fuzz.token_set_ratio(line_norm, skill.lower()),  # v31: subset match
            )
            if score >= 65:  # v31: lowered from 80
                log.debug(f"[BannerCheck] '{line}' scores {score} vs '{skill}' -> treated as skill")
                return False

    return True


# ---------------------------------------------------------------------------
# OCR blob segmentation
# ---------------------------------------------------------------------------

def split_ocr_blob(line: str, allowlist: list[str]) -> list[str]:
    """
    Segment a collapsed single-line OCR blob into individual skill phrases.

    Algorithm (v7 -- confidence-first):
      For each position i:
      1. Check win=1. If score >= _SINGLE_TOKEN_CONFIDENCE (85), commit immediately.
      2. Otherwise try all window lengths and commit to the highest scorer.
      3. If nothing clears the threshold, emit the raw token.
    """
    if not allowlist or not line.strip():
        return [line.strip()] if line.strip() else []

    sorted_allow = sorted(allowlist, key=lambda a: len(a.split()), reverse=True)
    norm_allow   = [(normalize(a), a) for a in sorted_allow]
    max_win      = max(len(a.split()) for a in sorted_allow) if sorted_allow else 1

    tokens = line.split()
    n      = len(tokens)
    result: list[str] = []
    i = 0

    while i < n:
        w1_norm = normalize(tokens[i])
        w1_thr  = (BLOB_FUZZY_THRESHOLD - 10 if len(w1_norm) <= 5
                   else BLOB_FUZZY_THRESHOLD - 5 if len(w1_norm) <= 9
        else BLOB_FUZZY_THRESHOLD)

        w1_best_score, w1_best_val = 0, ""
        for akey, aval in norm_allow:
            s = max(fuzz.token_sort_ratio(w1_norm, akey), fuzz.ratio(w1_norm, akey))
            if s > w1_best_score:
                w1_best_score, w1_best_val = s, aval

        if w1_best_score >= _SINGLE_TOKEN_CONFIDENCE:
            log.debug(f"[BlobSplit] i={i} single-token '{tokens[i]}' -> '{w1_best_val}' ({w1_best_score})")
            result.append(w1_best_val)
            i += 1
            continue

        best_score = w1_best_score if w1_best_score >= w1_thr else 0
        best_canon = w1_best_val   if w1_best_score >= w1_thr else None
        best_len   = 1             if w1_best_score >= w1_thr else 0

        for win_len in range(min(max_win, n - i), 1, -1):
            window_norm = normalize(" ".join(tokens[i : i + win_len]))
            if len(window_norm) < 3:
                continue
            wlen = len(window_norm)
            thr  = (BLOB_FUZZY_THRESHOLD - 10 if wlen <= 5
                    else BLOB_FUZZY_THRESHOLD - 5 if wlen <= 9
            else BLOB_FUZZY_THRESHOLD)

            win_best_score, win_best_val = 0, ""
            for akey, aval in norm_allow:
                s = max(fuzz.token_sort_ratio(window_norm, akey), fuzz.ratio(window_norm, akey))
                if s > win_best_score:
                    win_best_score, win_best_val = s, aval

            if win_best_score >= thr and win_best_score > best_score:
                best_score = win_best_score
                best_canon = win_best_val
                best_len   = win_len

        if best_canon:
            log.debug(f"[BlobSplit] i={i} multi-token '{best_canon}' (win={best_len}, score={best_score})")
            result.append(best_canon)
            i += best_len
        else:
            raw_tok = tokens[i].strip(".,;:()[]")
            if raw_tok:
                result.append(raw_tok)
            i += 1

    return result


# ---------------------------------------------------------------------------
# Line-merging helper (v24, fixed v26, v27, v29)
# ---------------------------------------------------------------------------

def _score_against_allowlist(text_norm: str, allowlist: list[str]) -> int:
    """Return the best fuzzy score for *text_norm* against any allowlist entry."""
    return max(
        (max(fuzz.ratio(text_norm, normalize(a)),
             fuzz.token_sort_ratio(text_norm, normalize(a)))
         for a in allowlist),
        default=0,
    )


def _merge_split_lines(lines: list[str], allowlist: list[str]) -> list[str]:
    """
    Merge consecutive OCR-split lines that together form a single allowlist skill.

    Some OCR layouts break compound skill names across lines, e.g.:
        "Healthcare"
        "Technology"
    instead of "Healthcare Technology".

    Algorithm:
      For each line i, if it is a single word AND the next line is also short
      (1-2 words), attempt both forward AND reversed joins and commit whichever
      scores higher against the allowlist (provided both guards pass).

    Guards — ALL must pass before any merge is committed:
      1. The current line is a single word (not already a complete phrase).
      2. The best join (forward or reversed) scores >= 80 against the allowlist.
      3. The current word alone does NOT score >= 85 (not a valid standalone skill).
      4. (v26, threshold raised in v29) The next line alone does NOT score >= 85
         (not a valid standalone skill). Prevents "SAP2OOO" + "Structural Analysis"
         fusing and losing "SAP2OOO" before OCR correction maps it to "SAP2000".
         Threshold raised from 80 to 85 in v29 to avoid weak partial matches
         (e.g. "Administration" scoring 80 against "Server Administration") from
         incorrectly blocking a valid merge like "System Administration".

    Reversed-pair logic (v27):
      OCR column layouts sometimes emit the words of a multi-word skill in
      reverse order, e.g. "Tracking" then "Bug" instead of "Bug Tracking".
      After checking the forward join (current + nxt), also check the reversed
      join (nxt + current). If the reversed join scores higher AND meets the
      threshold, use the reversed order as the merged line.

      Example:
        lines = ["Tracking", "Bug", "Usability Testing", ...]
        forward:  "Tracking Bug"   -> score ~60  (not a known skill)
        reversed: "Bug Tracking"   -> score ~100 (exact allowlist match)
        -> merged line = "Bug Tracking"  ✓

    v29 example:
        lines = ["Administration", "System", ...]
        current_score("Administration") = 80  <  85  -> guard does NOT fire
        forward:  "Administration System" -> low score
        reversed: "System Administration" -> score ~100
        -> merged line = "System Administration"  ✓
    """
    if not allowlist or len(lines) < 2:
        return lines

    merged: list[str] = []
    i = 0
    while i < len(lines):
        current       = lines[i].strip()
        current_words = current.split()

        if len(current_words) == 1 and i + 1 < len(lines):
            nxt       = lines[i + 1].strip()
            nxt_words = nxt.split()

            if 1 <= len(nxt_words) <= 2:
                current_norm  = normalize(current)
                nxt_norm      = normalize(nxt)

                current_score = _score_against_allowlist(current_norm, allowlist)
                nxt_score     = _score_against_allowlist(nxt_norm, allowlist)

                # v26 guard (threshold raised to 85 in v29):
                # if either word is already a high-confidence standalone skill,
                # do not attempt any merge — both lines are independent.
                # Threshold 85 (was 80) prevents weak partial matches like
                # "Administration" -> "Server Administration" at score 80 from
                # incorrectly blocking the merge of "System Administration".
                if current_score >= 85 or nxt_score >= 85:
                    if current_score < 85 and nxt_score >= 85:
                        log.debug(
                            f"[SkillSection] Skipping merge — next line is a valid "
                            f"standalone skill: {nxt!r} (nxt_score={nxt_score}). "
                            f"Keeping {current!r} separate for downstream OCR correction."
                        )
                    merged.append(current)
                    i += 1
                    continue

                # Both words are ambiguous on their own — try forward and reversed joins.
                forward  = current + " " + nxt
                reversed_ = nxt + " " + current

                forward_score  = _score_against_allowlist(normalize(forward),  allowlist)
                reversed_score = _score_against_allowlist(normalize(reversed_), allowlist)

                best_score     = max(forward_score, reversed_score)
                best_combined  = reversed_ if reversed_score > forward_score else forward

                if best_score >= 80:
                    log.debug(
                        f"[SkillSection] Merging split lines: "
                        f"{current!r} + {nxt!r} -> {best_combined!r} "
                        f"(fwd={forward_score}, rev={reversed_score}, "
                        f"current={current_score}, nxt={nxt_score})"
                    )
                    merged.append(best_combined)
                    i += 2
                    continue

        merged.append(current)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Open-parenthesis rejoining helper (v25)
# ---------------------------------------------------------------------------

def _rejoin_open_parens(lines: list[str]) -> list[str]:
    """
    Rejoin lines where OCR split an open parenthesis across multiple lines.

    Some OCR layouts break a compound skill with parentheses across lines, e.g.:
        "Process Simulation Software (Aspen"
        "HYSYS;"
        "Plus)"
    instead of:
        "Process Simulation Software (Aspen HYSYS; Plus)"

    Algorithm:
      Walk lines sequentially. When a line has more '(' than ')', start
      buffering. Append subsequent lines (space-separated) until the paren
      count balances. Emit the rejoined string as a single line.

    If the document ends with an unclosed parenthesis (malformed OCR), the
    buffer is emitted as-is so no content is silently lost.
    """
    result: list[str] = []
    buf = ""

    for line in lines:
        stripped = line.strip()
        if buf:
            buf = buf.rstrip() + " " + stripped
            if buf.count("(") <= buf.count(")"):
                result.append(buf)
                buf = ""
        else:
            if stripped.count("(") > stripped.count(")"):
                buf = stripped
            else:
                result.append(stripped)

    if buf:
        # Unclosed paren at end of block — emit anyway to avoid data loss.
        log.debug(f"[ReJoinParens] Unclosed parenthesis at EOF, emitting: {buf!r}")
        result.append(buf)

    return result


# ---------------------------------------------------------------------------
# Parenthesis-aware threshold helper (v25)
# ---------------------------------------------------------------------------

def _effective_threshold(candidate: str, base_threshold: int) -> int:
    """
    Return a (possibly lowered) fuzzy match threshold for *candidate*.

    Compound parenthetical skill names like:
        "Process Simulation Software (Aspen HYSYS, Aspen Plus)"
    naturally score lower on fuzzy ratio due to punctuation weight and
    string length. A blanket threshold of 82 rejects valid matches that
    score ~81.

    Strategy:
      - If the candidate contains '(' or ')', lower the threshold by 4
        points to account for punctuation drag on the fuzzy score.
      - Floor at 75 to prevent noise from very short or garbage strings.

    This reduction is scoped only to parenthetical candidates, so simple
    single-phrase skills remain subject to the stricter base threshold.
    """
    if "(" in candidate or ")" in candidate:
        return max(base_threshold - 4, 75)
    return base_threshold


# ---------------------------------------------------------------------------
# Parenthetical expansion helper (v36)
# ---------------------------------------------------------------------------

def _expand_parentheticals(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    For any entry containing a parenthetical, expand it into separate candidates
    ONLY when the inner parts are individually recognisable skills and the full
    compound is NOT itself a known allowlist skill.

    Three outcomes for an entry like "X (A, B, C)":
      1. full compound scores >= 75 against allowlist -> keep intact
         e.g. "Process Simulation Software (Aspen HYSYS, Aspen Plus)" is a
         known compound skill — kept as-is for downstream matching.
      2. full compound scores < 75 BUT at least one inner part scores >= 75
         -> expand: emit base + each inner part separately
         e.g. "MERN Stack (MongoDB, Express.js, React, Node.js)" — MongoDB
         etc. are individually in the allowlist, so expand.
      3. full compound scores < 75 AND no inner part scores >= 75
         -> keep intact — nothing inside is recognisable, don't destroy it
         e.g. "bla lba (asphen plus, asphen hyen)" — unknown compound,
         keep whole so downstream can still attempt a match.

    When no allowlist is provided, always expand (safe default).
    Entries without parentheses are always returned unchanged.

    Examples:
      "MERN Stack (MongoDB, Express.js, React, Node.js)"
      -> ["MERN Stack", "MongoDB", "Express.js", "React", "Node.js"]  ok

      "Process Simulation Software (Aspen HYSYS, Aspen Plus)"
      -> ["Process Simulation Software (Aspen HYSYS, Aspen Plus)"]  (kept intact) ok

      "bla lba (asphen plus, asphen hyen)"
      -> ["bla lba (asphen plus, asphen hyen)"]  (kept intact) ok

      "Flutter"  ->  ["Flutter"]  (no parens, unchanged) ok
    """
    expanded: list[str] = []
    for entry in entries:
        m = re.match(r'^(.*?)\s*\(([^)]+)\)\s*$', entry.strip())
        if not m:
            expanded.append(entry)
            continue

        base  = m.group(1).strip()
        inner = m.group(2).strip()
        parts = [p.strip() for p in re.split(r'[,;]\s*', inner) if p.strip()]

        # No allowlist — always expand.
        if not allowlist:
            if base:
                expanded.append(base)
            expanded.extend(parts)
            continue

        # Check if the full compound is a known allowlist skill.
        full_score = max(
            (max(fuzz.ratio(normalize(entry), normalize(a)),
                 fuzz.token_sort_ratio(normalize(entry), normalize(a)))
             for a in allowlist),
            default=0,
        )
        if full_score >= 75:
            log.debug(f"[ExpandParens] Keeping {entry!r} intact (full_score={full_score} >= 75)")
            expanded.append(entry)
            continue

        # Check if any inner part is individually recognisable.
        any_part_matches = any(
            max(
                (max(fuzz.ratio(normalize(p), normalize(a)),
                     fuzz.token_sort_ratio(normalize(p), normalize(a)))
                 for a in allowlist),
                default=0,
            ) >= 75
            for p in parts
        )

        if any_part_matches:
            # Enumerated technologies inside parens — expand into parts.
            if base:
                expanded.append(base)
                log.debug(f"[ExpandParens] base: {base!r}")
            for p in parts:
                expanded.append(p)
                log.debug(f"[ExpandParens] inner part: {p!r}")
        else:
            # Nothing inside is recognisable — keep original intact.
            log.debug(f"[ExpandParens] Keeping {entry!r} intact (no inner parts matched)")
            expanded.append(entry)

    return expanded


# ---------------------------------------------------------------------------
# Subheader stripping helper (v37)
# ---------------------------------------------------------------------------

def _strip_subheader(line: str) -> str:
    """
    Strip a category subheader prefix from a skill line.

    Some resumes format their skills section with subheaders like:
        "Programming: Python, Java, CSS"
        "Frameworks: React, Django, Flask"
        "Tools & Software: Git, Docker, Jira"

    The text before the colon is a label, not a skill. This helper removes
    it so only the actual skills reach the comma-split.

    Rules:
      - Only strip when the text before the colon is short (<= 5 words)
        and contains no comma.
      - If no colon is present, the line is returned unchanged.

    Examples:
      "Programming: Python, Java, CSS"  -> "Python, Java, CSS"
      "Tools & Software: Git, Docker"   -> "Git, Docker"
      "Python, Java, CSS"               -> unchanged (no colon)
    """
    if ":" not in line:
        return line
    colon_idx = line.index(":")
    before = line[:colon_idx].strip()
    after  = line[colon_idx + 1:].strip()
    if "," in before:
        return line
    if len(before.split()) > 5:
        return line
    if not after:
        return line
    log.debug(f"[StripSubheader] Stripping label {before!r} from {line!r}")
    return after


def extract_skills_section(
        text:          str,
        allowlist:     list[str] | None = None,
        is_native_pdf: bool = False,
) -> str:
    """
    Return the skills section as a comma-separated string.
    Each entry is one complete skill phrase.

    v17: commas inside parentheses are protected before splitting so that
    compound skills like "Process Simulation Software (Aspen HYSYS, Aspen Plus)"
    are not fragmented into broken partial strings.

    v22: added prose-sentence guard in the comma-branch. When a comma-delimited
    token contains more than 5 words it is treated as a prose sentence and
    extraction stops immediately.

    v24: _merge_split_lines() is called on kept_lines before branching to
    merge OCR-split compound skill names like "Healthcare" / "Technology"
    into "Healthcare Technology".

    v25: _rejoin_open_parens() is called after _merge_split_lines() to
    reassemble parenthetical skill names that OCR split across multiple lines,
    e.g. ["Process Simulation Software (Aspen", "HYSYS;", "Plus)"] ->
    ["Process Simulation Software (Aspen HYSYS; Plus)"].

    v26: _merge_split_lines() now also checks nxt_score so that an
    OCR-garbled token followed by a valid standalone skill (e.g. "SAP2OOO"
    + "Structural Analysis") is never fused into a single token.

    v27: _merge_split_lines() now also tries the reversed join (nxt + current)
    so that OCR-flipped word pairs like "Tracking" / "Bug" are correctly
    reconstructed as "Bug Tracking".

    v29: _merge_split_lines() guard threshold raised from 80 to 85 so that
    weak partial matches (e.g. "Administration" -> "Server Administration"
    at score 80) no longer abort valid merges like "System Administration".
    """
    m = _SKILL_HEADING_RE.search(text)
    if not m:
        return ""
    rest  = text[m.end():]
    end_m = _SKILLS_END_RE.search(rest)
    block = rest[: end_m.start()].strip() if end_m else rest.strip()

    # v35: look back up to 5 lines before the Skills heading to capture any
    # skills that the layout interleave placed just above it.
    before_text  = text[: m.start()]
    before_lines = [l.strip() for l in before_text.splitlines() if l.strip()]
    lookback     = before_lines[-5:] if len(before_lines) >= 5 else before_lines

    pre_heading_skills: list[str] = []
    for lb_line in reversed(lookback):
        if _SKILLS_END_RE.match(lb_line):
            break
        if _CONTACT_LINE_RE.search(lb_line):
            break
        if _EXPERIENCE_LINE_RE.search(lb_line):
            break
        if len(lb_line.split()) > 5:
            break
        pre_heading_skills.insert(0, lb_line)

    raw_lines = pre_heading_skills + block.splitlines()

    effective_allowlist = allowlist if allowlist is not None else _cached_allowlist

    if is_native_pdf:
        kept_lines: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if kept_lines and _CONTACT_LINE_RE.search(stripped):
                log.debug(f"[SkillSection/native] Stopping at contact info: {stripped!r}")
                break
            if kept_lines and _is_banner_line(stripped, effective_allowlist):
                log.debug(f"[SkillSection/native] Stopping at banner: {stripped!r}")
                break
            kept_lines.append(stripped)
    else:
        kept_lines: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            words = stripped.split()
            if kept_lines and _CONTACT_LINE_RE.search(stripped):
                log.debug(f"[SkillSection] Stopping at contact info: {stripped!r}")
                break
            if kept_lines and _is_banner_line(stripped, effective_allowlist):
                log.debug(f"[SkillSection] Stopping at banner: {stripped!r}")
                break
            if (kept_lines
                    and len(words) >= 3
                    and _EXPERIENCE_LINE_RE.search(stripped)):
                break
            kept_lines.append(stripped)

    if not kept_lines:
        return ""

    kept_lines = _merge_split_lines(kept_lines, effective_allowlist)
    kept_lines = _rejoin_open_parens(kept_lines)

    joined_block = "\n".join(kept_lines)
    _BULLET_RE   = re.compile(r"^\s*[\u2022\-\*\u25e6\u25aa]|\s*\d+\.\s+", re.M)
    entries: list[str] = []

    if "," in joined_block:
        protected_block, pmap = _protect_parens(joined_block)
        protected_lines = protected_block.splitlines()

        stopped = False
        for line in protected_lines:
            if stopped:
                break
            line = _strip_subheader(line)
            for part in re.split(r"\s*,\s*", line):
                part = _restore_parens(part.strip(), pmap)
                if not part:
                    continue
                # v34: exclude parenthetical content from word count
                part_no_parens = re.sub(r'\([^)]*\)', '', part).strip()
                if len(part_no_parens.split()) > 6:
                    log.debug(f"[SkillSection] Prose sentence detected, stopping: {part!r}")
                    stopped = True
                    break
                # v36: expand parentheticals inline so "MERN Stack (MongoDB,
                # Express.js, React, Node.js)" becomes ["MERN Stack", "MongoDB",
                # "Express.js", "React", "Node.js"] before leaving this function.
                # This must happen here — after _restore_parens reassembles the
                # full parenthetical — not downstream where the entry is already
                # a broken fragment like "MERN Stack (MongoDB".
                for sub in _expand_parentheticals([part], effective_allowlist):
                    entries.append(sub)

    elif ";" in joined_block:
        for line in kept_lines:
            line = _strip_subheader(line)
            for part in re.split(r"\s*;\s*", line):
                if part.strip():
                    entries.append(part.strip())
    elif "|" in joined_block:
        for line in kept_lines:
            line = _strip_subheader(line)
            for part in line.split("|"):
                if part.strip():
                    entries.append(part.strip())
    elif _BULLET_RE.search(joined_block):
        for line in kept_lines:
            line = _strip_subheader(line)
            clean = re.sub(r"^\s*[\u2022\-\*\u25e6\u25aa]\s*|\s*\d+\.\s+", "", line).strip()
            if clean:
                entries.append(clean)
    elif len(kept_lines) > 1:
        entries = [_strip_subheader(line).strip() for line in kept_lines if line.strip()]
    else:
        blob = kept_lines[0] if kept_lines else ""
        if blob:
            entries = (
                split_ocr_blob(blob, effective_allowlist)
                if effective_allowlist
                else [blob]
            )

    entries = _expand_ampersand(entries, effective_allowlist)
    entries = _restore_ocr_slash(entries, effective_allowlist)
    entries = _expand_slash(entries, effective_allowlist)
    return ", ".join(entries)


# ---------------------------------------------------------------------------
# Ampersand expansion helper (v28)
# ---------------------------------------------------------------------------

def _expand_ampersand(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    Split compound skill entries joined by ' & ' or ' and ' into separate
    skills — but ONLY when the full compound does not already match any
    allowlist entry at score >= 90.

    If the full compound IS in the allowlist (e.g. "Jest & Enzyme" exists as
    a single canonical entry), keep it as-is. If it is NOT recognised as a
    unit, split it so each part can be matched independently.

    When no allowlist is provided, always split (safe default for section
    debug calls where allowlist may be None).

    Examples:
      "HTML5 & CSS3"  allowlist has "HTML5", "CSS3" separately
        -> full_score ~81  -> split -> ["HTML5", "CSS3"]  ✓

      "Jest & Enzyme" allowlist has "Jest & Enzyme" as one entry
        -> full_score 100  -> keep  -> ["Jest & Enzyme"]  ✓

      "HTML/CSS & CSS3" (OCR-mangled form of "HTML5 & CSS3")
        -> full_score ~81 < 90  -> split -> ["HTML/CSS", "CSS3"]
        -> "HTML/CSS" fuzzy-matches "HTML5" downstream  ✓
    """
    _AMP_RE = re.compile(r'\s+(?:&|and)\s+', re.I)

    expanded: list[str] = []
    for entry in entries:
        if not _AMP_RE.search(entry):
            expanded.append(entry)
            continue

        # Check whether the full compound is a known allowlist skill.
        if allowlist:
            full_norm  = normalize(entry)
            full_score = max(
                (max(fuzz.ratio(full_norm, normalize(a)),
                     fuzz.token_sort_ratio(full_norm, normalize(a)))
                 for a in allowlist),
                default=0,
            )
            if full_score >= 90:
                log.debug(
                    f"[ExpandAmp] Keeping {entry!r} — "
                    f"full_score={full_score} >= 90 (known compound skill)"
                )
                expanded.append(entry)
                continue

        # Full compound not recognised — split on every connector.
        parts = [p.strip() for p in _AMP_RE.split(entry) if p.strip()]
        log.debug(f"[ExpandAmp] Splitting {entry!r} -> {parts}")
        expanded.extend(parts)

    return expanded


# ---------------------------------------------------------------------------
# SBERT-based skill discovery (v30)
# ---------------------------------------------------------------------------

def _sbert_discover_skills(
        entries:       list[str],
        allowlist:     list[str],
        already_found: set[str],
        threshold:     float = _SBERT_COSINE_THRESHOLD,
) -> list[str]:
    """
    For each entry in *entries* not already in *already_found*, encode it
    with SBERT and compare against SBERT embeddings of every allowlist skill.
    Accept the best-scoring allowlist skill when cosine similarity >= threshold.

    Returns a list of canonical allowlist skill names (deduplicated, preserving
    order of first match).

    v36: _expand_parentheticals() is called on entries before encoding so that
    compound entries like "MERN Stack (MongoDB, Express.js, React, Node.js)"
    are split into ["MERN Stack", "MongoDB", "Express.js", "React", "Node.js"]
    and each component is encoded and matched independently. Without this,
    the full compound string scores only ~0.66 against "MongoDB" and all four
    technologies are missed.

    Parameters
    ----------
    entries       : comma-delimited entries from skills_section_rule_based,
                    already expanded by _expand_ampersand.
    allowlist     : full Firestore-sourced allowlist (required_skills across
                    all jobs in corpus).
    already_found : dedup keys (canon.lower().strip()) from the direct-scan
                    fast path — skipped here to avoid re-processing.
    threshold     : cosine similarity floor (default 0.80).
    """
    if not entries or not allowlist:
        return []

    sbert = get_sbert()

    # v36: expand parentheticals before encoding so each component of a
    # compound entry like "MERN Stack (MongoDB, Express.js, React, Node.js)"
    # gets its own SBERT comparison rather than being buried in the compound.
    # allowlist check already done upstream in discover_skills_dynamically
    entries = _expand_parentheticals(entries)

    # Encode allowlist once — reused for every candidate entry.
    log.info(f"[SBERT] Encoding {len(allowlist)} allowlist skills...")
    allow_vecs = sbert.encode(
        allowlist,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)                          # shape: (N_allowlist, D)

    # Only encode entries that haven't already been matched by direct scan.
    pending = [e for e in entries if e.lower().strip() not in already_found]
    if not pending:
        return []

    log.info(f"[SBERT] Encoding {len(pending)} candidate entries...")
    entry_vecs = sbert.encode(
        pending,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)                          # shape: (N_pending, D)

    # Cosine similarity matrix: (N_pending, N_allowlist)
    # Because embeddings are L2-normalised, dot product == cosine similarity.
    sim_matrix = entry_vecs @ allow_vecs.T        # shape: (N_pending, N_allowlist)

    discovered: list[str] = []
    seen:       set[str]  = set(already_found)

    for i, entry in enumerate(pending):
        best_idx   = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_idx])

        if best_score >= threshold:
            canon     = allowlist[best_idx]
            dedup_key = canon.lower().strip()
            if dedup_key not in seen:
                seen.add(dedup_key)
                discovered.append(canon)
                log.debug(
                    f"[SBERT] '{entry}' -> '{canon}' "
                    f"(cosine={best_score:.4f} >= {threshold})"
                )
        else:
            log.debug(
                f"[SBERT] '{entry}' -> no match "
                f"(best='{allowlist[best_idx]}' cosine={best_score:.4f} < {threshold})"
            )

    return discovered


# ---------------------------------------------------------------------------
# OCR slash restoration helper (v33)
# ---------------------------------------------------------------------------

def _restore_ocr_slash(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    Restore slashes that OCR corrupted into 'I' (uppercase i) or 'l' (lowercase L).

    OCR commonly misreads '/' as 'I' or 'l' between two uppercase letter groups:
      "QAIQC Procedures"  -> "QA/QC Procedures"
      "TCPIIP"            -> "TCP/IP"

    Algorithm:
      For each entry, search for runs of uppercase letters separated by I or l.
      If replacing that character with '/' produces a string that scores higher
      against the allowlist than the original, commit the replacement.
      If no allowlist is provided, always attempt the replacement.

    This runs BEFORE _expand_slash so that corrupted forms like "QAIQC Procedures"
    are first restored to "QA/QC Procedures" and then correctly split into
    "QA Procedures" and "QC Procedures" by _expand_slash.
    """
    _OCR_SLASH_RE = re.compile(r'([A-Z]{1,6})[Il]([A-Z]{1,6})')

    restored: list[str] = []
    for entry in entries:
        m = _OCR_SLASH_RE.search(entry)
        if not m:
            restored.append(entry)
            continue

        candidate = (
                entry[:m.start()]
                + m.group(1) + '/' + m.group(2)
                + entry[m.end():]
        )

        if allowlist:
            orig_score = max(
                (max(fuzz.ratio(normalize(entry),     normalize(a)),
                     fuzz.token_sort_ratio(normalize(entry),     normalize(a)))
                 for a in allowlist),
                default=0,
            )
            cand_score = max(
                (max(fuzz.ratio(normalize(candidate), normalize(a)),
                     fuzz.token_sort_ratio(normalize(candidate), normalize(a)))
                 for a in allowlist),
                default=0,
            )
            if cand_score > orig_score:
                log.debug(
                    f"[RestoreSlash] '{entry}' -> '{candidate}' "
                    f"(orig_score={orig_score}, cand_score={cand_score})"
                )
                restored.append(candidate)
                continue
        else:
            log.debug(f"[RestoreSlash] '{entry}' -> '{candidate}' (no allowlist)")
            restored.append(candidate)
            continue

        restored.append(entry)

    return restored


# ---------------------------------------------------------------------------
# Slash expansion helper (v32)
# ---------------------------------------------------------------------------

def _expand_slash(entries: list[str], allowlist: list[str] | None = None) -> list[str]:
    """
    Split entries containing '/' into separate skills when the full compound
    is NOT a known allowlist skill but each half combined with trailing words is.

    The slash in skill names has two distinct meanings:
      - Compound acronym  : "TCP/IP", "HTML/CSS"   -> slash IS the skill name, keep
      - Two separate roles: "QA/QC Procedures"      -> slash means "and", split

    Algorithm:
      1. If the full entry scores >= 85 against the allowlist, it is a known
         compound skill — keep it as-is.
      2. Otherwise split on the slash: left_prefix + suffix, right_prefix + suffix
         (e.g. "QA/QC Procedures" -> "QA Procedures", "QC Procedures").
      3. If both halves score >= 75 against the allowlist, replace with the two
         separate skills.
      4. If either half fails the threshold, keep the original entry unchanged.

    Examples:
      "TCP/IP"           -> full_score=100 >= 85  -> keep  ✓
      "QA/QC Procedures" -> full_score low         -> split
                            "QA Procedures"  ~85 ✓
                            "QC Procedures"  ~85 ✓  -> ["QA Procedures", "QC Procedures"]
      "HTML/CSS"         -> full_score >= 85       -> keep  ✓
      "QA/QC"            -> full_score low, halves
                            "QA" and "QC" score low -> keep as-is  ✓
    """
    expanded: list[str] = []
    for entry in entries:
        if '/' not in entry:
            expanded.append(entry)
            continue

        # Step 1: check if the full entry is a known allowlist skill.
        if allowlist:
            full_norm  = normalize(entry)
            full_score = max(
                (max(fuzz.ratio(full_norm, normalize(a)),
                     fuzz.token_sort_ratio(full_norm, normalize(a)))
                 for a in allowlist),
                default=0,
            )
            if full_score >= 85:
                log.debug(
                    f"[ExpandSlash] Keeping {entry!r} — "
                    f"full_score={full_score} >= 85 (known compound skill)"
                )
                expanded.append(entry)
                continue

        # Step 2: find slash position and build left/right candidates.
        m = re.search(r'(\w+)/(\w+)(.*)', entry)
        if not m:
            expanded.append(entry)
            continue

        left_prefix  = m.group(1)
        right_prefix = m.group(2)
        suffix       = m.group(3)

        left_candidate  = (left_prefix  + suffix).strip()
        right_candidate = (right_prefix + suffix).strip()

        # Step 3: verify both halves are valid allowlist skills.
        if allowlist:
            left_score = max(
                (max(fuzz.ratio(normalize(left_candidate),  normalize(a)),
                     fuzz.token_sort_ratio(normalize(left_candidate), normalize(a)))
                 for a in allowlist),
                default=0,
            )
            right_score = max(
                (max(fuzz.ratio(normalize(right_candidate), normalize(a)),
                     fuzz.token_sort_ratio(normalize(right_candidate), normalize(a)))
                 for a in allowlist),
                default=0,
            )

            if left_score >= 75 and right_score >= 75:
                log.debug(
                    f"[ExpandSlash] Splitting {entry!r} -> "
                    f"['{left_candidate}' score={left_score}, "
                    f"'{right_candidate}' score={right_score}]"
                )
                expanded.extend([left_candidate, right_candidate])
                continue

            log.debug(
                f"[ExpandSlash] Keeping {entry!r} — "
                f"split halves scored ({left_score}, {right_score}), both need >= 75"
            )

        # Step 4: couldn't verify split — keep original.
        expanded.append(entry)

    return expanded


# ---------------------------------------------------------------------------
# Main skill discovery entry point
# ---------------------------------------------------------------------------

def discover_skills_dynamically(
        resume_text:   str,
        jobs_corpus:   list[dict],
        extra_skills:  list[str] | None = None,
        is_native_pdf: bool = False,
) -> list[str]:
    """
    Stage 2 — skill extraction via rule-based section parsing + SBERT discovery.

    Pipeline (v36 — parenthetical expansion before SBERT):
      1.  Build corpus allowlist from Firestore (jobs_corpus required_skills).
      2.  Build OcrCorrector.
      3.  extract_skills_section() — regex-based; produces skills_section_rule_based.
      4.  Expand ' & ' / ' and ' compounds (_expand_ampersand).
      5.  Restore OCR-corrupted slashes (_restore_ocr_slash).
      6.  Expand '/' compounds (_expand_slash).
      7.  Expand parentheticals (_expand_parentheticals): splits compound entries
          like "MERN Stack (MongoDB, Express.js, React, Node.js)" into individual
          candidates so each technology gets its own SBERT comparison.
      8.  SBERT PASS (primary) — _sbert_discover_skills: encodes all section entries
          (after parenthetical expansion) and all allowlist skills; accepts
          cosine >= 0.80 matches.
      9.  DIRECT SCAN (mop-up) — direct_scan_section: fuzzy string match for anything
          SBERT missed. Catches short acronyms (Git, AWS) and punctuation-heavy names
          (C#, .NET) where SBERT cosine may fall below 0.80 but fuzzy scores >= 82.
      10. SAFETY NET — fuzzy_scan fallback if both passes produced nothing.
      11. Drop parenthetical fragment duplicates (_is_subsumed).
    """
    global _cached_allowlist

    # ── 1. Build allowlist from Firestore corpus ──────────────────────────────
    allowlist: set[str] = set()
    for job in jobs_corpus:
        allowlist.update(job.get("required_skills", []))
    if extra_skills:
        allowlist.update(extra_skills)
    allowlist_list = sorted(allowlist)
    allow_norm     = {normalize(a): a for a in allowlist_list}
    sorted_pairs   = build_allow_norm_sorted(allowlist_list)
    corrector      = build_ocr_corrector(allowlist_list)

    _cached_allowlist = allowlist_list

    # ── 2. Extract rule-based skills section ──────────────────────────────────
    section = (
            extract_skills_section(
                resume_text,
                allowlist=allowlist_list,
                is_native_pdf=is_native_pdf,
            )
            or resume_text[:3000]
    )

    # ── 3. Expand compounds ───────────────────────────────────────────────────
    section_entries = [e.strip() for e in section.split(",") if e.strip()]
    section_entries = _expand_ampersand(section_entries, allowlist_list)
    section_entries = _restore_ocr_slash(section_entries, allowlist_list)
    section_entries = _expand_slash(section_entries, allowlist_list)
    # v36: expand parentheticals so "MERN Stack (MongoDB, Express.js, React, Node.js)"
    # yields ["MERN Stack", "MongoDB", "Express.js", "React", "Node.js"] — each
    # component gets its own SBERT comparison instead of being buried in one vector.
    section_entries = _expand_parentheticals(section_entries, allowlist_list)
    section         = ", ".join(section_entries)

    seen:      set[str]  = set()
    canonical: list[str] = []

    # ── 4. SBERT pass — semantic matching against Firestore allowlist ─────────
    # _sbert_discover_skills also calls _expand_parentheticals internally so
    # any parentheticals that survive to this point are handled consistently.
    sbert_hits = _sbert_discover_skills(
        entries       = section_entries,
        allowlist     = allowlist_list,
        already_found = seen,
        threshold     = _SBERT_COSINE_THRESHOLD,
    )
    for canon in sbert_hits:
        dedup_key = canon.lower().strip()
        if dedup_key not in seen:
            seen.add(dedup_key)
            canonical.append(canon)

    log.info(f"[Skills] After SBERT pass: {len(canonical)} skills matched.")

    # ── 5. Direct scan — mop up anything SBERT missed ────────────────────────
    for hit in direct_scan_section(
            section, sorted_pairs, threshold=82, corrector=corrector, allow_norm=allow_norm
    ):
        adjusted_threshold = _effective_threshold(hit, base_threshold=82)
        canon     = canonicalize(
            hit, allow_norm, threshold=adjusted_threshold,
            _sorted_pairs=sorted_pairs, corrector=corrector,
        )
        dedup_key = canon.lower().strip()
        if dedup_key and dedup_key not in seen:
            seen.add(dedup_key)
            canonical.append(canon)
            log.debug(f"[Skills/DirectScan] Mopped up after SBERT: '{hit}' -> '{canon}'")

    log.info(f"[Skills] After direct scan mop-up: {len(canonical)} skills total.")

    # ── 6. Safety net — fuzzy fallback if both passes found nothing ───────────
    if not canonical:
        log.info("[Skills] Direct scan + SBERT produced nothing — fuzzy fallback.")
        canonical = fuzzy_scan(section, allowlist_list)

    # ── 7. Drop parenthetical split fragments ─────────────────────────────────
    canonical = [s for s in canonical if not _is_subsumed(s, canonical)]

    log.info(f"[Skills] {len(canonical)} skills discovered.")
    return canonical