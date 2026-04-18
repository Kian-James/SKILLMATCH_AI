"""
utils/skill_normalizer.py

Fully dynamic skill normalizer — no hardcoded synonym dictionary.

  Tier 1 — Firestore synonym map (loaded at startup)
    load_dynamic_synonyms() fetches all documents from the Firestore
    skill_synonyms collection and builds _REVERSE_MAP in memory.
    generate_and_persist_synonyms() populates that collection by calling
    Zephyr 7B once per canonical skill (run once via /admin/generate-synonyms).
    After that, every startup is a fast Firestore read — no LLM calls.

  Tier 2 — Embedding fallback (runtime, for aliases not in Firestore)
    If Tier 1 misses, embed the raw skill and find the nearest canonical
    by cosine similarity. Canonical list = all skills ever seen in Firestore
    job corpus + any already in the synonym map.
    Threshold 0.82 keeps same-technology aliases (ReactJS→React, ~0.94)
    while rejecting cross-technology near-misses (Django→Flask, ~0.78).

  Tier 3 — Return as-is
    Genuinely unknown skills pass through unchanged. Correct: they may be
    valid skills used directly in the job dataset.

Performance:
  Tier 1: pure dict lookup — microseconds.
  Tier 2: embedding is diskcache-backed — fast after first call.
         Canonical embedding matrix computed once, held in memory.
  EmbeddingEngine lazy-loaded on first Tier 2 miss.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ai_modules.embedding_engine import EmbeddingEngine

EMBEDDING_FALLBACK_THRESHOLD: float = 0.82

# Runtime reverse map: alias.lower() → canonical skill name
# Populated entirely from Firestore at startup — no hardcoded entries.
_REVERSE_MAP: dict[str, str] = {}

# Canonical skill list — built from Firestore synonym map keys + job corpus.
# Used as the target vocabulary for Tier 2 embedding fallback.
_CANONICAL_LIST: list[str] = []


def _rebuild_canonical_list() -> None:
    """Rebuild _CANONICAL_LIST from whatever is currently in _REVERSE_MAP."""
    global _CANONICAL_LIST
    seen: dict[str, str] = {}  # lower → canonical (dedup)
    for canonical in _REVERSE_MAP.values():
        seen[canonical.lower()] = canonical
    _CANONICAL_LIST = sorted(seen.values())


class SkillNormalizer:
    """
    Fully dynamic two-tier skill normalizer.
    No hardcoded synonyms — all aliases come from Firestore.
    """

    def __init__(self):
        self._engine: "EmbeddingEngine | None" = None
        self._canonical_embs: np.ndarray | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def normalize(self, raw_skill: str) -> str | None:
        cleaned = self._clean(raw_skill)
        if not cleaned or len(cleaned) < 2:
            return None

        lower = cleaned.lower()

        # Tier 1: Firestore synonym map
        if lower in _REVERSE_MAP:
            return _REVERSE_MAP[lower]

        # Tier 2: embedding cosine similarity
        if _CANONICAL_LIST:
            try:
                canonical = self._embedding_normalize(cleaned)
                if canonical:
                    _REVERSE_MAP[lower] = canonical  # cache for this run
                    logger.debug(f"Embedding normalized: '{cleaned}' → '{canonical}'")
                    return canonical
            except Exception as e:
                logger.warning(f"Embedding fallback failed for '{cleaned}': {e}")

        # Tier 3: unknown skill — return as-is
        return cleaned.strip()

    def normalize_job_title(self, title: str) -> str:
        title = re.sub(r"\s*\(.*?\)", "", title).strip()
        return title.strip().title()

    def deduplicate_skills(self, skills: list[str]) -> list[str]:
        """
        1. Normalize via synonym map + embedding fallback
        2. Remove case duplicates
        3. Remove substring duplicates (keep the longer form)
        """
        normalized: list[str] = []
        seen_lower: set[str] = set()
        for s in skills:
            norm = self.normalize(s)
            if norm and norm.lower() not in seen_lower:
                seen_lower.add(norm.lower())
                normalized.append(norm)

        normalized.sort(key=len, reverse=True)
        final: list[str] = []
        for skill in normalized:
            if not any(skill.lower() in kept.lower() for kept in final):
                final.append(skill)
        return sorted(final)

    # ── Firestore synonym loading (Tier 1) ────────────────────────────────────

    @staticmethod
    def load_dynamic_synonyms() -> int:
        """
        Load all LLM-generated synonyms from Firestore into _REVERSE_MAP.
        Call once at startup after Firebase is initialized.
        Returns number of alias entries added.
        """
        try:
            from services.firebase_service import get_db
            db   = get_db()
            docs = db.collection("skill_synonyms").stream()
            added = 0
            for doc in docs:
                data      = doc.to_dict()
                canonical = data.get("canonical", doc.id)
                aliases: list[str] = data.get("aliases", [])
                # Map canonical → itself so exact-name lookups always hit Tier 1
                key_self = canonical.lower().strip()
                if key_self and key_self not in _REVERSE_MAP:
                    _REVERSE_MAP[key_self] = canonical
                    added += 1
                for alias in aliases:
                    key = alias.lower().strip()
                    if key and key not in _REVERSE_MAP:
                        _REVERSE_MAP[key] = canonical
                        added += 1
            _rebuild_canonical_list()
            logger.info(
                f"Dynamic synonyms loaded from Firestore: {added} entries, "
                f"{len(_CANONICAL_LIST)} canonical skills."
            )
            return added
        except Exception as e:
            logger.warning(f"Could not load dynamic synonyms from Firestore: {e}")
            return 0

    @staticmethod
    def generate_and_persist_synonyms(canonical_skills: list[str] | None = None) -> int:
        """
        One-time offline job: call Zephyr 7B to generate aliases for each
        canonical skill and persist to Firestore (skill_synonyms collection).
        Skips skills that already have a Firestore entry.
        Returns number of skills processed.

        canonical_skills: if None, uses all KNOWN_SKILLS from nlp_processor.
        """
        import json
        import re as _re
        from services.firebase_service import get_db
        from ai_modules.llm_resources import _call_llm

        if canonical_skills is None:
            # Fall back to skills already loaded from Firestore job corpus
            from ai_modules.nlp_processor import _FIRESTORE_SKILLS
            canonical_skills = list(_FIRESTORE_SKILLS)

        db   = get_db()
        coll = db.collection("skill_synonyms")

        existing      = {doc.id for doc in coll.stream()}
        to_generate   = [s for s in canonical_skills if s not in existing]

        if not to_generate:
            logger.info("All skills already have Firestore synonym entries.")
            return 0

        logger.info(f"Generating synonyms for {len(to_generate)} skills via LLM...")
        processed = 0

        system = (
            "You are a technical recruiter and software engineer. "
            "You always respond with valid JSON only — no markdown, no explanation."
        )

        for skill in to_generate:
            user = (
                f'List all common aliases, abbreviations, alternate spellings, '
                f'and package/library names for the technical skill "{skill}" '
                f'as they appear in resumes and job postings.\n'
                f'Return ONLY a JSON array of lowercase strings. Max 12 items. '
                f'Do not include the canonical skill name itself. No markdown.'
            )
            try:
                raw   = _call_llm(system, user, max_tokens=200)
                match = _re.search(r"\[.*?\]", raw, _re.DOTALL)
                if not match:
                    logger.warning(f"  No JSON array for '{skill}'")
                    continue

                aliases: list[str] = json.loads(match.group())
                aliases = [
                    a.lower().strip()
                    for a in aliases
                    if isinstance(a, str) and len(a.strip()) > 1
                ]

                coll.document(skill).set({
                    "canonical": skill,
                    "aliases":   aliases,
                    "source":    "llm_zephyr",
                })

                # Update live map immediately
                _REVERSE_MAP[skill.lower()] = skill
                for alias in aliases:
                    if alias not in _REVERSE_MAP:
                        _REVERSE_MAP[alias] = skill
                _rebuild_canonical_list()

                logger.info(f"  '{skill}' → {len(aliases)} aliases persisted.")
                processed += 1

            except Exception as e:
                logger.warning(f"  Failed for '{skill}': {e}")

        logger.info(f"Synonym generation done: {processed}/{len(to_generate)} skills.")
        return processed

    # ── Embedding fallback (Tier 2) ───────────────────────────────────────────

    def _get_engine(self) -> "EmbeddingEngine":
        if self._engine is None:
            from ai_modules.embedding_engine import EmbeddingEngine
            self._engine = EmbeddingEngine()
        return self._engine

    def _get_canonical_embs(self) -> np.ndarray:
        """Compute canonical embeddings once per process, refresh if list grew."""
        if (self._canonical_embs is None
                or self._canonical_embs.shape[0] != len(_CANONICAL_LIST)):
            logger.debug(
                f"Building canonical embedding matrix ({len(_CANONICAL_LIST)} skills)..."
            )
            self._canonical_embs = self._get_engine().embed_batch(_CANONICAL_LIST)
        return self._canonical_embs

    def _embedding_normalize(self, raw: str) -> str | None:
        if not _CANONICAL_LIST:
            return None
        emb        = self._get_engine().embed(raw)
        canon_embs = self._get_canonical_embs()
        sims       = canon_embs @ emb
        best_idx   = int(np.argmax(sims))
        best_sim   = float(sims[best_idx])
        if best_sim >= EMBEDDING_FALLBACK_THRESHOLD:
            return _CANONICAL_LIST[best_idx]
        return None

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[^\w\s.#+/&]", " ", text)
        return re.sub(r"\s+", " ", text).strip()