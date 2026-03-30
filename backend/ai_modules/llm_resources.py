"""
ai_modules/llm_resources.py
Uses Meta-Llama-3-8B-Instruct via HuggingFace InferenceClient (free tier) to:
  1. Generate learning resources for missing skills
  2. Normalize / enrich job dataset entries

Uses huggingface_hub InferenceClient with the hf-inference provider —
the correct modern approach replacing the deprecated raw inference API.
"""

from __future__ import annotations

import json
import re

from huggingface_hub import InferenceClient
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import HUGGINGFACE_API_TOKEN, LLM_MODEL_ID
from utils.cache_manager import get_llm_response, set_llm_response, make_key

# Singleton client
_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        if not HUGGINGFACE_API_TOKEN:
            raise RuntimeError(
                "HUGGINGFACE_API_TOKEN is not set. "
                "Get yours at https://huggingface.co/settings/tokens"
            )
        _client = InferenceClient(
            model=LLM_MODEL_ID,
            token=HUGGINGFACE_API_TOKEN,
        )
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    """
    Call LLM via HuggingFace InferenceClient chat_completion endpoint.
    This is the current correct approach for HuggingFace serverless inference.
    Retries up to 3 times on transient failures.
    """
    client = _get_client()

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=0.9,
    )

    return response.choices[0].message.content.strip()


def generate_learning_resources(skill: str) -> list[dict]:
    """
    Generate curated learning resources for a given skill.
    Cache-first: same skill won't hit the LLM twice within 24 hours.
    Returns a list of resource dicts with title, type, platform, duration, description.
    """
    cache_key = make_key("resources", skill.lower().strip())
    cached = get_llm_response(cache_key)
    if cached:
        logger.debug(f"Cache hit for skill resources: {skill}")
        return cached

    system = (
        "You are an expert career advisor and software engineer. "
        "You always respond with valid JSON only - no markdown, no explanation."
    )
    user = f"""List exactly 4 learning resources for the skill "{skill}" for software/engineering professionals.

Return ONLY a JSON array. Each item must have:
  - "title": resource name
  - "type": one of ["Course", "Book", "Tutorial", "Documentation", "Video"]
  - "platform": platform name (e.g., Coursera, YouTube, official docs, O'Reilly)
  - "duration": estimated time to complete (e.g., "10 hours", "3 days", "Self-paced")
  - "description": one sentence summary

No markdown. No explanation. Only the JSON array."""

    try:
        raw = _call_llm(system, user, max_tokens=600)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            set_llm_response(cache_key, result, ttl=86400)  # cache 24h
            return result
    except Exception as e:
        logger.error(f"LLM resource generation failed for '{skill}': {e}")

    return _fallback_resources(skill)


def _fallback_resources(skill: str) -> list[dict]:
    """Generic fallback if LLM call fails."""
    return [
        {
            "title": f"{skill} Official Documentation",
            "type": "Documentation",
            "platform": "Official Docs",
            "duration": "Self-paced",
            "description": f"Official documentation and guides for {skill}.",
        },
        {
            "title": f"Learn {skill} - Beginner to Advanced",
            "type": "Course",
            "platform": "Coursera / Udemy",
            "duration": "20-40 hours",
            "description": f"Comprehensive course covering {skill} fundamentals and advanced topics.",
        },
    ]


def normalize_job_with_llm(raw_title: str, skills: list[str]) -> dict:
    """
    Normalize a job title and validate/expand its skills list.
    Returns dict with normalized title and refined skills.
    """
    system = (
        "You are a senior technical recruiter. "
        "You always respond with valid JSON only - no markdown, no explanation."
    )
    user = f"""Normalize this job title and verify these skills are appropriate for it.
Job title: "{raw_title}"
Skills: {json.dumps(skills)}

Return ONLY a JSON object with:
  - "normalized_title": standard job title in Computing/Engineering (e.g., "Software Engineer")
  - "validated_skills": list of relevant skills (max 12, remove duplicates)
  - "suggested_certifications": list of 2-3 relevant certifications

No markdown. No explanation."""

    try:
        raw = _call_llm(system, user, max_tokens=400)
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.error(f"Job normalization failed for '{raw_title}': {e}")

    return {
        "normalized_title": raw_title.strip().title(),
        "validated_skills": skills[:12],
        "suggested_certifications": [],
    }