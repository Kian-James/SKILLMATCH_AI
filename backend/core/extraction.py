"""
core/extraction.py
==================
Stage 1 — Text acquisition (EasyOCR-only, via remote OCR microservice).

EasyOCR itself no longer runs in this process. It runs in a separate
Render service (see ocr-service/) so that this backend's memory footprint
(sentence-transformers + faiss + firebase-admin) stays independent of
EasyOCR's (easyocr + torch). This function keeps the exact same signature
and return shape it always had, so pipeline.py and everything downstream
needs zero changes.

Configure the microservice URL via the OCR_SERVICE_URL env var
(config.OCR_SERVICE_URL).
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from config import OCR_SERVICE_URL, OCR_SERVICE_TIMEOUT

log = logging.getLogger(__name__)


def extract_text(file_path: str) -> tuple[str, float, bool]:
    """
    Stage 1: EasyOCR for everything (PDF or image), run by the remote
    OCR microservice.

    Returns (text, ocr_quality_percent, is_native_pdf); is_native_pdf is
    always False.
    """
    log.info("[Stage 1] Calling OCR microservice at %s ...", OCR_SERVICE_URL)

    try:
        with open(file_path, "rb") as f:
            # Send only the basename as the multipart filename -- file_path
            # here is a LOCAL path on this machine (e.g. "resumes/foo.pdf" or
            # an absolute path) and is meaningless to the OCR service on the
            # other end. The service uses this filename to build its own
            # temp file path; if it receives a name containing "/" (our full
            # relative/absolute path), it tries to write into a subdirectory
            # that doesn't exist there and fails with FileNotFoundError. Only
            # the filename itself is relevant, never the directory it lives
            # in on this side.
            files = {"file": (Path(file_path).name, f)}
            resp = httpx.post(
                f"{OCR_SERVICE_URL}/extract-text",
                files=files,
                timeout=OCR_SERVICE_TIMEOUT,
            )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        log.error("[Stage 1] OCR microservice call failed: %s", exc)
        return "", 0.0, False

    text = data.get("text", "")
    quality = data.get("quality", 0.0)
    log.info("[Stage 1] OCR done (%d chars, quality %.1f%%).", len(text), quality)
    return text, quality, False