"""
core/extraction.py
==================
Stage 1 — Text acquisition.

Tries PyMuPDF native text first; falls back to EasyOCR when the native
extraction is too short or the file is an image.

Returns (text, ocr_quality_percent, is_native_pdf).
"""

from __future__ import annotations

import logging
import tempfile

import fitz          # PyMuPDF
import numpy as np
from tqdm import tqdm

from config import OCR_LANGUAGES, MIN_NATIVE_CHARS

log = logging.getLogger(__name__)

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        log.info("[OCR] Loading EasyOCR reader (gpu=False) ...")
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def _native_pdf_text(file_path: str) -> str:
    doc   = fitz.open(file_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return " ".join(pages).strip()


def _ocr_fallback(file_path: str) -> str:
    reader = _get_ocr_reader()

    if file_path.lower().endswith(".pdf"):
        try:
            from pdf2image import convert_from_path
            with tempfile.TemporaryDirectory() as tmpdir:
                pages = convert_from_path(file_path, dpi=200, output_folder=tmpdir)
                lines: list[str] = []
                for img in tqdm(pages, desc="OCR pages", unit="page", leave=False):
                    lines.extend(reader.readtext(np.array(img), detail=0, paragraph=True))
                return "\n".join(lines)
        except Exception as exc:
            log.warning(f"[OCR] pdf2image failed ({exc}); direct image fallback ...")

    result = reader.readtext(file_path, detail=0, paragraph=True)
    return "\n".join(result)


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(c.isalnum() for c in text) / len(text) * 100, 2)


def extract_text(file_path: str) -> tuple[str, float, bool]:
    """
    Stage 1: Try PyMuPDF native text; fall back to EasyOCR.

    Returns
    -------
    (text, ocr_quality_percent, is_native_pdf)
    """
    native = _native_pdf_text(file_path) if file_path.lower().endswith(".pdf") else ""

    if len(native) >= MIN_NATIVE_CHARS:
        log.info(f"[Stage 1] Native PDF text ({len(native)} chars).")
        return native, _alnum_ratio(native), True

    log.info("[Stage 1] Native text too short — EasyOCR fallback ...")
    ocr_text = _ocr_fallback(file_path)
    quality  = _alnum_ratio(ocr_text)
    log.info(f"[Stage 1] OCR done ({len(ocr_text)} chars, quality {quality}%).")
    return ocr_text, quality, False