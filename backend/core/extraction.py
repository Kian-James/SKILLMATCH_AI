"""
core/extraction.py
==================
Stage 1 — Text acquisition.

Tries PyMuPDF native text first; falls back to EasyOCR when the native
extraction is too short or the file is an image.

Returns (text, ocr_quality_percent, is_native_pdf).

Apple Silicon / macOS note
--------------------------
EasyOCR ships its own torch + libomp. On ARM64 Macs the OpenMP thread pool
can deadlock or segfault during tensor copy operations.  We prevent this by
setting thread-count env vars to 1 BEFORE any torch/numpy import, and by
running each OCR call in a spawned subprocess so that a crash there cannot
kill the whole evaluation run.
"""

from __future__ import annotations

# ── Must be set before torch / numpy are imported anywhere in this process ──
import os
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import logging
import multiprocessing
import tempfile

import fitz          # PyMuPDF
import numpy as np

from config import OCR_LANGUAGES, MIN_NATIVE_CHARS

log = logging.getLogger(__name__)

_ocr_reader = None


def _get_ocr_reader():
    """Lazy-load the EasyOCR reader, pinning torch to single-thread."""
    global _ocr_reader
    if _ocr_reader is None:
        # Pin torch threads BEFORE easyocr initialises its model
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        import easyocr
        log.info("[OCR] Loading EasyOCR reader (gpu=False) ...")
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def _native_pdf_text(file_path: str) -> str:
    doc   = fitz.open(file_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return " ".join(pages).strip()


# ---------------------------------------------------------------------------
# OCR subprocess worker
# ---------------------------------------------------------------------------
# Running OCR in a spawned child process means that any crash (segfault,
# OMP deadlock, OOM) kills only that child and not the whole eval run.
# The worker writes its result as JSON to a temp file; the parent reads it.
# ---------------------------------------------------------------------------

def _ocr_worker(file_path: str, out_file: str) -> None:
    """
    Entry point for the spawned OCR worker process.

    Sets thread-count env vars again (spawn copies the env, but just to be
    safe we re-apply them before importing torch via easyocr).
    """
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    reader = _get_ocr_reader()
    lines: list[str] = []

    if file_path.lower().endswith(".pdf"):
        try:
            from pdf2image import convert_from_path
            with tempfile.TemporaryDirectory() as tmpdir:
                pages = convert_from_path(file_path, dpi=200, output_folder=tmpdir)
                for img in pages:
                    lines.extend(reader.readtext(np.array(img), detail=0, paragraph=True))
        except Exception as exc:
            log.warning("[OCR] pdf2image failed (%s); direct image fallback ...", exc)
            try:
                lines = reader.readtext(file_path, detail=0, paragraph=True)
            except Exception as exc2:
                log.warning("[OCR] Direct fallback also failed (%s).", exc2)
    else:
        try:
            lines = reader.readtext(file_path, detail=0, paragraph=True)
        except Exception as exc:
            log.warning("[OCR] readtext failed (%s).", exc)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lines, f)


def _ocr_fallback(file_path: str, timeout: int = 120) -> str:
    """
    Run OCR in a spawned child process with a timeout.

    Returns the extracted text, or an empty string if the child crashes,
    times out, or produces no output.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_file = tmp.name

    try:
        # 'spawn' is required on macOS to avoid inheriting a half-initialised
        # OMP/torch state from the parent (which is what causes the deadlock).
        ctx = multiprocessing.get_context("spawn")
        p   = ctx.Process(target=_ocr_worker, args=(file_path, out_file))
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join(5)
            log.warning(
                "[OCR] Worker timed out after %ds for %s — skipping OCR.",
                timeout, file_path,
            )
            return ""

        if p.exitcode != 0:
            log.warning(
                "[OCR] Worker exited with code %s for %s — "
                "likely OMP deadlock / segfault, skipping OCR.",
                p.exitcode, file_path,
            )
            return ""

        if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
            log.warning("[OCR] Worker produced no output for %s.", file_path)
            return ""

        with open(out_file, "r", encoding="utf-8") as f:
            lines = json.load(f)
        return "\n".join(lines)

    finally:
        try:
            os.unlink(out_file)
        except OSError:
            pass


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
        log.info("[Stage 1] Native PDF text (%d chars).", len(native))
        return native, _alnum_ratio(native), True

    log.info("[Stage 1] Native text too short — EasyOCR fallback ...")
    ocr_text = _ocr_fallback(file_path)
    quality  = _alnum_ratio(ocr_text)
    log.info("[Stage 1] OCR done (%d chars, quality %.1f%%).", len(ocr_text), quality)
    return ocr_text, quality, False