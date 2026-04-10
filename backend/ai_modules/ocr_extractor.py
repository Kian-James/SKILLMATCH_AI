"""
ai_modules/ocr_extractor.py
Extracts raw text from uploaded resume files (PDF or image).

FIX (two-column PDF):
  PyPDF2 direct extraction on two-column PDF layouts merges both columns
  into a single text stream, causing:
    1. The SKILLS section header regex to fail (header immediately followed
       by interleaved column text, no clean newline boundary)
    2. Certifications from left and right columns to concatenate into one
       string (e.g. "Cisco ITF+FreeCodeCamp – Responsive Web Design")
    3. Project description text (right-side column) bleeding into what
       appears to be the skills section

  Fix: pdfplumber with bbox-based column splitting.
    - Detect if the page has a multi-column layout by checking whether
      text characters are clustered in two distinct x-ranges.
    - If two-column: crop left half and right half separately, extract
      each, then join them with clear section breaks.
    - If single-column: use pdfplumber's normal extract_text() which is
      more robust than PyPDF2 for most PDF types.
    - Fall back to Tesseract OCR if pdfplumber yields < 100 chars.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from loguru import logger

from config import TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed — falling back to PyPDF2 for direct extraction.")

# EasyOCR optional
try:
    import easyocr as _easyocr
    _EASYOCR_AVAILABLE = True
    logger.info("EasyOCR is available — will use as fallback for low-quality scans.")
except ImportError:
    _EASYOCR_AVAILABLE = False

_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if not _EASYOCR_AVAILABLE:
        return None
    if _easyocr_reader is None:
        logger.info("Loading EasyOCR model (first time may be slow)...")
        _easyocr_reader = _easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


# ─── Column detection ─────────────────────────────────────────────────────────

def _is_two_column(page) -> bool:
    """
    Detect whether a pdfplumber page has a two-column layout.
    Checks if text characters cluster into two distinct x-ranges
    with a clear gap between them (typical of two-column resumes).
    """
    try:
        chars = page.chars
        if not chars:
            return False
        xs = [c["x0"] for c in chars]
        page_mid = (page.bbox[0] + page.bbox[2]) / 2
        left_xs  = [x for x in xs if x < page_mid]
        right_xs = [x for x in xs if x >= page_mid]
        # Two-column if both halves have substantial text
        return len(left_xs) > 50 and len(right_xs) > 50
    except Exception:
        return False


def _extract_pdf_text_pdfplumber(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF using pdfplumber.
    For two-column layouts: crops left and right halves separately
    so section headers and their content stay together.
    For single-column: uses standard extract_text().
    """
    pages_text: list[str] = []

    with _pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            x0, top, x1, bottom = page.bbox

            if _is_two_column(page):
                logger.debug(f"Page {page_num + 1}: two-column layout detected — splitting columns")
                mid = x0 + (x1 - x0) * 0.55  # 55% split handles asymmetric layouts

                left_text  = page.crop((x0, top, mid, bottom)).extract_text() or ""
                right_text = page.crop((mid, top, x1, bottom)).extract_text() or ""

                # Join left column full text then right column full text
                # This keeps section headers (SKILLS, CERTIFICATIONS) with
                # their content instead of interleaving the two columns
                combined = left_text.strip()
                if right_text.strip():
                    combined += "\n\n" + right_text.strip()

                pages_text.append(combined)
                logger.debug(
                    f"  Left: {len(left_text)} chars | Right: {len(right_text)} chars"
                )
            else:
                logger.debug(f"Page {page_num + 1}: single-column layout")
                text = page.extract_text() or ""
                pages_text.append(text)

    return "\n\n".join(pages_text).strip()


# ─── OCR helpers ──────────────────────────────────────────────────────────────

def extract_text_from_image(image: Image.Image) -> str:
    """
    Run Tesseract OCR on a PIL Image.
    Falls back to EasyOCR only if installed AND Tesseract returns < 50 chars.
    """
    tess_config = "--psm 6 --oem 3"
    try:
        text = pytesseract.image_to_string(image, config=tess_config).strip()
        if len(text) >= 50:
            return text
        logger.warning("Tesseract returned minimal text.")
    except Exception as e:
        logger.error(f"Tesseract failed: {e}")

    if _EASYOCR_AVAILABLE:
        logger.info("Falling back to EasyOCR...")
        reader = _get_easyocr_reader()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            results = reader.readtext(tmp.name, detail=0, paragraph=True)
        return " ".join(results).strip()

    logger.warning("EasyOCR not available — returning Tesseract output as-is.")
    try:
        return pytesseract.image_to_string(image, config="--psm 3 --oem 3").strip()
    except Exception as e:
        logger.error(f"Tesseract retry failed: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF.

    Priority order:
      1. pdfplumber with column-aware extraction (handles two-column resumes)
      2. PyPDF2 direct extraction (fallback if pdfplumber not installed)
      3. Tesseract OCR on rasterized pages (scanned/image-only PDFs)
    """
    # ── Attempt 1: pdfplumber (column-aware) ──────────────────────────────────
    if _PDFPLUMBER_AVAILABLE:
        try:
            text = _extract_pdf_text_pdfplumber(pdf_bytes)
            if len(text) >= 100:
                logger.info(
                    f"PDF text extracted via pdfplumber ({len(text)} chars)."
                )
                return text
            logger.warning(
                f"pdfplumber returned only {len(text)} chars — trying OCR."
            )
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e} — trying PyPDF2.")

    # ── Attempt 2: PyPDF2 (fallback, no column awareness) ────────────────────
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        combined = "\n".join(pages_text).strip()
        if len(combined) >= 100:
            logger.info("PDF text extracted via PyPDF2 (no OCR needed).")
            return combined
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")

    # ── Attempt 3: OCR on rasterized pages (scanned PDF) ─────────────────────
    logger.info("Rasterizing PDF pages for OCR...")
    images = convert_from_bytes(pdf_bytes, dpi=300)
    all_text: list[str] = []
    for i, img in enumerate(images):
        logger.debug(f"  OCR on page {i + 1}/{len(images)}")
        all_text.append(extract_text_from_image(img))
    return "\n\n".join(all_text).strip()


def extract_text(file_bytes: bytes, content_type: str) -> str:
    """
    Unified entry point. Routes to PDF or image extractor based on content type.
    """
    content_type = content_type.lower()

    if "pdf" in content_type:
        return extract_text_from_pdf(file_bytes)

    if any(fmt in content_type for fmt in ["image", "png", "jpg", "jpeg", "tiff", "bmp", "webp"]):
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return extract_text_from_image(image)

    raise ValueError(f"Unsupported file type: {content_type}. Use PDF or image.")