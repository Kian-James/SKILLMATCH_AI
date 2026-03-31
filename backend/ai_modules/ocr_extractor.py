"""
ai_modules/ocr_extractor.py
Extracts raw text from uploaded resume files (PDF or image).
Uses Tesseract as primary OCR. EasyOCR is optional — imported only
if installed. On Render free tier, EasyOCR is not installed (too heavy),
so Tesseract is the sole OCR engine there.
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

# Configure Tesseract binary path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# EasyOCR is optional — only available when installed locally
try:
    import easyocr as _easyocr
    _EASYOCR_AVAILABLE = True
    logger.info("EasyOCR is available — will use as fallback for low-quality scans.")
except ImportError:
    _EASYOCR_AVAILABLE = False
    logger.info("EasyOCR not installed — Tesseract only mode (fine for Render).")

_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if not _EASYOCR_AVAILABLE:
        return None
    if _easyocr_reader is None:
        logger.info("Loading EasyOCR model (first time may be slow)...")
        _easyocr_reader = _easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def extract_text_from_image(image: Image.Image) -> str:
    """
    Run Tesseract OCR on a PIL Image.
    Falls back to EasyOCR only if it is installed AND Tesseract
    returns fewer than 50 characters.
    """
    tess_config = "--psm 6 --oem 3"
    try:
        text = pytesseract.image_to_string(image, config=tess_config).strip()
        if len(text) >= 50:
            return text
        logger.warning("Tesseract returned minimal text.")
    except Exception as e:
        logger.error(f"Tesseract failed: {e}")

    # EasyOCR fallback — only if available
    if _EASYOCR_AVAILABLE:
        logger.info("Falling back to EasyOCR...")
        reader = _get_easyocr_reader()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            results = reader.readtext(tmp.name, detail=0, paragraph=True)
        return " ".join(results).strip()

    # Tesseract-only mode: return whatever we got even if short
    logger.warning("EasyOCR not available — returning Tesseract output as-is.")
    try:
        return pytesseract.image_to_string(image, config="--psm 3 --oem 3").strip()
    except Exception as e:
        logger.error(f"Tesseract retry failed: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF.
    Tries direct text extraction first (fast), falls back to OCR for scanned PDFs.
    """
    # Attempt 1: Direct text extraction (fast for text-based PDFs)
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        combined = "\n".join(pages_text).strip()
        if len(combined) >= 100:
            logger.info("PDF text extracted directly (no OCR needed).")
            return combined
    except Exception as e:
        logger.warning(f"PyPDF2 direct extraction failed: {e}")

    # Attempt 2: OCR on rasterized pages
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