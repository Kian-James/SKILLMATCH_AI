"""
ai_modules/ocr_extractor.py
Extracts raw text from uploaded resume files (PDF or image).
Uses Tesseract as primary OCR, EasyOCR as fallback for low-quality scans.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytesseract
import easyocr
from PIL import Image
from pdf2image import convert_from_bytes
from loguru import logger

from config import TESSERACT_CMD

# Configure Tesseract binary path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# EasyOCR reader (lazy-init; supports English by default)
_easyocr_reader: easyocr.Reader | None = None


def _get_easyocr_reader() -> easyocr.Reader:
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Loading EasyOCR model (first time may be slow)...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def extract_text_from_image(image: Image.Image) -> str:
    """
    Run Tesseract OCR on a PIL Image.
    Falls back to EasyOCR if Tesseract returns fewer than 50 characters.
    """
    # Tesseract: use page segmentation mode 6 (assume single block of text)
    tess_config = "--psm 6 --oem 3"
    try:
        text = pytesseract.image_to_string(image, config=tess_config).strip()
        if len(text) >= 50:
            return text
        logger.warning("Tesseract returned minimal text; switching to EasyOCR.")
    except Exception as e:
        logger.error(f"Tesseract failed: {e}. Falling back to EasyOCR.")

    # EasyOCR fallback
    reader = _get_easyocr_reader()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        results = reader.readtext(tmp.name, detail=0, paragraph=True)
    return " ".join(results).strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Convert each PDF page to an image and apply OCR.
    Handles both text-based and scanned PDFs.
    First attempts PyPDF2 direct text extraction for speed; uses OCR as fallback.
    """
    # Attempt 1: Direct text extraction (fast for text-based PDFs)
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            pages_text.append(t.strip())
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
        page_text = extract_text_from_image(img)
        all_text.append(page_text)
    return "\n\n".join(all_text).strip()


def extract_text(file_bytes: bytes, content_type: str) -> str:
    """
    Unified entry point. Detects file type and routes to correct extractor.
    Supported: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP
    """
    content_type = content_type.lower()

    if "pdf" in content_type:
        return extract_text_from_pdf(file_bytes)

    if any(fmt in content_type for fmt in ["image", "png", "jpg", "jpeg", "tiff", "bmp", "webp"]):
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return extract_text_from_image(image)

    raise ValueError(f"Unsupported file type: {content_type}. Use PDF or image.")
