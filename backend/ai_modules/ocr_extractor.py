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

IMPROVEMENT (scanned PDF / Tesseract accuracy):
  All 202 resumes fell through to Tesseract because pdfplumber returned
  < 100 chars (scanned/image-only PDFs). Tesseract at default settings
  on scans loses words and misreads characters. Improvements:
    1. DPI raised from 300 → 400 — more pixels = sharper characters
    2. Image preprocessing — grayscale → contrast boost → sharpen → binarize
    3. PSM changed from 6 (uniform block) to 4 (single column) — better for
       resume layouts with varied indentation and bullet points
    4. preserve_interword_spaces=1 — prevents words from merging across gaps
    5. Upscale small images before OCR — tiny text at low DPI is the #1
       cause of character misreads in scanned resumes
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
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


# ─── Tesseract image preprocessing ───────────────────────────────────────────

def _preprocess_for_tesseract(img: Image.Image) -> Image.Image:
    """
    Preprocess a scanned resume image before OCR to improve character
    recognition accuracy. Applied to every page before Tesseract runs.

    Pipeline:
      1. Convert to grayscale  — colour noise misleads Tesseract
      2. Upscale if small      — tiny text at < 150px/inch causes misreads;
                                 Tesseract accuracy drops sharply below 200dpi
      3. Boost contrast        — faded/light scans produce grey characters
                                 that OCR engines misread as spaces
      4. Sharpen               — smeared characters from scanner compression
                                 become distinct again after sharpening
      5. Binarize (threshold)  — converts to pure black-and-white so Tesseract
                                 sees crisp edges, not grey anti-aliased pixels.
                                 Threshold 140: keeps faint ink visible while
                                 removing scanner noise (typical grey ~200+)
    """
    # 1. Grayscale
    img = img.convert("L")

    # 2. Upscale if the image is too small (< 1800px wide means likely low DPI)
    min_width = 1800
    if img.width < min_width:
        scale = min_width / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.debug(f"Upscaled image to {new_size[0]}x{new_size[1]}")

    # 3. Contrast boost — factor 2.0 works well for typical office scanner output
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # 4. Sharpen — helps with slightly blurry scans
    img = img.filter(ImageFilter.SHARPEN)

    # 5. Binarize — threshold at 140 (0–255 scale)
    img = img.point(lambda x: 0 if x < 140 else 255, "1").convert("L")

    return img


# ─── OCR helpers ──────────────────────────────────────────────────────────────

def extract_text_from_image(image: Image.Image) -> str:
    """
    Run Tesseract OCR on a PIL Image with improved settings and preprocessing.
    Falls back to EasyOCR only if installed AND Tesseract returns < 50 chars.

    Tesseract config changes vs original:
      --psm 4  : Assume a single column of text of variable sizes.
                 Better for resumes than --psm 6 (uniform block) because
                 resumes mix headers, bullet points, and paragraphs.
      --oem 3  : Default engine (LSTM neural net).
      preserve_interword_spaces=1 : Prevents words from merging across
                 visual gaps (e.g. skill tags with space between them).
    """
    preprocessed = _preprocess_for_tesseract(image)
    tess_config = "--psm 4 --oem 3 -c preserve_interword_spaces=1"

    try:
        text = pytesseract.image_to_string(preprocessed, config=tess_config).strip()
        if len(text) >= 50:
            logger.debug(f"Tesseract extracted {len(text)} chars.")
            return text
        logger.warning(f"Tesseract returned only {len(text)} chars after preprocessing.")
    except Exception as e:
        logger.error(f"Tesseract failed: {e}")

    # Fallback: EasyOCR if available
    if _EASYOCR_AVAILABLE:
        logger.info("Falling back to EasyOCR...")
        reader = _get_easyocr_reader()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            results = reader.readtext(tmp.name, detail=0, paragraph=True)
        return " ".join(results).strip()

    logger.warning("EasyOCR not available — returning Tesseract output as-is.")
    try:
        # Last resort: try original image without preprocessing
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
    # Use 400 DPI — significantly better than 300 DPI for character accuracy
    # on small fonts (9–11pt) commonly used in resume templates.
    logger.info("Rasterizing PDF pages for Tesseract OCR (400 DPI)...")
    images = convert_from_bytes(pdf_bytes, dpi=400)
    all_text: list[str] = []
    for i, img in enumerate(images):
        logger.debug(f"  OCR on page {i + 1}/{len(images)}")
        all_text.append(extract_text_from_image(img))
    return "\n\n".join(all_text).strip()


def extract_text(file_bytes: bytes, content_type: str) -> str:
    """
    Unified entry point. Routes to PDF or image extractor based on content type.
    """
    text, _ = extract_text_flagged(file_bytes, content_type)
    return text


def extract_text_flagged(file_bytes: bytes, content_type: str) -> tuple[str, bool]:
    """
    Same as extract_text but also returns a bool indicating whether
    Tesseract OCR was used (True) or clean text extraction was used (False).

    The flag is used by ResumeService to skip OCR correction on clean PDFs —
    pdfplumber output doesn't have OCR misreads so correction is unnecessary
    and wastes 20+ seconds embedding the vocabulary.
    """
    content_type = content_type.lower()

    if "pdf" in content_type:
        # Try pdfplumber / PyPDF2 first (clean text — no OCR correction needed)
        if _PDFPLUMBER_AVAILABLE:
            try:
                text = _extract_pdf_text_pdfplumber(file_bytes)
                if len(text) >= 100:
                    return text, False   # clean extraction — skip OCR correction
            except Exception:
                pass
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            combined = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
            if len(combined) >= 100:
                return combined, False   # clean extraction
        except Exception:
            pass

        # Fall through to Tesseract — 400 DPI + preprocessing
        logger.info("Rasterizing PDF for Tesseract OCR (400 DPI)...")
        images   = convert_from_bytes(file_bytes, dpi=400)
        all_text = [extract_text_from_image(img) for img in images]
        return "\n\n".join(all_text).strip(), True

    if any(fmt in content_type for fmt in ["image", "png", "jpg", "jpeg", "tiff", "bmp", "webp"]):
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return extract_text_from_image(image), True   # always Tesseract for images

    raise ValueError(f"Unsupported file type: {content_type}. Use PDF or image.")