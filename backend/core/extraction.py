"""
core/extraction.py
==================
Stage 1 — Text acquisition (EasyOCR-only).

v40: removed PyMuPDF / native-PDF text extraction entirely. All inputs
(PDF or image) are routed through EasyOCR via _ocr_fallback(). PDFs are
rasterized to images with pdf2image before OCR.

Returns (text, ocr_quality_percent, is_native_pdf).
is_native_pdf is always False — retained in the return signature for
backward compatibility with core/skills.py and pipeline.py, which branch
on this flag.

Apple Silicon / macOS note
--------------------------
EasyOCR ships its own torch + libomp. On ARM64 Macs the OpenMP thread pool
can deadlock or segfault during tensor copy operations. We prevent this by
setting thread-count env vars to 1 BEFORE any torch/numpy import, and by
running each OCR call in a spawned subprocess so that a crash there cannot
kill the whole evaluation run.

Column-layout note
------------------
Two-column resume layouts (sidebar + main content) can cause EasyOCR to
interleave sidebar items (contact info, skills) with main-column items
(work experience bullets) when using paragraph=True. The fix uses detail=1
(bounding boxes) and _sort_boxes_reading_order(), which dynamically detects
a column split from the x-center distribution of OCR boxes and emits
left-column text first, then right-column text — with no hardcoded pixel
values or layout assumptions.

Fix log (v4, retained — applies to OCR output regardless of source)
---------------------------------------------------------------------
1. Aaron Villareal (narrow right-side skills sidebar)
   - Skills column sits in the far-right ~20% of the page.
   - The old 0.20–0.70 column-divider zone excluded it (midpoint > 0.70).
   - Fix: widen the right boundary of the search zone to 0.85.

2. Brendon Lucero (dark left sidebar, very narrow)
   - Left sidebar occupies ~30% of page width; split midpoint is near 0.30.
   - Lower minimum gap threshold from 8% to 5% of page width.

3. Camille Reyes (flat single-column, mixed bullets)
   - Non-standard bullet symbols (○, ✔, ➦, !) stripped via
     _STANDALONE_BULLET_RE / _INLINE_BULLET_RE.

4-5. Two-column / centered layouts — inline bullet stripping cleans
   ➦ / ○ / ✔ characters from work bullet lines.

6. Multi-page PDFs — pages joined with "\n\n" so section parsers see a
   clear break between the last line of one page and the next.

7. Standalone decorative bullet blocks stripped via _STANDALONE_BULLET_RE.

8. Label-only sidebar interleaving (Adrian Santos)
   - _is_label_only_sidebar() detects left columns that are ONLY short
     section labels ("Skills", "Work Experience", "—") with no real content.
   - When True, _interleave_label_sidebar() pairs each label with the
     right-column content in the same vertical band.

v40 note on column rulings
---------------------------
Previously PyMuPDF supplied authoritative column-split rulings (in PDF
point-space) that were scaled into OCR pixel-space. With PyMuPDF removed,
_sort_boxes_reading_order() always self-detects the column split directly
from EasyOCR bounding boxes (the v5 "Kevin Garcia" fallback path is now the
only path, for every input).
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
import re
import tempfile

import numpy as np

from config import OCR_LANGUAGES

log = logging.getLogger(__name__)

_ocr_reader = None

# ---------------------------------------------------------------------------
# Bullet / decoration cleanup
# ---------------------------------------------------------------------------

_STANDALONE_BULLET_RE = re.compile(
    r"(?m)^[ \t]*[•◦▪\-\*○➦✔✓!@✉][ \t]*$"
)

_INLINE_BULLET_RE = re.compile(
    r"^[ \t]*[•◦▪○➦✔✓!\-\*][ \t]+"
)

_CONTENT_LINE_RE = re.compile(
    r"(\d{4})"
    r"|@"
    r"|\(\d{2,}\)"
    r"|(?:\w+\s+){4,}\w+"
)


def _clean_line(line: str) -> str:
    return _INLINE_BULLET_RE.sub("", line)


def _clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if _STANDALONE_BULLET_RE.fullmatch(line.strip()) or (
                line.strip() in {"•", "◦", "▪", "○", "➦", "✔", "✓", "!", "@", "✉"}
        ):
            continue
        cleaned.append(_clean_line(line))

    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


# ---------------------------------------------------------------------------
# Column-split detection
# ---------------------------------------------------------------------------

def _detect_column_split(x_centers: list[float]) -> float | None:
    """
    Dynamically detect the x coordinate that divides a two-column layout.

    * Search zone: 0.20–0.85 of page width.
    * Minimum qualifying gap: 5% of page width.
    """
    if len(x_centers) < 4:
        return None

    xs         = sorted(x_centers)
    page_width = max(xs)
    if page_width == 0:
        return None

    best_gap   = 0.0
    best_split: float | None = None

    for i in range(len(xs) - 1):
        gap = xs[i + 1] - xs[i]
        mid = (xs[i] + xs[i + 1]) / 2.0
        if 0.20 * page_width < mid < 0.85 * page_width and gap > best_gap:
            best_gap   = gap
            best_split = mid

    if best_split is not None and best_gap > 0.05 * page_width:
        return best_split
    return None


# ---------------------------------------------------------------------------
# Label-only sidebar detection
# ---------------------------------------------------------------------------

def _is_label_only_sidebar(left_blocks: list[tuple[float, float, str]]) -> bool:
    """
    Return True when every block in the left column looks like a short section
    label ("Work Experience", "Skills", "Education", "—") rather than real
    content (contact details, skill lists, multi-line bullets, dates).
    """
    if len(left_blocks) < 2:
        return False

    for (_y, _x, text) in left_blocks:
        stripped = text.strip()
        if stripped in {"—", "–", "−", ""}:
            continue
        word_count = len(stripped.split())
        if word_count > 4:
            return False
        if _CONTENT_LINE_RE.search(stripped):
            return False

    return True


# ---------------------------------------------------------------------------
# Y-zone interleaving (label-only sidebar path only)
# ---------------------------------------------------------------------------

def _interleave_label_sidebar(
        left: list[tuple[float, float, str]],
        right: list[tuple[float, float, str]],
) -> list[str]:
    """
    Pair each left-column label with the right-column blocks in the same
    vertical band, then emit label → right-band content in order.

    Only called when _is_label_only_sidebar() is True.
    """
    _DECORATIVE = {"—", "–", "−", ""}

    left_sorted  = sorted(left,  key=lambda e: e[0])
    right_sorted = sorted(right, key=lambda e: e[0])

    real_labels = [(y, x, t) for y, x, t in left_sorted
                   if t.strip() not in _DECORATIVE]

    first_y = real_labels[0][0] if real_labels else 0.0
    result: list[str] = [e[2] for e in right_sorted if e[0] < first_y]

    _ZONE_TOLERANCE = 20.0

    for i, (ly, _lx, label) in enumerate(real_labels):
        result.append(label)
        zone_end = real_labels[i + 1][0] if i + 1 < len(real_labels) else float("inf")
        zone_start = ly - _ZONE_TOLERANCE
        result.extend(e[2] for e in right_sorted if zone_start <= e[0] < zone_end
                      and e[2] not in result)

    return result


# ---------------------------------------------------------------------------
# Reading-order sort (EasyOCR path)
# ---------------------------------------------------------------------------

def _sort_boxes_reading_order(raw_results: list) -> list[str]:
    """
    Convert EasyOCR detail=1 output → reading-order list of text strings.

    v40: pdf_layout parameter removed (PyMuPDF gone). Column split is always
    self-detected from EasyOCR bounding boxes — this is the only path now,
    for both PDFs (rasterized) and images.
    """
    if not raw_results:
        return []

    boxes: list[tuple[float, float, str]] = []
    for item in raw_results:
        bbox, text, _conf = item
        x_center = (bbox[0][0] + bbox[2][0]) / 2.0
        y_center = (bbox[0][1] + bbox[2][1]) / 2.0
        if text.strip():
            boxes.append((y_center, x_center, text.strip()))

    if not boxes:
        return []

    x_centers = [b[1] for b in boxes]
    col_split = _detect_column_split(x_centers)
    is_label_only = False

    if col_split is not None:
        left_blocks = [(y, x, t) for y, x, t in boxes if x <= col_split]
        is_label_only = _is_label_only_sidebar(left_blocks)
        log.debug(
            "[OCR] Self-detected col_split=%.1f is_label_only=%s",
            col_split, is_label_only,
        )

    if col_split is not None:
        left  = sorted([b for b in boxes if b[1] <= col_split], key=lambda b: b[0])
        right = sorted([b for b in boxes if b[1]  > col_split], key=lambda b: b[0])

        if is_label_only:
            log.debug("[OCR] Applying Y-zone interleave (label-only sidebar).")
            return _interleave_label_sidebar(left, right)
        else:
            return [b[2] for b in (left + right)]
    else:
        log.debug("[OCR] Single-column layout — plain top-to-bottom order.")
        ordered = sorted(boxes, key=lambda b: b[0])
        return [b[2] for b in ordered]


# ---------------------------------------------------------------------------
# OCR subprocess worker
# ---------------------------------------------------------------------------

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
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


def _ocr_worker(file_path: str, out_file: str) -> None:
    """
    Entry point for the spawned OCR worker process.

    v40: layout_file / pdf_layouts removed — _sort_boxes_reading_order()
    always self-detects column layout from OCR boxes.
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
                    raw = reader.readtext(np.array(img), detail=1, paragraph=False)
                    lines.extend(_sort_boxes_reading_order(raw))
                    # Page boundary marker so downstream section parsers see
                    # a clear break (preserves old "\n\n" page-join behavior).
                    lines.append("")
        except Exception as exc:
            log.warning("[OCR] pdf2image failed (%s); direct image fallback ...", exc)
            try:
                raw   = reader.readtext(file_path, detail=1, paragraph=False)
                lines = _sort_boxes_reading_order(raw)
            except Exception as exc2:
                log.warning("[OCR] Direct fallback also failed (%s).", exc2)
    else:
        try:
            raw   = reader.readtext(file_path, detail=1, paragraph=False)
            lines = _sort_boxes_reading_order(raw)
        except Exception as exc:
            log.warning("[OCR] readtext failed (%s).", exc)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lines, f)


def _ocr_fallback(file_path: str, timeout: int = 120) -> str:
    """
    Run OCR in a spawned child process with a timeout.

    v40: layout_file removed along with PyMuPDF.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_file = tmp.name

    try:
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
        return _clean_text("\n\n".join(lines))

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
    Stage 1: EasyOCR for everything (PDF or image).

    v40: PyMuPDF native-text path removed. PDFs are rasterized via
    pdf2image and OCR'd page-by-page like images.

    Returns
    -------
    (text, ocr_quality_percent, is_native_pdf)
    is_native_pdf is always False.
    """
    log.info("[Stage 1] EasyOCR extraction (v40 — OCR-only) ...")
    ocr_text = _ocr_fallback(file_path)
    quality  = _alnum_ratio(ocr_text)
    log.info("[Stage 1] OCR done (%d chars, quality %.1f%%).", len(ocr_text), quality)
    return ocr_text, quality, False