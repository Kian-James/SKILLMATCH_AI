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

Column-layout note
------------------
Two-column resume layouts (sidebar + main content) were causing EasyOCR to
interleave sidebar items (contact info, skills) with main-column items
(work experience bullets) when using paragraph=True.  The fix uses detail=1
(bounding boxes) and _sort_boxes_reading_order(), which dynamically detects
a column split from the x-center distribution of text boxes and emits
left-column text first, then right-column text — with no hardcoded pixel
values or layout assumptions.

Fix log (v4)
------------
Resumes tested and issues resolved:

1. Aaron Villareal (narrow right-side skills sidebar)
   - Skills column sits in the far-right ~20% of the page.
   - The old 0.20–0.70 column-divider zone excluded it (midpoint > 0.70).
   - Fix: widen the right boundary of the search zone to 0.85.

2. Brendon Lucero (dark left sidebar, very narrow)
   - Left sidebar occupies ~30% of page width; split midpoint is near 0.30.
   - Was already within range, but minimum gap threshold (8%) could miss it
     when the gap between sidebar and main column is small in absolute terms.
   - Fix: lower minimum gap threshold from 8% to 5% of page width.

3. Camille Reyes (flat single-column, mixed bullets)
   - No column split; skills and work bullets share the same x-band.
   - Issue was non-standard bullet symbols (○, ✔, ➦, !) not being stripped,
     producing garbled lines like "○ :" at the start of work entries.
   - Fix: extend _STANDALONE_BULLET_RE to cover all observed symbols, and
     add _INLINE_BULLET_RE to strip leading bullet chars from content lines.

4. Cedric Tolentino / Daniel Cortez (two-column, centered layout)
   - Work bullets used ➦ / ○ which were not stripped, cluttering text.
   - Fix: same inline bullet stripping as above.

5. Angelica Mae Soriano (two-column, right-heavy)
   - Single work entry; layout detected correctly. No structural fix needed.
   - Inline bullet stripping cleans ✔ characters from work bullet lines.

6. Adrian Santos — Eng resume (two-page PDF)
   - Page 2 contains only the education continuation ("Structural Engineering /
     University of Santo Tomas / 2015 - 2019").
   - The old code joined pages with "\n".join() which lost the page boundary.
   - Fix: join pages with "\n\n" so section parsers see a clear break between
     the last line of page 1 and the continuation on page 2.
   - Also strips lone bullet/icon lines (!, ✓, ✔, ○, ➦) that appear as
     standalone artifact lines in this resume's PDF encoding.

7. All resumes — standalone decorative bullets
   - Several templates emit a block of bare bullet characters (•, ○, ➦, !)
     as separate text blocks, separate from the actual skill/work text.
   - Extended _STANDALONE_BULLET_RE to match all observed symbols.

8. Adrian Santos — label-only sidebar interleaving (v4 fix)
   - Adrian's left column contains ONLY short section labels ("Work Experience",
     "Skills", "Education", "—") with no actual content.
   - The standard left+right dump put all labels first, then all right-column
     content after — so "Skills" appeared at the top but ETABS/STAAD.Pro/etc.
     appeared at the very bottom, detached from the label.
   - Fix: _is_label_only_sidebar() detects this pattern (all left-column blocks
     are short, ≤4 words, and none contain contact/skill/date content).
   - Only when this is True does _native_pdf_text() use Y-zone interleaving to
     pair each label with its right-column content.
   - All other resume layouts continue to use the original left+right per-page
     logic, so no regressions on Aaron Villareal, Brendon Lucero, etc.

9. Kevin Garcia — PyMuPDF rulings reused in OCR path (v5 fix)
   - Kevin's resume falls to EasyOCR (native char count below MIN_NATIVE_CHARS).
   - Previously the OCR path ran _detect_column_split() independently on OCR
     bounding boxes, which could produce a different (or no) split vs PyMuPDF.
   - Fix: _extract_pdf_layout() runs PyMuPDF on the PDF before OCR to extract
     the authoritative col_split and is_label_only flag per page. These rulings
     are passed into _sort_boxes_reading_order() so the OCR boxes are assigned
     to left/right columns using the same x-coordinate PyMuPDF found, and
     _interleave_label_sidebar() is applied when appropriate.
   - For image-only uploads (no PDF) or when PyMuPDF finds no split, the OCR
     path falls back to its own _detect_column_split() as before.
   - Result: "Skills" label and skill bullets are correctly adjacent in the
     OCR output, so _SKILL_HEADING_RE finds them and extracts the full list.
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

import fitz          # PyMuPDF
import numpy as np

from config import OCR_LANGUAGES, MIN_NATIVE_CHARS

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

    # Strip decorative dividers from the label list before building Y zones.
    # "—" lines have their own Y position and would otherwise steal right-column
    # content that belongs to the next real section label.
    # Example (Adrian Santos): "—" at y=580 sits just above "Skills" at y=590.
    # Without filtering, ETABS (right col, y=585) falls into the "—" zone and
    # appears before the Skills heading instead of after it.
    real_labels = [(y, x, t) for y, x, t in left_sorted
                   if t.strip() not in _DECORATIVE]

    first_y = real_labels[0][0] if real_labels else 0.0
    result: list[str] = [e[2] for e in right_sorted if e[0] < first_y]

    # Tolerance: right-col items that appear slightly above a section label
    # (within 20 pts / ~7mm) are considered part of that label's zone.
    # This handles cases like ETABS sitting at y=585 when Skills label is y=590
    # — visually it is the first skill, but its Y is just above the label's Y.
    _ZONE_TOLERANCE = 20.0

    for i, (ly, _lx, label) in enumerate(real_labels):
        result.append(label)
        zone_end = real_labels[i + 1][0] if i + 1 < len(real_labels) else float("inf")
        zone_start = ly - _ZONE_TOLERANCE
        result.extend(e[2] for e in right_sorted if zone_start <= e[0] < zone_end
                      and e[2] not in result)

    return result


# ---------------------------------------------------------------------------
# PyMuPDF layout extraction (v5) — rulings for OCR path
# ---------------------------------------------------------------------------

def _extract_pdf_layout(file_path: str) -> list[dict]:
    """
    Run PyMuPDF on a PDF to extract the authoritative column layout per page.

    Returns a list of per-page dicts:
      {
        "col_split":    float | None,   # x-coordinate of column divider
        "is_label_only": bool,          # True if left col is label-only sidebar
        "page_width":   float,          # max x-center across all blocks
      }

    Used by the OCR path so it can reuse PyMuPDF's column rulings instead of
    independently re-detecting them from noisy OCR bounding boxes.

    If the file is not a PDF or PyMuPDF raises an exception, returns [].
    """
    if not file_path.lower().endswith(".pdf"):
        return []

    try:
        doc    = fitz.open(file_path)
        pages: list[dict] = []

        for page in doc:
            blocks  = page.get_text("blocks")
            entries: list[tuple[float, float, str]] = []

            for b in blocks:
                x0, y0, x1, y1, text, _block_no, block_type = b
                if block_type != 0:
                    continue
                text = text.strip()
                if not text:
                    continue
                xc = (x0 + x1) / 2.0
                entries.append((y0, xc, text))

            if not entries:
                pages.append({"col_split": None, "is_label_only": False, "page_width": 0.0})
                continue

            x_centers  = [e[1] for e in entries]
            col_split  = _detect_column_split(x_centers)
            page_width = max(x_centers)

            is_label_only = False
            if col_split is not None:
                left_blocks = [(y, x, t) for y, x, t in entries if x <= col_split]
                is_label_only = _is_label_only_sidebar(left_blocks)

            pages.append({
                "col_split":    col_split,
                "is_label_only": is_label_only,
                "page_width":   page_width,
            })

            log.debug(
                "[PDFLayout] page col_split=%.1f is_label_only=%s",
                col_split or 0.0, is_label_only,
                )

        doc.close()
        return pages

    except Exception as exc:
        log.warning("[PDFLayout] PyMuPDF layout extraction failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Reading-order sort (EasyOCR path)
# ---------------------------------------------------------------------------

def _sort_boxes_reading_order(
        raw_results: list,
        pdf_layout:  dict | None = None,
) -> list[str]:
    """
    Convert EasyOCR detail=1 output → reading-order list of text strings.

    v5: accepts an optional pdf_layout dict (one entry from _extract_pdf_layout)
    so the OCR path reuses PyMuPDF's authoritative column rulings instead of
    independently re-detecting the column split from noisy OCR boxes.

    When pdf_layout provides a col_split:
      - OCR boxes are assigned left/right using the same x-coordinate PyMuPDF found.
      - If is_label_only is True, _interleave_label_sidebar() is applied so each
        section label (Skills, Work Experience, Education) appears directly above
        its content in the output.

    When pdf_layout is None or provides no col_split, the original
    _detect_column_split() fallback runs on the OCR boxes as before.
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

    # ── Determine column split ────────────────────────────────────────────────
    # Priority: PyMuPDF ruling > OCR self-detection
    col_split:    float | None = None
    is_label_only: bool        = False

    if pdf_layout and pdf_layout.get("col_split") is not None:
        # Scale PyMuPDF's point-space col_split to OCR pixel-space.
        # PyMuPDF x-centers are in PDF points; OCR x-centers are in pixels
        # at whatever DPI pdf2image used (200 DPI -> scale = 200/72 ≈ 2.78).
        # We derive the scale factor from the ratio of max x-centers so no
        # hardcoded DPI value is needed.
        pdf_max_x = pdf_layout.get("page_width", 0.0)
        ocr_max_x = max(b[1] for b in boxes) if boxes else 0.0
        if pdf_max_x > 0 and ocr_max_x > 0:
            scale     = ocr_max_x / pdf_max_x
            col_split = pdf_layout["col_split"] * scale
            is_label_only = pdf_layout.get("is_label_only", False)
            log.debug(
                "[OCR] Using PyMuPDF ruling: col_split=%.1f (scale=%.3f) is_label_only=%s",
                col_split, scale, is_label_only,
            )
        else:
            log.debug("[OCR] PyMuPDF page_width=0 — falling back to OCR self-detection.")

    if col_split is None:
        # Fallback: detect split from OCR boxes directly.
        x_centers = [b[1] for b in boxes]
        col_split  = _detect_column_split(x_centers)
        if col_split is not None:
            left_blocks = [(y, x, t) for y, x, t in boxes if x <= col_split]
            is_label_only = _is_label_only_sidebar(left_blocks)
            log.debug(
                "[OCR] Self-detected col_split=%.1f is_label_only=%s",
                col_split, is_label_only,
            )

    # ── Sort boxes into reading order ─────────────────────────────────────────
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
# Native PDF extraction
# ---------------------------------------------------------------------------

def _native_pdf_text(file_path: str) -> str:
    """
    Extract text from a native (non-scanned) PDF using PyMuPDF.
    """
    doc = fitz.open(file_path)
    pages_text: list[str] = []

    for page in doc:
        blocks = page.get_text("blocks")

        entries: list[tuple[float, float, str]] = []
        for b in blocks:
            x0, y0, x1, y1, text, _block_no, block_type = b
            if block_type != 0:
                continue
            text = text.strip()
            if not text:
                continue
            xc = (x0 + x1) / 2.0
            entries.append((y0, xc, text))

        x_centers = [e[1] for e in entries]
        col_split  = _detect_column_split(x_centers)

        if col_split is not None:
            log.debug("[NativePDF] Two-column layout — split at x=%.1f", col_split)
            left  = sorted([e for e in entries if e[1] <= col_split], key=lambda e: e[0])
            right = sorted([e for e in entries if e[1]  > col_split], key=lambda e: e[0])

            if _is_label_only_sidebar(left):
                log.debug("[NativePDF] Label-only sidebar detected — using interleave.")
                page_lines = _interleave_label_sidebar(left, right)
            else:
                page_lines = [e[2] for e in (left + right)]
        else:
            log.debug("[NativePDF] Single-column layout.")
            page_lines = [e[2] for e in sorted(entries, key=lambda e: e[0])]

        pages_text.append("\n".join(page_lines))

    doc.close()
    return _clean_text("\n\n".join(pages_text))


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


def _ocr_worker(file_path: str, out_file: str, layout_file: str) -> None:
    """
    Entry point for the spawned OCR worker process.

    v5: reads per-page PyMuPDF layout rulings from layout_file and passes
    each page's ruling into _sort_boxes_reading_order() so OCR boxes are
    sorted using the same column split PyMuPDF found.
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

    # Load PyMuPDF layout rulings written by the parent process.
    pdf_layouts: list[dict] = []
    try:
        if os.path.exists(layout_file):
            with open(layout_file, "r", encoding="utf-8") as f:
                pdf_layouts = json.load(f)
    except Exception as exc:
        log.warning("[OCR] Could not read layout file: %s", exc)

    reader = _get_ocr_reader()
    lines: list[str] = []

    if file_path.lower().endswith(".pdf"):
        try:
            from pdf2image import convert_from_path
            with tempfile.TemporaryDirectory() as tmpdir:
                pages = convert_from_path(file_path, dpi=200, output_folder=tmpdir)
                for page_idx, img in enumerate(pages):
                    # Use PyMuPDF ruling for this page if available.
                    layout = pdf_layouts[page_idx] if page_idx < len(pdf_layouts) else None
                    raw    = reader.readtext(np.array(img), detail=1, paragraph=False)
                    lines.extend(_sort_boxes_reading_order(raw, pdf_layout=layout))
        except Exception as exc:
            log.warning("[OCR] pdf2image failed (%s); direct image fallback ...", exc)
            try:
                raw   = reader.readtext(file_path, detail=1, paragraph=False)
                # No page index available in direct fallback — use first layout.
                layout = pdf_layouts[0] if pdf_layouts else None
                lines  = _sort_boxes_reading_order(raw, pdf_layout=layout)
            except Exception as exc2:
                log.warning("[OCR] Direct fallback also failed (%s).", exc2)
    else:
        try:
            raw   = reader.readtext(file_path, detail=1, paragraph=False)
            lines = _sort_boxes_reading_order(raw, pdf_layout=None)
        except Exception as exc:
            log.warning("[OCR] readtext failed (%s).", exc)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lines, f)


def _ocr_fallback(file_path: str, timeout: int = 120) -> str:
    """
    Run OCR in a spawned child process with a timeout.

    v5: extracts PyMuPDF layout rulings in the parent process and writes them
    to a temp file that the OCR worker reads. This avoids re-importing fitz
    inside the spawned process and keeps the worker lightweight.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_file = tmp.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as lf:
        layout_file = lf.name
        # Extract PyMuPDF rulings in the parent (before spawning).
        pdf_layouts = _extract_pdf_layout(file_path)
        json.dump(pdf_layouts, lf)
        log.debug("[OCR] Wrote %d page layout(s) to %s", len(pdf_layouts), layout_file)

    try:
        ctx = multiprocessing.get_context("spawn")
        p   = ctx.Process(target=_ocr_worker, args=(file_path, out_file, layout_file))
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
        for path in (out_file, layout_file):
            try:
                os.unlink(path)
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