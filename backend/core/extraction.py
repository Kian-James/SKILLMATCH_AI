"""
core/extraction.py
==================
Stage 1 — Text acquisition (EasyOCR-only).

v41: Extended layout detection to handle all observed resume formats:

Layout inventory (from evaluated resumes)
------------------------------------------
1.  Aaron Villareal       – single-column, narrow right-side skills sidebar
2.  Adrian Velasco        – two-column with colored left sidebar (contact + skills)
3.  Angelica Mae Soriano  – two-column, centered header, left sidebar with CONTACT/
                            EDUCATION/SKILLS, right side has SUMMARY + WORK EXPERIENCE
4.  Brendon Lucero        – dark narrow left sidebar (~30% width)
5.  Camille Reyes         – flat single-column, mixed non-standard bullets (○ ✔ ➦ !)
6.  Cedric Tolentino      – clean single-column, professional two-tone header
7.  Daniel Cortez         – full-width dark navy header banner + two-column body
8.  Adrian Santos         – label-only left sidebar ("Work Experience", "Skills", "—")
                            already handled via _is_label_only_sidebar()
9.  Kian Andrei James     – student single-column with PROJECTS + CERTIFICATIONS
                            sections (multi-row, no work experience header required)

v41 changes vs v40
-------------------
A. _detect_column_split()
   - Added `min_boxes` guard: skip column detection when total box count < 6
     (prevents false two-column split on sparse pages like Kian's certs section).
   - Added `density_ratio` guard: require the minority column to contain at
     least 15% of boxes — rules out a lone date/city label being misread as
     a second column (seen with Cedric Tolentino's right-aligned city names).

B. _is_full_width_header_band()  [NEW]
   - Detects a top banner that spans >85% of page width (Daniel Cortez navy
     header).  Boxes inside the band are emitted first, top-to-bottom, before
     the two-column body below them.

C. _is_contact_sidebar()  [NEW]
   - Detects a left column that is ONLY contact / meta lines (email, phone,
     address, short labels like "CONTACT INFO", "EDUCATION", skill names).
   - Distinct from _is_label_only_sidebar(): contact sidebars DO have real
     content (email, phone, school, skill names) but none of it is work
     narrative — they should still be emitted first (left → right).

D. _sort_boxes_reading_order()
   - Extracts full-width header band first if detected.
   - Then applies existing left→right column logic on the remaining boxes.
   - Falls back to plain top-to-bottom on true single-column layouts.

E. _clean_text() / bullet regexes
   - Added `→` and `✗` to bullet sets (seen in some template variants).
   - Added `☑` and `►` defensively.

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

Fix log (v4, retained)
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

6. Multi-page PDFs — pages joined with "\\n\\n" so section parsers see a
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
# Bullet / decoration cleanup  (v41: added →, ✗, ☑, ►)
# ---------------------------------------------------------------------------

_STANDALONE_BULLET_RE = re.compile(
    r"(?m)^[ \t]*[•◦▪\-\*○➦✔✓→✗☑►!@✉][ \t]*$"
)

_INLINE_BULLET_RE = re.compile(
    r"^[ \t]*[•◦▪○➦✔✓→✗☑►!\-\*][ \t]+"
)

_CONTENT_LINE_RE = re.compile(
    r"(\d{4})"          # year → real content
    r"|@"               # email → real content
    r"|\(\d{2,}\)"      # phone fragment → real content
    r"|(?:\w+\s+){4,}\w+"  # 5+ word run → real narrative
)

# Matches contact-info patterns: email, phone, URL, short address fragments
_CONTACT_LINE_RE = re.compile(
    r"@"                        # email address
    r"|\(\d{2,}\)"             # phone number fragment like (63)
    r"|\d{3}[-.\s]\d{3,}"     # phone digits
    r"|https?://"              # URL
    r"|linkedin\.com"          # LinkedIn handle
    r"|github\.com"            # GitHub handle
    r"|philippines"            # country/city mention (case-insensitive applied via flag)
    r"|manila|makati|quezon|pasig|taguig|cebu|pasay|mandaluyong|pampanga",
    re.IGNORECASE,
)

# Section-header keywords found in sidebar columns
_SIDEBAR_HEADER_RE = re.compile(
    r"^(contact\s*info|education|skills|summary|languages|certifications"
    r"|references|profile|about\s*me|hobbies|interests|awards|projects"
    r"|training|volunteer|publications|expertise)$",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str:
    return _INLINE_BULLET_RE.sub("", line)


def _clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _STANDALONE_BULLET_RE.fullmatch(stripped) or stripped in {
            "•", "◦", "▪", "○", "➦", "✔", "✓", "→", "✗", "☑", "►",
            "!", "@", "✉",
        }:
            continue
        cleaned.append(_clean_line(line))

    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


# ---------------------------------------------------------------------------
# Column-split detection  (v41: density_ratio + min_boxes guards)
# ---------------------------------------------------------------------------

def _detect_column_split(
        x_centers: list[float],
        *,
        min_boxes: int = 6,
        min_density_ratio: float = 0.15,
) -> float | None:
    """
    Dynamically detect the x coordinate that divides a two-column layout.

    Guards added in v41
    -------------------
    min_boxes
        Skip detection entirely when there are fewer than `min_boxes` OCR
        boxes on the page.  Sparse pages (e.g. a section with only a header
        and two bullet lines) frequently produce a false column split.

    min_density_ratio
        After finding the largest gap, verify that the minority column
        (whichever of left/right has fewer boxes) contains at least this
        fraction of all boxes.  A single stray right-aligned city label
        (like "Cebu City, Philippines" in Cedric Tolentino's layout) would
        otherwise be misidentified as a second column.

    Search zone: 0.20–0.85 of page width.
    Minimum qualifying gap: 5% of page width.
    """
    if len(x_centers) < min_boxes:
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

    if best_split is None or best_gap <= 0.05 * page_width:
        return None

    # Density guard: reject splits where one column has almost no boxes
    n_left  = sum(1 for x in x_centers if x <= best_split)
    n_right = len(x_centers) - n_left
    minority = min(n_left, n_right)
    if minority / len(x_centers) < min_density_ratio:
        log.debug(
            "[OCR] Column split at %.1f rejected: minority column density %.1f%% < %.0f%%",
            best_split, minority / len(x_centers) * 100, min_density_ratio * 100,
                        )
        return None

    return best_split


# ---------------------------------------------------------------------------
# Full-width header band detection  (v41 NEW — Daniel Cortez navy banner)
# ---------------------------------------------------------------------------

def _is_full_width_header_band(
        boxes: list[tuple[float, float, str]],
        *,
        width_threshold: float = 0.85,
        max_band_height_ratio: float = 0.20,
) -> float | None:
    """
    Detect a full-width banner at the top of the page (e.g. Daniel Cortez's
    dark navy header containing name + job title).

    Strategy
    --------
    1. Find boxes in the top `max_band_height_ratio` fraction of page height.
    2. Compute what fraction of page width those boxes collectively span.
    3. If that span exceeds `width_threshold`, the band exists.
    4. Return the y-coordinate of the bottom of the band so callers can
       partition boxes into "header" vs "body".

    Returns the band's bottom y (float) if a full-width band is detected,
    or None otherwise.
    """
    if not boxes:
        return None

    ys         = [b[0] for b in boxes]
    xs         = [b[1] for b in boxes]
    page_height = max(ys) if ys else 0
    page_width  = max(xs) if xs else 0
    if page_height == 0 or page_width == 0:
        return None

    band_y_limit = max_band_height_ratio * page_height
    top_boxes    = [b for b in boxes if b[0] <= band_y_limit]

    if len(top_boxes) < 2:
        return None

    top_xs  = [b[1] for b in top_boxes]
    span    = (max(top_xs) - min(top_xs)) / page_width

    if span >= width_threshold:
        band_bottom = max(b[0] for b in top_boxes)
        log.debug(
            "[OCR] Full-width header band detected: y<=%.1f, span=%.0f%%",
            band_bottom, span * 100,
                         )
        return band_bottom

    return None


# ---------------------------------------------------------------------------
# Contact sidebar detection  (v41 NEW — Adrian Velasco / Angelica Soriano)
# ---------------------------------------------------------------------------

def _is_contact_sidebar(left_blocks: list[tuple[float, float, str]]) -> bool:
    """
    Return True when the left column looks like a contact/meta sidebar
    (contact info, education summary, skill names) rather than narrative
    work content.

    Unlike _is_label_only_sidebar(), a contact sidebar CAN have real text
    (email address, school name, skill names) — it just contains no
    multi-sentence work narrative.  The heuristic checks:

    - At least one contact-info pattern (email / phone / city name).
    - No line longer than 8 words that also looks like a work bullet
      (i.e. starts with a verb or contains a date range like "2021 - 2023").

    This correctly classifies:
    - Adrian Velasco  : left side has email, phone, city, EDUCATION block,
                        Node.js / Express.js / MongoDB skill list.
    - Angelica Soriano: left side has email, phone, city, EDUCATION block,
                        Next.js / React.js / TypeScript skill list.
    - Brendon Lucero  : dark sidebar with email, phone, city, skill list.
    - Aaron Villareal : right-side skills sidebar (handled by col_split
                        already; this path not needed but won't misfire).
    """
    if not left_blocks:
        return False

    has_contact_signal  = False
    has_work_narrative  = False

    _DATE_RANGE_RE = re.compile(r"\d{4}\s*[-–]\s*(\d{4}|current|present)", re.IGNORECASE)
    _WORK_VERB_RE  = re.compile(
        r"^(developed|designed|built|implemented|created|managed|led|maintained"
        r"|optimized|deployed|reduced|improved|collaborated|assisted|conducted"
        r"|produced|supported|participated|engineered|automated|configured"
        r"|coordinated|analyzed|integrated|handled|generated|achieved)",
        re.IGNORECASE,
    )

    for (_y, _x, text) in left_blocks:
        stripped = text.strip()
        if not stripped or stripped in {"—", "–", "−"}:
            continue

        if _CONTACT_LINE_RE.search(stripped):
            has_contact_signal = True

        if _SIDEBAR_HEADER_RE.match(stripped):
            # Section headers are neutral — skip them
            continue

        words = stripped.split()
        if len(words) > 8 and (
                _DATE_RANGE_RE.search(stripped) or _WORK_VERB_RE.match(stripped)
        ):
            has_work_narrative = True
            break

    return has_contact_signal and not has_work_narrative


# ---------------------------------------------------------------------------
# Label-only sidebar detection  (unchanged from v40)
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
# Y-zone interleaving (label-only sidebar path only — unchanged from v40)
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
        result.extend(
            e[2] for e in right_sorted
            if zone_start <= e[0] < zone_end and e[2] not in result
        )

    return result


# ---------------------------------------------------------------------------
# Reading-order sort  (v41: header-band extraction + contact-sidebar path)
# ---------------------------------------------------------------------------

def _sort_boxes_reading_order(raw_results: list) -> list[str]:
    """
    Convert EasyOCR detail=1 output → reading-order list of text strings.

    Decision tree (evaluated in order)
    ------------------------------------
    1. Detect full-width header band (Daniel Cortez-style navy banner).
       If found, emit header boxes first; apply column logic only to body.

    2. Detect column split from x-center distribution.
       If no split → single-column (Camille Reyes, Cedric Tolentino, Kian
       Andrei James): plain top-to-bottom order.

    3. Classify left column:
       a. Label-only sidebar (Adrian Santos)  → Y-zone interleave.
       b. Contact/meta sidebar (Adrian Velasco, Angelica Soriano, Brendon
          Lucero, Aaron Villareal)            → left-first, then right.
       c. Content column (genuine two-column narrative) → left-first, then right.

    v40: pdf_layout parameter removed (PyMuPDF gone). Column split is always
    self-detected from EasyOCR bounding boxes.
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

    # ── Step 1: Full-width header band (Daniel Cortez) ────────────────────
    header_bottom = _is_full_width_header_band(boxes)
    header_lines: list[str] = []
    body_boxes = boxes

    if header_bottom is not None:
        header_boxes = sorted(
            [b for b in boxes if b[0] <= header_bottom],
            key=lambda b: b[0],
        )
        body_boxes = [b for b in boxes if b[0] > header_bottom]
        header_lines = [b[2] for b in header_boxes]
        log.debug("[OCR] Emitting %d header-band boxes first.", len(header_lines))

    # ── Step 2: Column split on body boxes ───────────────────────────────
    x_centers = [b[1] for b in body_boxes]
    col_split = _detect_column_split(x_centers)

    if col_split is None:
        # Single-column layout (or too sparse to detect split)
        log.debug("[OCR] Single-column layout — plain top-to-bottom order.")
        ordered = sorted(body_boxes, key=lambda b: b[0])
        return header_lines + [b[2] for b in ordered]

    left  = sorted([b for b in body_boxes if b[1] <= col_split], key=lambda b: b[0])
    right = sorted([b for b in body_boxes if b[1]  > col_split], key=lambda b: b[0])

    # ── Step 3: Classify left column ─────────────────────────────────────
    is_label_only  = _is_label_only_sidebar(left)
    is_contact     = (not is_label_only) and _is_contact_sidebar(left)

    log.debug(
        "[OCR] Self-detected col_split=%.1f is_label_only=%s is_contact=%s",
        col_split, is_label_only, is_contact,
    )

    if is_label_only:
        log.debug("[OCR] Applying Y-zone interleave (label-only sidebar).")
        body_lines = _interleave_label_sidebar(left, right)
    else:
        # Both contact-sidebar and genuine two-column layouts:
        # emit left column first (top-to-bottom), then right column.
        body_lines = [b[2] for b in (left + right)]

    return header_lines + body_lines


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
    log.info("[Stage 1] EasyOCR extraction (v41 — OCR-only) ...")
    ocr_text = _ocr_fallback(file_path)
    quality  = _alnum_ratio(ocr_text)
    log.info("[Stage 1] OCR done (%d chars, quality %.1f%%).", len(ocr_text), quality)
    return ocr_text, quality, False