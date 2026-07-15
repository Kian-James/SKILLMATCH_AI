"""
ocr-service/main.py
====================
Standalone EasyOCR microservice. Deploy this as its OWN Render web service,
separate from the main resume-analysis backend.

Why this exists
----------------
EasyOCR pulls in torch, which is heavy on its own. The main backend also
needs torch (via sentence-transformers) plus faiss, firebase-admin, etc.
Running both in one process regularly exceeds Render's 512MB free-tier
limit. Splitting OCR into its own service means each process only carries
the dependencies it actually needs.

Contract
--------
POST /extract-text
  multipart/form-data, field name "file" — a PDF or image.

  Response JSON:
    {
      "text": "...",
      "quality": 87.3,
      "is_native_pdf": false
    }

The main backend's core/extraction.py calls this endpoint and returns the
same (text, quality, is_native_pdf) tuple it always did, so nothing else
in pipeline.py, core/skills.py, etc. needs to change.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import multiprocessing
import re
import shutil
import tempfile

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OCR_LANGUAGES = os.getenv("EASYOCR_LANGUAGES", "en").split(",")
OCR_TIMEOUT_SECONDS = int(os.getenv("OCR_TIMEOUT_SECONDS", "120"))

app = FastAPI(
    title="OCR Microservice",
    description="EasyOCR-only text extraction. Called by the main resume-analysis backend.",
)

_ocr_reader = None

# ---------------------------------------------------------------------------
# Bullet / decoration cleanup (unchanged from core/extraction.py)
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
# Column-split / reading-order logic (unchanged from core/extraction.py)
# ---------------------------------------------------------------------------

def _detect_column_split(x_centers: list[float]) -> float | None:
    if len(x_centers) < 4:
        return None

    xs = sorted(x_centers)
    page_width = max(xs)
    if page_width == 0:
        return None

    best_gap = 0.0
    best_split: float | None = None

    for i in range(len(xs) - 1):
        gap = xs[i + 1] - xs[i]
        mid = (xs[i] + xs[i + 1]) / 2.0
        if 0.20 * page_width < mid < 0.85 * page_width and gap > best_gap:
            best_gap = gap
            best_split = mid

    if best_split is not None and best_gap > 0.05 * page_width:
        return best_split
    return None


def _is_label_only_sidebar(left_blocks: list[tuple[float, float, str]]) -> bool:
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


def _interleave_label_sidebar(
        left: list[tuple[float, float, str]],
        right: list[tuple[float, float, str]],
) -> list[str]:
    _DECORATIVE = {"—", "–", "−", ""}

    left_sorted = sorted(left, key=lambda e: e[0])
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


def _sort_boxes_reading_order(raw_results: list) -> list[str]:
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

    if col_split is not None:
        left = sorted([b for b in boxes if b[1] <= col_split], key=lambda b: b[0])
        right = sorted([b for b in boxes if b[1] > col_split], key=lambda b: b[0])

        if is_label_only:
            return _interleave_label_sidebar(left, right)
        else:
            return [b[2] for b in (left + right)]
    else:
        ordered = sorted(boxes, key=lambda b: b[0])
        return [b[2] for b in ordered]


# ---------------------------------------------------------------------------
# OCR subprocess worker (unchanged from core/extraction.py)
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
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
                    lines.append("")  # page-break marker
        except Exception as exc:
            log.warning("[OCR] pdf2image failed (%s); direct image fallback ...", exc)
            try:
                raw = reader.readtext(file_path, detail=1, paragraph=False)
                lines = _sort_boxes_reading_order(raw)
            except Exception as exc2:
                log.warning("[OCR] Direct fallback also failed (%s).", exc2)
    else:
        try:
            raw = reader.readtext(file_path, detail=1, paragraph=False)
            lines = _sort_boxes_reading_order(raw)
        except Exception as exc:
            log.warning("[OCR] readtext failed (%s).", exc)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lines, f)


def _ocr_fallback(file_path: str, timeout: int = OCR_TIMEOUT_SECONDS) -> str:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_file = tmp.name

    try:
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=_ocr_worker, args=(file_path, out_file))
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join(5)
            log.warning("[OCR] Worker timed out after %ds for %s.", timeout, file_path)
            return ""

        if p.exitcode != 0:
            log.warning("[OCR] Worker exited with code %s for %s.", p.exitcode, file_path)
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


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)) -> dict:
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        text = _ocr_fallback(temp_path)
        quality = _alnum_ratio(text)
    except Exception as exc:
        log.exception("[OCR service] extraction error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"text": text, "quality": quality, "is_native_pdf": False}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
