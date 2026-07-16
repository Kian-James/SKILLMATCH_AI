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

Reader lifecycle
-----------------
The EasyOCR Reader is loaded ONCE, at process startup, and reused for
every request. It is NOT reloaded per-request and OCR is NOT run in a
spawned subprocess.

(An earlier version of this code, ported from a macOS dev environment,
ran each OCR call in a spawned subprocess to work around a torch
deadlock/segfault bug specific to Apple Silicon. That workaround doesn't
apply on Render's Linux containers, and spawning a fresh process per
request means re-importing torch and rebuilding the Reader from scratch
every single call — a second full model load stacked on top of the
already-loaded one in the main process. On a 512MB instance that's
exactly the kind of thing that gets a container OOM-killed mid-request,
which is what a 502 Bad Gateway right after "Loading EasyOCR reader"
in the logs usually means. Loading once and keeping it in memory avoids
that entirely and is also much faster per request.)

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

import logging
import re
import tempfile
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OCR_LANGUAGES = os.getenv("EASYOCR_LANGUAGES", "en").split(",")

_ocr_reader = None

# Only one OCR run at a time. FastAPI runs sync route handlers in a thread
# pool that can execute several requests concurrently by default — without
# this, two resumes uploaded close together could run readtext() in
# parallel and double the memory spike right when there's no headroom for
# it. This forces a second request to genuinely wait, not run alongside
# the first.
_ocr_semaphore = threading.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the reader once, at startup, from the weights pre-baked into
    # the Docker image at build time — not on the first request.
    global _ocr_reader
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    import easyocr
    log.info("[OCR] Loading EasyOCR reader (gpu=False) at startup ...")
    _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    log.info("[OCR] Reader ready.")

    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log.info("[OCR] Peak RSS after reader load: %.1f MB", peak_mb)
    except Exception as exc:
        log.warning("[OCR] Could not read memory usage: %s", exc)

    yield


app = FastAPI(
    title="OCR Microservice",
    description="EasyOCR-only text extraction. Called by the main resume-analysis backend.",
    lifespan=lifespan,
)

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
# OCR — runs in-process against the reader loaded at startup
# ---------------------------------------------------------------------------

def _run_ocr(file_path: str) -> str:
    reader = _ocr_reader
    if reader is None:
        raise RuntimeError("OCR reader is not ready yet.")

    import torch
    import gc

    lines: list[str] = []

    if file_path.lower().endswith(".pdf"):
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
            from pdf2image import convert_from_path

            info = pdfinfo_from_path(file_path)
            num_pages = info["Pages"]

            for page_num in range(1, num_pages + 1):
                with tempfile.TemporaryDirectory() as tmpdir:
                    # first_page/last_page limits rendering to ONE page at a
                    # time, instead of rasterizing the whole document into
                    # memory at once — this is what actually caps peak
                    # memory regardless of how many pages the resume has.
                    page_imgs = convert_from_path(
                        file_path, dpi=150, output_folder=tmpdir,
                        first_page=page_num, last_page=page_num,
                    )
                    img = page_imgs[0]
                    img_arr = np.array(img)

                    with torch.inference_mode():
                        raw = reader.readtext(
                            img_arr, detail=1, paragraph=False,
                            canvas_size=1280,
                        )
                    lines.extend(_sort_boxes_reading_order(raw))
                    lines.append("")  # page-break marker

                    del img, img_arr, page_imgs, raw
                    gc.collect()
        except Exception as exc:
            log.warning("[OCR] Page-by-page PDF handling failed (%s); direct image fallback ...", exc)
            try:
                with torch.inference_mode():
                    raw = reader.readtext(file_path, detail=1, paragraph=False, canvas_size=1280)
                lines = _sort_boxes_reading_order(raw)
            except Exception as exc2:
                log.warning("[OCR] Direct fallback also failed (%s).", exc2)
    else:
        try:
            with torch.inference_mode():
                raw = reader.readtext(file_path, detail=1, paragraph=False, canvas_size=1280)
            lines = _sort_boxes_reading_order(raw)
        except Exception as exc:
            log.warning("[OCR] readtext failed (%s).", exc)

    return _clean_text("\n\n".join(lines))


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(c.isalnum() for c in text) / len(text) * 100, 2)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "reader_ready": _ocr_reader is not None}


# Plain `def` (not `async def`) — FastAPI runs sync path operations in a
# threadpool automatically, which keeps this CPU-bound OCR call from
# blocking the event loop.
@app.post("/extract-text")
def extract_text_endpoint(file: UploadFile = File(...)) -> dict:
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buf:
        buf.write(file.file.read())

    try:
        log.info("[OCR] Waiting for OCR slot (queue depth may apply) ...")
        with _ocr_semaphore:
            log.info("[OCR] Slot acquired, running OCR ...")
            text = _run_ocr(temp_path)
            quality = _alnum_ratio(text)
    except Exception as exc:
        log.exception("[OCR service] extraction error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        import gc
        gc.collect()

    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log.info("[OCR] Peak RSS after this request: %.1f MB", peak_mb)
    except Exception:
        pass

    return {"text": text, "quality": quality, "is_native_pdf": False}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)