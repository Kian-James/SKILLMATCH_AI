"""
core/ocr_worker_proc.py
=======================
Persistent OCR worker process.

The worker starts LAZILY on the first scanned-PDF request, not at app startup.
This keeps boot RAM low — native PDFs never trigger the worker at all.

Communication: two multiprocessing.Queues (request_q, response_q).
  - Parent sends: (request_id, file_path, layout_json_str)
  - Worker sends: (request_id, lines_list | None, error_str | None)

If the worker dies (OOM, segfault), send_request() detects it and restarts
before retrying once automatically.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
import uuid
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_WORKER_TIMEOUT  = 120  # seconds to wait for OCR result
_STARTUP_TIMEOUT = 60   # seconds to wait for worker ready signal


# ── Worker entry point (runs in the child process) ───────────────────────────

def _worker_main(
    request_q:  multiprocessing.Queue,
    response_q: multiprocessing.Queue,
    languages:  list[str],
) -> None:
    """
    Child-process entry point. Loads EasyOCR once, then loops processing
    requests. All heavy imports are deferred to here so the parent process
    never touches EasyOCR or PyTorch.
    """
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    import easyocr
    from pdf2image import convert_from_path
    import tempfile
    from core.extraction import _sort_boxes_reading_order

    log.info("[OCR-worker] Loading EasyOCR reader ...")
    reader = easyocr.Reader(languages, gpu=False)
    log.info("[OCR-worker] EasyOCR ready.")

    # Signal parent that we're ready
    response_q.put(("__ready__", None, None))

    while True:
        try:
            item = request_q.get(timeout=5)
        except Exception:
            continue

        if item is None:
            log.info("[OCR-worker] Shutdown signal received.")
            break

        req_id, file_path, layout_json = item

        try:
            pdf_layouts: list[dict] = json.loads(layout_json) if layout_json else []
            lines: list[str] = []

            if file_path.lower().endswith(".pdf"):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pages = convert_from_path(file_path, dpi=200, output_folder=tmpdir)
                        for page_idx, img in enumerate(pages):
                            layout = pdf_layouts[page_idx] if page_idx < len(pdf_layouts) else None
                            raw    = reader.readtext(np.array(img), detail=1, paragraph=False)
                            lines.extend(_sort_boxes_reading_order(raw, pdf_layout=layout))
                except Exception as exc:
                    log.warning("[OCR-worker] pdf2image failed (%s); direct fallback ...", exc)
                    layout = pdf_layouts[0] if pdf_layouts else None
                    raw    = reader.readtext(file_path, detail=1, paragraph=False)
                    lines  = _sort_boxes_reading_order(raw, pdf_layout=layout)
            else:
                raw   = reader.readtext(file_path, detail=1, paragraph=False)
                lines = _sort_boxes_reading_order(raw, pdf_layout=None)

            response_q.put((req_id, lines, None))

        except Exception as exc:
            log.exception("[OCR-worker] Error processing %s", file_path)
            response_q.put((req_id, None, str(exc)))


# ── Persistent worker manager ─────────────────────────────────────────────────

class _OcrWorkerManager:
    def __init__(self) -> None:
        import threading
        self._lock     = threading.Lock()
        self._process: Optional[multiprocessing.Process] = None
        self._req_q:   Optional[multiprocessing.Queue]   = None
        self._res_q:   Optional[multiprocessing.Queue]   = None
        self._languages: list[str] = ["en"]

    def start(self, languages: list[str] = ("en",)) -> None:
        self._languages = list(languages)
        self._spawn()

    def _spawn(self) -> None:
        ctx           = multiprocessing.get_context("spawn")
        self._req_q   = ctx.Queue()
        self._res_q   = ctx.Queue()
        self._process = ctx.Process(
            target = _worker_main,
            args   = (self._req_q, self._res_q, self._languages),
            daemon = True,
            name   = "ocr-worker",
        )
        self._process.start()
        log.info("[OCR-manager] Worker PID %d spawned.", self._process.pid)

        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            try:
                item = self._res_q.get(timeout=2)
                if item[0] == "__ready__":
                    log.info("[OCR-manager] Worker ready.")
                    return
            except Exception:
                pass
            if not self._process.is_alive():
                raise RuntimeError(
                    f"OCR worker died during startup (exit {self._process.exitcode})."
                )
        raise RuntimeError("OCR worker did not signal ready in time.")

    def _ensure_alive(self) -> None:
        if self._process is None or not self._process.is_alive():
            log.warning("[OCR-manager] Worker dead — restarting ...")
            self._spawn()

    def send_request(
        self,
        file_path:   str,
        layout_json: str = "",
        timeout:     int = _WORKER_TIMEOUT,
    ) -> Optional[list[str]]:
        with self._lock:
            self._ensure_alive()
            req_id = str(uuid.uuid4())
            self._req_q.put((req_id, file_path, layout_json))

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    remaining = deadline - time.monotonic()
                    item = self._res_q.get(timeout=min(remaining, 5))
                except Exception:
                    if not self._process.is_alive():
                        log.error("[OCR-manager] Worker died mid-request — restarting.")
                        self._spawn()
                        self._req_q.put((req_id, file_path, layout_json))
                    continue

                got_id, lines, error = item
                if got_id != req_id:
                    continue  # stale response, discard

                if error:
                    log.error("[OCR-manager] Worker error: %s", error)
                    return None
                return lines

            log.warning("[OCR-manager] OCR timed out after %ds.", timeout)
            try:
                self._process.terminate()
                self._process.join(5)
            except Exception:
                pass
            self._spawn()
            return None

    def shutdown(self) -> None:
        if self._req_q is not None:
            try:
                self._req_q.put(None)
            except Exception:
                pass
        if self._process is not None:
            self._process.join(10)


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: Optional[_OcrWorkerManager] = None


def get_ocr_manager() -> _OcrWorkerManager:
    """
    Return the OCR worker, starting it lazily on the first OCR request.

    EasyOCR + PyTorch only load when a scanned PDF actually arrives.
    Native PDFs never trigger this, so startup RAM stays well under 512 MB.
    """
    global _manager
    if _manager is None:
        start_ocr_worker()
    return _manager


def start_ocr_worker(languages: list[str] = None) -> None:
    """Spawn the persistent OCR worker. Safe to call multiple times."""
    global _manager
    if _manager is not None:
        return  # already running
    from config import OCR_LANGUAGES
    langs = languages or OCR_LANGUAGES
    _manager = _OcrWorkerManager()
    _manager.start(langs)
    log.info("[OCR] Persistent worker started (PID %d).", _manager._process.pid)


def shutdown_ocr_worker() -> None:
    global _manager
    if _manager is not None:
        _manager.shutdown()
        _manager = None