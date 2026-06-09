"""
core/ocr_worker_proc.py
=======================
Persistent OCR worker process.

Problem with the old approach
------------------------------
The old _ocr_fallback() used multiprocessing.spawn to create a *new* child
process for every single OCR request. On Linux, spawn copies the entire parent
address space before exec-ing. The parent already holds SBERT (~90 MB) +
FastAPI + PyMuPDF in RAM. The child then loads EasyOCR + PyTorch on top of
that copy. Peak RAM at the moment the child starts = parent RAM + child RAM,
which easily blows through 512 MB.

New approach
------------
Start ONE persistent worker process at app startup (before any models are
loaded into the parent). The worker sits idle, owns EasyOCR entirely, and
never shares RAM with the parent's models.

Communication: two multiprocessing.Queues (request_q, response_q).
  - Parent sends: (file_path, layout_json_str, request_id)
  - Worker sends: (request_id, lines_list | None, error_str | None)

The worker is started with "spawn" so it begins with a clean address space —
no copy of the parent's heap. It loads EasyOCR once and loops forever.

If the worker dies (OOM, segfault), _send_ocr_request() detects the dead
process and restarts it before retrying once.
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

# ── Constants ────────────────────────────────────────────────────────────────
_WORKER_TIMEOUT  = 120   # seconds to wait for OCR result
_STARTUP_TIMEOUT = 60    # seconds to wait for worker ready signal


# ── Worker entry point (runs in the child process) ───────────────────────────

def _worker_main(
    request_q:  multiprocessing.Queue,
    response_q: multiprocessing.Queue,
    languages:  list[str],
) -> None:
    """
    Child-process entry point.  Loads EasyOCR once, then loops processing
    requests from request_q and writing results to response_q.

    Imports are intentionally deferred to after the process starts so that
    the parent process never touches these modules.
    """
    # Force single-threaded math libs *before* any import
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"

    # Late imports — only in the child
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    import easyocr
    from pdf2image import convert_from_path
    import tempfile

    # Import the column-sort helper from extraction — it has no heavy deps
    # and we need it to replicate the reading-order logic.
    from core.extraction import _sort_boxes_reading_order

    log.info("[OCR-worker] Loading EasyOCR reader ...")
    reader = easyocr.Reader(languages, gpu=False)
    log.info("[OCR-worker] EasyOCR ready. Signalling parent.")

    # Tell the parent we're alive and ready
    response_q.put(("__ready__", None, None))

    while True:
        try:
            item = request_q.get(timeout=5)
        except Exception:
            # Timeout is fine — just loop and wait
            continue

        if item is None:
            # Poison pill — graceful shutdown
            log.info("[OCR-worker] Received shutdown signal.")
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


# ── Persistent worker manager (lives in the parent process) ──────────────────

class _OcrWorkerManager:
    """
    Manages one persistent OCR worker child process.

    Thread-safety: FastAPI runs request handlers in a thread pool, so
    send_request() may be called concurrently.  We protect the queues and
    process handle with a threading.Lock so only one request is in-flight
    at a time (EasyOCR is single-threaded anyway — parallelism here buys
    nothing and would require multiple workers).
    """

    def __init__(self) -> None:
        import threading
        self._lock     = threading.Lock()
        self._process: Optional[multiprocessing.Process] = None
        self._req_q:   Optional[multiprocessing.Queue]   = None
        self._res_q:   Optional[multiprocessing.Queue]   = None
        self._languages: list[str] = ["en"]
        self._started  = False

    def start(self, languages: list[str] = ("en",)) -> None:
        """
        Spawn the worker.  Call this once at app startup, *before* loading
        SBERT or any other large model into the parent process.
        """
        self._languages = list(languages)
        self._spawn()
        self._started = True

    def _spawn(self) -> None:
        ctx        = multiprocessing.get_context("spawn")
        self._req_q = ctx.Queue()
        self._res_q = ctx.Queue()

        self._process = ctx.Process(
            target  = _worker_main,
            args    = (self._req_q, self._res_q, self._languages),
            daemon  = True,   # dies automatically when parent exits
            name    = "ocr-worker",
        )
        self._process.start()
        log.info("[OCR-manager] Worker PID %d spawned.", self._process.pid)

        # Wait for the ready signal
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            try:
                item = self._res_q.get(timeout=2)
                if item[0] == "__ready__":
                    log.info("[OCR-manager] Worker signalled ready.")
                    return
            except Exception:
                pass
            if not self._process.is_alive():
                raise RuntimeError(
                    f"[OCR-manager] Worker process died during startup "
                    f"(exit code {self._process.exitcode})."
                )
        raise RuntimeError("[OCR-manager] Worker did not signal ready in time.")

    def _ensure_alive(self) -> None:
        if self._process is None or not self._process.is_alive():
            log.warning("[OCR-manager] Worker is dead — restarting ...")
            self._spawn()

    def send_request(
        self,
        file_path:   str,
        layout_json: str = "",
        timeout:     int = _WORKER_TIMEOUT,
    ) -> Optional[list[str]]:
        """
        Send an OCR request and block until the result arrives or timeout.

        Returns the list of text lines, or None on failure.
        Restarts the worker once if it has died.
        """
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
                    # Queue.get timed out — check if worker is still alive
                    if not self._process.is_alive():
                        log.error(
                            "[OCR-manager] Worker died while processing "
                            "(exit %s) — restarting.", self._process.exitcode
                        )
                        self._spawn()
                        # Re-send after restart
                        self._req_q.put((req_id, file_path, layout_json))
                        continue
                    continue

                got_id, lines, error = item
                if got_id != req_id:
                    # Stale response from a previous request — discard
                    log.debug("[OCR-manager] Discarding stale response for %s", got_id)
                    continue

                if error:
                    log.error("[OCR-manager] Worker reported error: %s", error)
                    return None

                return lines

            log.warning(
                "[OCR-manager] OCR timed out after %ds for %s", timeout, file_path
            )
            # Kill and restart the hung worker
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
                self._req_q.put(None)   # poison pill
            except Exception:
                pass
        if self._process is not None:
            self._process.join(10)


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: Optional[_OcrWorkerManager] = None


def get_ocr_manager() -> _OcrWorkerManager:
    global _manager
    if _manager is None:
        raise RuntimeError(
            "OCR worker not started. Call start_ocr_worker() at app startup."
        )
    return _manager


def start_ocr_worker(languages: list[str] = None) -> None:
    """
    Spawn the persistent OCR worker.  Call this from the FastAPI lifespan
    handler, BEFORE loading SBERT or any other model.

    Example (app/main.py):

        from contextlib import asynccontextmanager
        from core.ocr_worker_proc import start_ocr_worker, shutdown_ocr_worker

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            start_ocr_worker()   # ← first, before SBERT loads
            yield
            shutdown_ocr_worker()

        app = FastAPI(lifespan=lifespan, ...)
    """
    global _manager
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