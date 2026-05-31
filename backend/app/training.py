"""
backend/app/training.py
=======================
Spawns and manages the YOLOv9 training subprocess.

Only one training job is allowed at a time — concurrent runs would
oversubscribe the GPU and produce garbage. ``start()`` raises
:class:`RuntimeError` if a job is already in progress.

The subprocess is ``python -m training.train ...`` so all the
microscopy-tuned augmentation defaults stay in one place
(``training/train.py``). We just supply CLI args.

Logs are captured by a background reader thread into a bounded
``deque``; the polling endpoint reads the tail of it. We don't stream
via WS/SSE because polling is plenty for a single-user local app and
adds no infra complexity.
"""
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional


# Strip CSI escape sequences (\x1b[...m and friends) so logs stay readable.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _resolve_train_device(spec: str) -> Optional[str]:
    """Translate the form's device value into a concrete --device argument.

    - "auto" / "" → "0" if a CUDA GPU is visible, else None (Ultralytics
      will fall back to CPU). This is what makes "auto" actually use the
      NVIDIA GPU by default; without it we depended on Ultralytics' own
      detection, which has historically been unreliable.
    - any explicit value ("0", "cpu", "0,1", "cuda:0", ...) passes
      through unchanged.

    Returning None means "don't pass --device at all".
    """
    if spec and spec.strip().lower() not in ("", "auto"):
        return spec.strip()
    try:
        import torch  # already loaded by the backend; cheap import
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return None


log = logging.getLogger(__name__)


# Resolve once at import time so the subprocess always runs from the
# repo root regardless of where uvicorn was started.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TrainingService:
    """Singleton owner of the training subprocess + log buffer."""

    def __init__(self, log_capacity: int = 4000):
        self._proc: Optional[subprocess.Popen] = None
        self._logs: Deque[str] = deque(maxlen=log_capacity)
        self._lock = threading.Lock()
        self._state: str = "idle"   # idle | running | completed | failed | killed
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._return_code: Optional[int] = None
        self._name: Optional[str] = None
        self._command: Optional[List[str]] = None
        self._reader: Optional[threading.Thread] = None

    # ----------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ----------------------------------------------------------------------

    def start(
        self,
        data: str,
        weights: str,
        epochs: int,
        batch: int,
        imgsz: int,
        device: str,
        name: Optional[str] = None,
    ) -> dict:
        """Spawn a training run. Raises RuntimeError if one is in progress."""
        with self._lock:
            if self.is_running:
                raise RuntimeError("A training run is already in progress.")

            if not name:
                name = f"ecoli_{int(time.time())}"

            cmd: List[str] = [
                sys.executable, "-m", "training.train",
                "--data", data,
                "--weights", weights,
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--imgsz", str(imgsz),
                "--name", name,
            ]
            resolved_device = _resolve_train_device(device)
            if resolved_device is not None:
                cmd += ["--device", resolved_device]
                log.info("Training will use --device=%s", resolved_device)
            else:
                log.info("No CUDA detected — training on CPU.")

            log.info("Spawning training subprocess: %s (cwd=%s)",
                     " ".join(cmd), _REPO_ROOT)

            env = os.environ.copy()
            # Force unbuffered stdout so log lines arrive in near real time.
            env["PYTHONUNBUFFERED"] = "1"
            # Tell the child Python to emit UTF-8 on stdout/stderr.
            # Without this, Ultralytics' Unicode glyphs (✓, ✗, ⚡, etc.)
            # come out in the system codepage and corrupt our reader.
            env["PYTHONIOENCODING"] = "utf-8"

            # encoding="utf-8" + errors="replace" makes the reader robust
            # to any odd byte sequences (ANSI escapes, emoji, partial UTF-8
            # at chunk boundaries). Without errors="replace" the reader
            # thread crashes on the first bad byte and we lose all logs.
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge — saves one stream to demux
                bufsize=1,                  # line-buffered
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(_REPO_ROOT),
                env=env,
            )

            self._state = "running"
            self._started_at = time.time()
            self._finished_at = None
            self._return_code = None
            self._name = name
            self._command = cmd
            self._logs.clear()
            self._logs.append(f"$ {' '.join(cmd)}")

            self._reader = threading.Thread(
                target=self._read_loop, daemon=True, name="train-log-reader",
            )
            self._reader.start()

        return self._status_locked()

    # ----------------------------------------------------------------------

    def stop(self) -> dict:
        """Terminate the subprocess. No-op if nothing is running."""
        with self._lock:
            if not self.is_running or self._proc is None:
                return self._status_locked()
            log.info("Terminating training process pid=%s", self._proc.pid)
            try:
                self._proc.terminate()
            except Exception as exc:
                log.warning("terminate() raised: %s", exc)
            self._state = "killed"
        # Wait outside the lock so the reader can drain final logs.
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning("Training process didn't exit after SIGTERM; killing.")
                try:
                    self._proc.kill()
                except Exception:
                    pass
        return self.status()

    # ----------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Drain subprocess stdout into the ring buffer until EOF.

        Per-line read is wrapped in its own try/except so a single
        decode hiccup can't kill the entire reader (we still want
        subsequent lines).
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                try:
                    line = proc.stdout.readline()
                except Exception as exc:
                    self._logs.append(f"[log reader: {exc!r}]")
                    continue
                if not line:
                    break
                # ANSI escape sequences from Ultralytics look ugly in a
                # plain log box; strip the most common ones. The byte
                # \x1b is ESC; the pattern matches CSI sequences.
                line = _ANSI_RE.sub("", line)
                self._logs.append(line.rstrip("\r\n"))
        finally:
            rc = proc.wait()
            with self._lock:
                self._finished_at = time.time()
                self._return_code = rc
                if self._state == "running":
                    self._state = "completed" if rc == 0 else "failed"
            log.info("Training process exited with code %d (state=%s)",
                     rc, self._state)

    # ----------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict:
        return {
            "state": self._state,
            "pid": self._proc.pid if (self._proc and self.is_running) else None,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "return_code": self._return_code,
            "name": self._name,
            "command": list(self._command) if self._command else None,
            # Cap at last 400 lines per poll — older lines stay in the buffer
            # but don't ship over the wire on every tick.
            "log_lines": list(self._logs)[-400:],
        }


# Module-level singleton — imported by routes.
training_service = TrainingService()
