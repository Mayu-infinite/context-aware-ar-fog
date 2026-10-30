"""Frame sources: phone stream (IP Webcam URL), local webcam, or a recorded video."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def is_live(source: str) -> bool:
    return source.isdigit() or "://" in source


class LatestFrameReader:
    """Reads a live stream on a thread and keeps only the newest frame.

    Network streams buffer frames; reading them in order adds seconds of lag when the
    processing is slower than the camera. Dropping stale frames keeps the cue current.
    """

    def __init__(self, source: str):
        self.capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera stream {source!r}")
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._fresh = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()
            self._fresh.set()

    def read(self, timeout_s: float = 5.0) -> tuple[np.ndarray, float] | None:
        if not self._fresh.wait(timeout_s):
            return None
        with self._lock:
            self._fresh.clear()
            return self._frame, self._stamp

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.capture.release()


def iter_frames(source: str, *, every: int = 1) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (BGR frame, timestamp s): video files use their own clock, live sources wall time."""
    if is_live(source):
        reader = LatestFrameReader(source)
        try:
            while True:
                item = reader.read()
                if item is None:
                    raise RuntimeError("No frames from the camera stream for 5 s")
                yield item
        finally:
            reader.close()
        return

    if not Path(source).exists():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(source)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                yield frame, index / fps
            index += 1
    finally:
        capture.release()
