from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def iter_video_frames(
    video_path: str | Path,
    stride: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Iterate through a video.

    Returns:
        (frame_index, frame)
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % stride == 0:
                yield frame_index, frame

            frame_index += 1

    finally:
        capture.release()


def video_properties(
    video_path: str | Path,
) -> dict[str, float]:
    """Return basic video properties."""

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    width = capture.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )

    height = capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    capture.release()

    duration = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    return {
        "fps": float(fps),
        "frame_count": float(frame_count),
        "width": float(width),
        "height": float(height),
        "duration_seconds": float(duration),
    }