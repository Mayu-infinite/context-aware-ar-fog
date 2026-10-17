from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class Camera:
    """OpenCV camera wrapper."""

    def __init__(
        self,
        config: CameraConfig | None = None,
    ) -> None:
        self.config = config or CameraConfig()

        self.capture = cv2.VideoCapture(
            self.config.index
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.config.width,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.config.height,
        )

        self.capture.set(
            cv2.CAP_PROP_FPS,
            self.config.fps,
        )

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> Camera:
        if not self.is_opened():
            raise RuntimeError(
                f"Could not open camera {self.config.index}"
            )

        return self

    def __exit__(self, *args) -> None:
        self.release()