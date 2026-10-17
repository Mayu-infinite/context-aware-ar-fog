from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """One object detected in an image."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class YOLODetector:
    """Lightweight YOLO-based object detector."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.35,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

        self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR OpenCV frame."""

        kwargs = {
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(frame, **kwargs)

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = result.names[class_id]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

        return detections