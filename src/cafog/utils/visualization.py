from __future__ import annotations

import cv2
import numpy as np

from cafog.context.state import SceneState
from cafog.perception.detector import Detection
from cafog.policy.cue_policy import Cue


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw YOLO detections on an OpenCV frame."""

    output = frame.copy()

    for detection in detections:
        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output


def draw_pipeline_status(
    frame: np.ndarray,
    scene_state: SceneState,
    cue: Cue,
) -> np.ndarray:
    """Display context and recommended action."""

    output = frame.copy()

    lines = [
        f"Context: {scene_state.context.value}",
        f"Action: {cue.action.value}",
        f"Visual: {cue.visual_cue}",
        f"Audio: {cue.audio_cue}",
    ]

    x = 20
    y = 30

    for line in lines:
        cv2.putText(
            output,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        y += 30

    return output