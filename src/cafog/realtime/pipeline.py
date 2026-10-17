
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cafog.context.rules import ContextEngine
from cafog.context.state import SceneState
from cafog.perception.detector import Detection, YOLODetector
from cafog.policy.cue_policy import Cue, CuePolicy


@dataclass
class PipelineResult:
    detections: list[Detection]
    scene_state: SceneState
    cue: Cue


class RealtimePipeline:
    """Research prototype: perception → context → cue selection.

    Object detections are preliminary evidence, not proof that a
    route is safe. Without calibrated geometry, this pipeline does
    not infer object distances, doorway widths, or walking directions.
    """

    OBSTACLE_CLASSES = {
        "chair",
        "couch",
        "bench",
        "bicycle",
        "motorcycle",
        "car",
        "suitcase",
    }

    def __init__(
        self,
        detector: YOLODetector,
        context_engine: ContextEngine,
        cue_policy: CuePolicy,
    ) -> None:
        self.detector = detector
        self.context_engine = context_engine
        self.cue_policy = cue_policy

    def process(self, frame: np.ndarray) -> PipelineResult:
        detections = self.detector.detect(frame)
        scene_state = self._build_scene_state(detections)

        self.context_engine.classify(scene_state)
        cue = self.cue_policy.select(scene_state.context)

        return PipelineResult(
            detections=detections,
            scene_state=scene_state,
            cue=cue,
        )

    def _build_scene_state(
        self,
        detections: list[Detection],
    ) -> SceneState:
        """Build conservative state from generic object detections.

        Detection of an obstacle-like object sets an evidence flag only.
        It does not provide a calibrated distance or establish a safe path.
        """

        person_detected = any(
            detection.class_name == "person"
            for detection in detections
        )

        obstacle_detected = any(
            detection.class_name in self.OBSTACLE_CLASSES
            for detection in detections
        )

        return SceneState(
            obstacle_detected=obstacle_detected,
            person_detected=person_detected,
            confidence=max(
                (detection.confidence for detection in detections),
                default=0.0,
            ),
        )
