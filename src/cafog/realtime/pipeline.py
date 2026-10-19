from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cafog.context.rules import ContextEngine
from cafog.context.state import SceneState
from cafog.perception.detector import Detection, YOLODetector
from cafog.planning.value_field import ValueField, ValueInputs
from cafog.policy.actions import ActionType
from cafog.policy.cue_policy import Cue, CuePolicy


@dataclass
class PipelineResult:
    detections: list[Detection]
    scene_state: SceneState
    cue: Cue


class RealtimePipeline:
    """Research prototype: perception → context → value → cue selection.

    Object detections are preliminary evidence, not proof that a route is
    safe. Without calibrated geometry, this pipeline does not infer object
    distances, doorway widths, clearance, or walking directions.
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
        value_field: ValueField | None = None,
    ) -> None:
        self.detector = detector
        self.context_engine = context_engine
        self.cue_policy = cue_policy
        self.value_field = value_field or ValueField()

    def process(self, frame: np.ndarray) -> PipelineResult:
        detections = self.detector.detect(frame)
        scene_state = self._build_scene_state(detections)

        self.context_engine.classify(scene_state)

        # Detection-only frames do not provide calibrated clearance,
        # goal progress, or route geometry. Keep these inputs unavailable.
        estimate = self.value_field.estimate(
            ValueInputs(
                goal_progress=None,
                clearance=None,
                uncertainty=1.0,
                turn_complexity=None,
            )
        )
        scene_state.spatial_value = estimate.value
        scene_state.spatial_value_confidence = estimate.confidence
        scene_state.spatial_value_change = None
        scene_state.route_evidence_sufficient = False

        cue = self.cue_policy.select(
            scene_state.context,
            value_estimate=estimate,
            evidence_sufficient=False,
        )

        # Defense in depth: generic detections must not produce route or
        # directional instructions without separately assessed geometry.
        if cue.action in {
            ActionType.FOLLOW_SAFE_PATH,
            ActionType.CENTER_PATH,
            ActionType.TURN_LEFT,
            ActionType.TURN_RIGHT,
        }:
            cue = Cue(
                action=ActionType.NO_AUTOMATIC_ACTION,
                visual_cue="NONE",
                audio_cue="NONE",
                text_feedback="NONE",
                explanation=(
                    "Automatic route guidance suppressed: "
                    "route geometry has not been assessed."
                ),
            )

        return PipelineResult(
            detections=detections,
            scene_state=scene_state,
            cue=cue,
        )

    def _build_scene_state(
        self,
        detections: list[Detection],
    ) -> SceneState:
        """Build conservative state from generic object detections."""

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
