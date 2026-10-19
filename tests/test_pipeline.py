
import numpy as np

from cafog.context.rules import ContextEngine
from cafog.context.state import ContextType
from cafog.perception.detector import Detection
from cafog.policy.cue_policy import CuePolicy
from cafog.realtime.pipeline import RealtimePipeline


class FakeDetector:
    """Deterministic detector for pipeline unit tests."""

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detections


def make_pipeline(
    detections: list[Detection],
) -> RealtimePipeline:
    """Create a pipeline without loading a real YOLO model."""

    return RealtimePipeline(
        detector=FakeDetector(detections),
        context_engine=ContextEngine(),
        cue_policy=CuePolicy(),
    )


def test_pipeline_can_be_created_without_loading_model() -> None:
    pipeline = make_pipeline([])

    assert pipeline is not None


def test_blank_detection_result_stays_unknown() -> None:
    pipeline = make_pipeline([])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process(frame)

    assert result.detections == []
    assert result.scene_state.context == ContextType.UNKNOWN
    assert result.cue.action.value == "NO_AUTOMATIC_ACTION"
    assert result.cue.visual_cue == "NONE"
    assert result.cue.audio_cue == "NONE"
    assert result.cue.text_feedback == "NONE"


def test_obstacle_detection_without_distance_stays_unknown() -> None:
    detection = Detection(
        class_id=56,
        class_name="chair",
        confidence=0.90,
        x1=200,
        y1=200,
        x2=350,
        y2=400,
    )
    pipeline = make_pipeline([detection])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process(frame)

    assert result.detections == [detection]
    assert result.scene_state.obstacle_detected is True
    assert result.scene_state.obstacle_distance_m is None
    assert result.scene_state.context == ContextType.UNKNOWN
    assert result.cue.action.value == "NO_AUTOMATIC_ACTION"
    assert result.cue.visual_cue == "NONE"
    assert result.cue.audio_cue == "NONE"
    assert result.cue.text_feedback == "NONE"


def test_person_detection_does_not_imply_obstacle() -> None:
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        x1=200,
        y1=100,
        x2=350,
        y2=450,
    )
    pipeline = make_pipeline([detection])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process(frame)

    assert result.scene_state.person_detected is True
    assert result.scene_state.obstacle_detected is False
    assert result.scene_state.context == ContextType.UNKNOWN
    assert result.cue.action.value == "NO_AUTOMATIC_ACTION"


def test_unrelated_object_does_not_count_as_obstacle() -> None:
    detection = Detection(
        class_id=63,
        class_name="laptop",
        confidence=0.88,
        x1=220,
        y1=150,
        x2=400,
        y2=300,
    )
    pipeline = make_pipeline([detection])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process(frame)

    assert result.scene_state.obstacle_detected is False
    assert result.scene_state.context == ContextType.UNKNOWN
    assert result.cue.action.value == "NO_AUTOMATIC_ACTION"
def test_pipeline_records_value_estimate_without_claiming_route_safety():
    pipeline = make_pipeline([])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process(frame)

    assert result.scene_state.spatial_value is not None
    assert result.scene_state.spatial_value_confidence == 0.0
    assert result.scene_state.spatial_value_change is None
    assert result.scene_state.route_evidence_sufficient is False


def test_obstacle_detection_does_not_enable_route_guidance():
    detection = Detection(
        class_id=56,
        class_name="chair",
        confidence=0.90,
        x1=200,
        y1=200,
        x2=350,
        y2=400,
    )
    pipeline = make_pipeline([detection])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process(frame)

    assert result.scene_state.obstacle_detected is True
    assert result.scene_state.route_evidence_sufficient is False
    assert result.cue.action.value == "NO_AUTOMATIC_ACTION"


def test_pipeline_does_not_report_value_change_without_temporal_pair():
    pipeline = make_pipeline([])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process(frame)

    assert result.scene_state.spatial_value_change is None