from cafog.perception.detector import Detection
from cafog.perception.geometry import (
    horizontal_angle_from_center,
)


def test_center_object_has_zero_angle() -> None:
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=400,
        y1=100,
        x2=600,
        y2=500,
    )

    angle = horizontal_angle_from_center(
        detection,
        image_width=1000,
        horizontal_fov_deg=90,
    )

    assert abs(angle) < 0.001