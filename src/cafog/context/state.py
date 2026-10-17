
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContextType(StrEnum):
    """High-level environmental contexts."""

    OPEN_PATH = "OPEN_PATH"
    NARROW_DOORWAY = "NARROW_DOORWAY"
    OBSTACLE = "OBSTACLE"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    COMPLEX_SPACE = "COMPLEX_SPACE"
    UNKNOWN = "UNKNOWN"


@dataclass
class SceneState:
    """Structured representation of the environment.

    This object is intentionally separate from raw object detection.
    """

    door_detected: bool = False
    door_distance_m: float | None = None
    door_width_m: float | None = None

    obstacle_detected: bool = False
    obstacle_distance_m: float | None = None

    path_width_m: float | None = None
    turn_angle_deg: float | None = None
    walkable_path: str | None = None

    person_detected: bool = False
    confidence: float = 0.0
    context: ContextType = ContextType.UNKNOWN
