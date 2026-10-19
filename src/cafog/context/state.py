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
    """Structured scene representation with explicit evidence limitations."""

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

    # Diagnostic value-estimation outputs; these do not certify safety.
    spatial_value: float | None = None
    spatial_value_confidence: float = 0.0
    spatial_value_change: float | None = None

    # False until route geometry and relevant scene evidence are assessed.
    route_evidence_sufficient: bool = False
