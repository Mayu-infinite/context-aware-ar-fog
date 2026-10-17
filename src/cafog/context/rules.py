
from __future__ import annotations

from cafog.context.state import ContextType, SceneState


class ContextEngine:
    """Classify scene context only when the available evidence supports it.

    Important:
    - Distance and width values must be calibrated measurements in meters.
    - Missing detections do not establish that a path is clear.
    - This rule engine is a research prototype, not a validated mobility aid.
    """

    def __init__(
        self,
        max_doorway_width_m: float = 0.75,
        doorway_trigger_distance_m: float = 1.50,
        obstacle_trigger_distance_m: float = 1.20,
        minimum_turn_angle_deg: float = 30.0,
        minimum_path_width_m: float = 0.75,
    ) -> None:
        self.max_doorway_width_m = max_doorway_width_m
        self.doorway_trigger_distance_m = doorway_trigger_distance_m
        self.obstacle_trigger_distance_m = obstacle_trigger_distance_m
        self.minimum_turn_angle_deg = minimum_turn_angle_deg
        self.minimum_path_width_m = minimum_path_width_m

    def classify(self, state: SceneState) -> ContextType:
        """Classify the scene, falling back to UNKNOWN when evidence is insufficient."""

        # A nearby, measured narrow doorway is the most specific context.
        if (
            state.door_detected
            and state.door_width_m is not None
            and state.door_distance_m is not None
            and 0.0 < state.door_width_m <= self.max_doorway_width_m
            and 0.0 <= state.door_distance_m <= self.doorway_trigger_distance_m
        ):
            state.context = ContextType.NARROW_DOORWAY
            return state.context

        # Do not infer obstacle distance from detection confidence or relative depth.
        if (
            state.obstacle_detected
            and state.obstacle_distance_m is not None
            and 0.0 <= state.obstacle_distance_m <= self.obstacle_trigger_distance_m
        ):
            # A doorway plus a nearby obstacle may make the scene more complex.
            if state.door_detected:
                state.context = ContextType.COMPLEX_SPACE
            else:
                state.context = ContextType.OBSTACLE
            return state.context

        # Turns require an explicit, meaningful geometric estimate.
        if state.turn_angle_deg is not None:
            if state.turn_angle_deg <= -self.minimum_turn_angle_deg:
                state.context = ContextType.TURN_LEFT
                return state.context

            if state.turn_angle_deg >= self.minimum_turn_angle_deg:
                state.context = ContextType.TURN_RIGHT
                return state.context

        # A measured path width alone does not prove that the path is safe.
        # Keep unrecognized or incompletely measured scenes uncertain.
        state.context = ContextType.UNKNOWN
        return state.context
