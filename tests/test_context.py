
from cafog.context.rules import ContextEngine
from cafog.context.state import ContextType, SceneState


def test_narrow_doorway_context() -> None:
    engine = ContextEngine()
    state = SceneState(
        door_detected=True,
        door_distance_m=1.0,
        door_width_m=0.60,
    )

    assert engine.classify(state) == ContextType.NARROW_DOORWAY


def test_empty_scene_is_unknown_not_open_path() -> None:
    engine = ContextEngine()
    state = SceneState()

    assert engine.classify(state) == ContextType.UNKNOWN


def test_obstacle_context_with_measured_distance() -> None:
    engine = ContextEngine()
    state = SceneState(
        obstacle_detected=True,
        obstacle_distance_m=0.8,
    )

    assert engine.classify(state) == ContextType.OBSTACLE


def test_obstacle_without_distance_is_unknown() -> None:
    engine = ContextEngine()
    state = SceneState(obstacle_detected=True)

    assert engine.classify(state) == ContextType.UNKNOWN


def test_doorway_outside_trigger_distance_is_unknown() -> None:
    engine = ContextEngine()
    state = SceneState(
        door_detected=True,
        door_distance_m=2.0,
        door_width_m=0.60,
    )

    assert engine.classify(state) == ContextType.UNKNOWN


def test_turn_requires_explicit_angle() -> None:
    engine = ContextEngine()
    state = SceneState(turn_angle_deg=-35.0)

    assert engine.classify(state) == ContextType.TURN_LEFT


def test_small_turn_angle_is_unknown() -> None:
    engine = ContextEngine()
    state = SceneState(turn_angle_deg=15.0)

    assert engine.classify(state) == ContextType.UNKNOWN


def test_nearby_obstacle_and_doorway_is_complex_space() -> None:
    engine = ContextEngine()
    state = SceneState(
        door_detected=True,
        door_distance_m=1.0,
        door_width_m=0.90,
        obstacle_detected=True,
        obstacle_distance_m=0.8,
    )

    assert engine.classify(state) == ContextType.COMPLEX_SPACE
