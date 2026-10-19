
import pytest

from cafog.context.state import ContextType
from cafog.planning.value_field import ValueEstimate
from cafog.policy.actions import ActionType
from cafog.policy.cue_policy import CuePolicy


def test_open_path_has_no_text_feedback():
    cue = CuePolicy().select(ContextType.OPEN_PATH)

    assert cue.text_feedback == "NONE"
    assert cue.action == ActionType.CONTINUE


def test_obstacle_has_cautious_text_feedback():
    cue = CuePolicy().select(ContextType.OBSTACLE)

    assert cue.text_feedback == (
        "An obstacle may be ahead. Proceed carefully."
    )


def test_unknown_has_no_directional_feedback():
    cue = CuePolicy().select(ContextType.UNKNOWN)

    assert cue.text_feedback == "NONE"
    assert cue.action == ActionType.NO_AUTOMATIC_ACTION
    assert cue.visual_cue == "NONE"
    assert cue.audio_cue == "NONE"


def test_insufficient_evidence_suppresses_automatic_action():
    cue = CuePolicy().select(
        ContextType.TURN_LEFT,
        evidence_sufficient=False,
    )

    assert cue.action == ActionType.NO_AUTOMATIC_ACTION
    assert cue.visual_cue == "NONE"
    assert cue.audio_cue == "NONE"


def test_low_confidence_value_estimate_suppresses_action():
    estimate = ValueEstimate(
        value=0.8,
        confidence=0.1,
        available_terms=("uncertainty",),
    )

    cue = CuePolicy(minimum_value_confidence=0.25).select(
        ContextType.TURN_RIGHT,
        value_estimate=estimate,
    )

    assert cue.action == ActionType.NO_AUTOMATIC_ACTION
    assert cue.visual_cue == "NONE"
    assert cue.audio_cue == "NONE"


def test_negative_value_change_alone_does_not_create_directional_cue():
    cue = CuePolicy().select(
        ContextType.OPEN_PATH,
        value_change=-0.5,
    )

    # Preserve OPEN_PATH's configured behavior.
    assert cue.action == ActionType.CONTINUE
    assert cue.visual_cue == "NONE"
    assert cue.audio_cue == "NONE"


def test_high_confidence_estimate_preserves_configured_open_path_behavior():
    estimate = ValueEstimate(
        value=0.7,
        confidence=0.9,
        available_terms=("clearance", "uncertainty"),
    )

    cue = CuePolicy().select(
        ContextType.OPEN_PATH,
        value_estimate=estimate,
    )

    assert cue.action == ActionType.CONTINUE
    assert cue.text_feedback == "NONE"


@pytest.mark.parametrize(
    "confidence",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.1,
        1.1,
    ],
)
def test_invalid_value_confidence_suppresses_cue(confidence):
    estimate = ValueEstimate(
        value=0.8,
        confidence=confidence,
        available_terms=("clearance",),
    )

    cue = CuePolicy().select(
        ContextType.OPEN_PATH,
        value_estimate=estimate,
        evidence_sufficient=True,
    )

    assert cue.action == ActionType.NO_AUTOMATIC_ACTION
    assert cue.visual_cue == "NONE"
    assert cue.audio_cue == "NONE"


def test_non_finite_minimum_confidence_is_rejected():
    with pytest.raises(ValueError):
        CuePolicy(minimum_value_confidence=float("nan"))
