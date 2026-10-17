from cafog.context.state import ContextType
from cafog.policy.cue_policy import CuePolicy


def test_open_path_has_no_text_feedback():
    policy = CuePolicy()

    cue = policy.select(ContextType.OPEN_PATH)

    assert cue.text_feedback == "NONE"
    assert cue.action.value == "CONTINUE"


def test_obstacle_has_cautious_text_feedback():
    policy = CuePolicy()

    cue = policy.select(ContextType.OBSTACLE)

    assert cue.text_feedback == (
        "An obstacle may be ahead. Proceed carefully."
    )


def test_unknown_has_no_directional_feedback():
    policy = CuePolicy()

    cue = policy.select(ContextType.UNKNOWN)

    assert cue.text_feedback == "NONE"
    assert cue.action.value == "NO_AUTOMATIC_ACTION"
    assert cue.visual_cue == "NONE"
    assert cue.audio_cue == "NONE"
