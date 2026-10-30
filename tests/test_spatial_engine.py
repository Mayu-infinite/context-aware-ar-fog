import numpy as np
import pytest

from cafog.sources.synthetic import corridor_turn, doorway, run_scene, summarize_run
from cafog.spatial import EngineConfig, SpatialCueEngine, make_profile
from cafog.spatial.config import PROFILE_PRESETS


def summary(scene, profile):
    return summarize_run(run_scene(scene, EngineConfig(profile=make_profile(profile)), fps=8))


@pytest.mark.parametrize("profile", sorted(PROFILE_PRESETS))
def test_straight_corridor_never_cues(profile):
    s = summary(corridor_turn(0, before_m=3.0, after_m=3.0), profile)
    assert s["active_frames"] == 0


def test_freezer_cued_ahead_of_sharp_turn():
    s = summary(corridor_turn(90), "pd_fog")
    assert s["false_cue_frames"] == 0
    assert s["first_cue_distance_m"] is not None and s["first_cue_distance_m"] >= 1.5


def test_freezer_cued_ahead_of_narrow_door_but_not_wide_one():
    narrow = summary(doorway(0.8), "pd_fog")
    assert narrow["first_cue_distance_m"] is not None and narrow["first_cue_distance_m"] >= 1.5
    assert summary(doorway(1.2), "pd_fog")["active_frames"] == 0


def test_healthy_profile_never_cued_at_turn():
    assert summary(corridor_turn(120), "healthy")["active_frames"] == 0


def test_engine_latency_budget_and_outputs():
    from cafog.sources.synthetic import walk

    engine = SpatialCueEngine(EngineConfig(profile=make_profile("pd_fog")))
    totals = []
    for frame in walk(corridor_turn(90), fps=5):
        result = engine.process(frame)
        totals.append(result.timings_ms["total"])
        assert result.field.value_pd.shape == engine.map.shape
        for step in result.cue.footsteps:
            r, c = engine.map.world_to_cell(step.position)
            assert result.field.states[r, c] != 2
    assert np.median(totals) < 50.0


def test_config_round_trip_with_preset(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "profile:\n  preset: pd_nonfog\n  cue_step_length_m: 0.6\ncue:\n  on_risk: 0.5\n"
    )
    cfg = EngineConfig.from_yaml(path)
    assert cfg.profile.delta_lim == 0.008
    assert cfg.profile.cue_step_length_m == 0.6
    assert cfg.cue.on_risk == 0.5
    with pytest.raises(ValueError):
        EngineConfig.from_dict({"cue": {"not_a_key": 1}})
