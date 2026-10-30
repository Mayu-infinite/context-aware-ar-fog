import numpy as np
from spatial_helpers import straight_path

from cafog.spatial.config import CueConfig, ValueConfig, make_profile
from cafog.spatial.cues import CueController, CuePhase, plan_footsteps
from cafog.spatial.gen import assess_path


def run(controller, path, times, user_z=0.0, speed=0.8):
    risk = assess_path(path, make_profile("pd_fog"), ValueConfig())
    out = None
    for t in times:
        out = controller.update(risk, path, np.array([0.0, user_z]), speed, t)
    return out


def test_footsteps_alternate_and_follow_path():
    path = straight_path()
    steps = plan_footsteps(path, 3.0, make_profile("pd_fog"), CueConfig())
    assert len(steps) >= 4
    sides = [s.side for s in steps]
    assert all(a == -b for a, b in zip(sides, sides[1:], strict=False))
    gaps = np.diff([s.position[1] for s in steps])
    assert np.allclose(gaps, make_profile("pd_fog").cue_step_length_m, atol=1e-6)
    assert all(abs(s.position[0]) > 0.05 for s in steps)


def test_footsteps_stop_at_unknown_floor():
    path = straight_path()
    path.known_free[path.arc_m > 1.2] = False
    steps = plan_footsteps(path, 4.0, make_profile("pd_fog"), CueConfig())
    assert steps and max(s.position[1] for s in steps) <= 1.2


def test_cue_requires_sustained_risk_then_activates():
    ctrl = CueController(CueConfig(on_hold_s=0.2), make_profile("pd_fog"))
    path = straight_path(turn_deg_at=(2.0, 90))
    assert run(ctrl, path, [0.0]).phase == CuePhase.ARMING
    out = run(ctrl, path, [0.1, 0.25])
    assert out.phase == CuePhase.ACTIVE
    assert out.footsteps and out.metronome_bpm and out.hazard is not None


def test_no_cue_on_clear_straight_path():
    ctrl = CueController(CueConfig(), make_profile("pd_fog"))
    out = run(ctrl, straight_path(), np.arange(0, 3, 0.1))
    assert out.phase == CuePhase.IDLE and not out.footsteps


def test_cue_releases_after_hold_once_demand_resolves():
    cfg = CueConfig(on_hold_s=0.0, min_active_s=0.5, off_hold_s=0.5)
    ctrl = CueController(cfg, make_profile("pd_fog"))
    run(ctrl, straight_path(turn_deg_at=(2.0, 90)), [0.0, 0.1])
    assert ctrl.phase == CuePhase.ACTIVE
    out = run(ctrl, straight_path(), [1.0, 1.2])
    assert out.phase == CuePhase.RELEASING
    out = run(ctrl, straight_path(), [1.6, 1.8])
    assert out.phase == CuePhase.IDLE


def test_audio_and_visual_can_be_disabled_per_patient():
    ctrl = CueController(CueConfig(on_hold_s=0.0, audio_enabled=False), make_profile("pd_fog"))
    out = run(ctrl, straight_path(turn_deg_at=(2.0, 90)), [0.0, 0.1])
    assert out.active and out.metronome_bpm is None and out.footsteps
