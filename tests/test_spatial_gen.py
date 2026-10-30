import numpy as np
import pytest
from spatial_helpers import straight_path

from cafog.spatial.config import ValueConfig, make_profile
from cafog.spatial.gen import assess_path, gen_speed, logsig, step_half_width, step_time_s


def test_gen_equations_match_paper_for_healthy():
    hc = make_profile("healthy")
    dv = np.array([0.0, 0.05, -0.05])
    v = gen_speed(dv, np.zeros(3), hc, 0.0)
    go = 3 * logsig(dv)
    nogo = 3 * logsig(-10 * dv)
    assert np.allclose(v, np.maximum(0, go - nogo))


def test_fog_factor_from_td_clamp():
    assert make_profile("healthy").fog_factor == 0.0
    assert make_profile("pd_nonfog").fog_factor == pytest.approx(1.25)
    assert make_profile("pd_fog").fog_factor == pytest.approx(10.0)


def test_healthy_profile_never_at_risk():
    risk = assess_path(
        straight_path(turn_deg_at=(2.0, 120)), make_profile("healthy"), ValueConfig()
    )
    assert np.all(risk.risk < 1e-9)


def test_straight_clear_path_is_low_risk_for_freezers():
    risk = assess_path(straight_path(), make_profile("pd_fog"), ValueConfig())
    assert risk.risk.max() < 0.05


def test_risk_grows_with_turn_angle_and_severity():
    cfg = ValueConfig()
    maxima = {}
    for profile in ("pd_nonfog", "pd_fog"):
        maxima[profile] = [
            assess_path(straight_path(turn_deg_at=(2.0, a)), make_profile(profile), cfg).risk.max()
            for a in (30, 60, 90, 120)
        ]
        assert all(np.diff(maxima[profile]) >= -1e-9)
    assert all(f >= n for f, n in zip(maxima["pd_fog"], maxima["pd_nonfog"], strict=True))
    assert maxima["pd_fog"][2] > 0.9


def test_value_canyon_produces_negative_value_gradient_before_turn():
    risk = assess_path(straight_path(turn_deg_at=(2.0, 90)), make_profile("pd_fog"), ValueConfig())
    before = risk.arc_m <= 2.0
    assert risk.dv_pd[before].min() < 0 < risk.dv_hc.min()


def test_narrow_passage_raises_risk():
    cfg = ValueConfig()
    wide = assess_path(straight_path(width=1.5), make_profile("pd_fog"), cfg).risk.max()
    narrow = assess_path(straight_path(width=0.75), make_profile("pd_fog"), cfg).risk.max()
    assert narrow > wide + 0.5


def test_lip_step_time_and_width_compensation():
    p = make_profile("pd_fog")
    assert 0.25 <= step_time_s(0.8, p) <= 0.9
    assert step_time_s(0.01, p) == 0.9
    assert step_half_width(np.array(0.0), p) == pytest.approx(p.step_half_width_m)
    assert step_half_width(np.array(np.pi / 2), p) == pytest.approx(2 * p.step_half_width_m)
