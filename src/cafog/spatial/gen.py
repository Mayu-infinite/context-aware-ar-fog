"""Go/Explore/No-go action selection from Bhattacharya et al. (2026), evaluated along a path.

The paper drives speed with the value difference dV (Eq. 8):
    G = A_G * logsig(l_go * (dV - th_go))                       (Eq. 9)
    N = A_N * logsig(l_nogo * (dV - th_nogo))                   (Eq. 10)
    v_target = max(0, G - N)                                     (Eq. 15, test phase: no E)
and scales the pathways with disease severity and environmental difficulty:
    FOG_factor = delta_lim_ref / delta_lim                       (Eq. 28)
    A_G,cond   = A_G (1 - 0.25 FOG_factor r)                     (Eq. 30)
    A_N,cond   = A_N (1 + 0.25 FOG_factor r)                     (Eq. 31)
    th_nogo    = 0.15 FOG_factor r                               (Eq. 32)
where the paper's angle ratio (Eq. 29) is generalised to a local demand ratio r that also
covers narrow passages. dV is expressed in units of "value gained per metre on a clear
path", so the No-go slope is rescaled from -100 to -10 for those units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cafog.spatial.config import PatientProfile, ValueConfig
from cafog.spatial.path import PredictedPath
from cafog.spatial.value import value_canyon_kappa


def logsig(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


@dataclass
class RiskProfile:
    arc_m: np.ndarray
    demand: np.ndarray
    value_hc: np.ndarray
    value_pd: np.ndarray
    dv_hc: np.ndarray
    dv_pd: np.ndarray
    speed_hc: np.ndarray
    speed_pd: np.ndarray
    risk: np.ndarray
    turn_ratio: np.ndarray
    narrow_ratio: np.ndarray

    def first_above(self, threshold: float, max_distance_m: float) -> int | None:
        idx = np.flatnonzero((self.risk >= threshold) & (self.arc_m <= max_distance_m))
        return int(idx[0]) if idx.size else None

    def max_within(self, max_distance_m: float) -> float:
        mask = self.arc_m <= max_distance_m
        return float(self.risk[mask].max()) if mask.any() else 0.0


def gen_speed(dv: np.ndarray, demand: np.ndarray, profile: PatientProfile, fog_factor: float):
    severity = fog_factor * np.asarray(demand)
    a_go = profile.go_amplitude * np.maximum(0.0, 1.0 - profile.go_scaling * severity)
    a_nogo = profile.nogo_amplitude * (1.0 + profile.nogo_scaling * severity)
    th_nogo = profile.nogo_threshold_shift * severity
    go = a_go * logsig(profile.go_slope * (dv - profile.go_threshold))
    nogo = a_nogo * logsig(profile.nogo_slope * (dv - th_nogo))
    return np.maximum(0.0, go - nogo)


def assess_path(path: PredictedPath, profile: PatientProfile, cfg: ValueConfig) -> RiskProfile:
    """Predict where a patient with `profile` would slow or stop along the path."""
    step = profile.gen_step_m
    arc = np.arange(0.0, path.length_m + 1e-9, step)
    if arc.size < 2:
        arc = np.array([0.0, path.length_m])

    cost = np.interp(arc, path.arc_m, np.where(np.isfinite(path.cost), path.cost, 50.0))
    turn = np.interp(arc, path.arc_m, path.turn_ratio)
    narrow = np.interp(arc, path.arc_m, path.narrow_ratio)
    demand = np.maximum(turn, narrow)

    # Cost-to-go along the path, anchored at the far end so it is monotone by construction.
    end_ctg = path.cost_to_go[-1] if np.isfinite(path.cost_to_go[-1]) else 0.0
    seg = 0.5 * (cost[1:] + cost[:-1]) * np.diff(arc)
    ctg = end_ctg + np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

    log_gamma = math.log(cfg.discount_per_m)
    ln_v_hc = log_gamma * ctg
    ln_v_pd = ln_v_hc - value_canyon_kappa(profile) * demand

    unit = -log_gamma * np.maximum(np.diff(arc), 1e-6)
    dv_hc = np.diff(ln_v_hc) / unit
    dv_pd = np.diff(ln_v_pd) / unit
    local_demand = np.maximum(demand[1:], demand[:-1])

    speed_hc = gen_speed(dv_hc, np.zeros_like(local_demand), profile, 0.0)
    speed_pd = gen_speed(dv_pd, local_demand, profile, profile.fog_factor)
    risk = np.clip(1.0 - speed_pd / np.maximum(speed_hc, 1e-6), 0.0, 1.0)

    return RiskProfile(
        arc_m=arc[1:],
        demand=local_demand,
        value_hc=np.exp(ln_v_hc[1:]),
        value_pd=np.exp(ln_v_pd[1:]),
        dv_hc=dv_hc,
        dv_pd=dv_pd,
        speed_hc=speed_hc,
        speed_pd=speed_pd,
        risk=risk,
        turn_ratio=np.maximum(turn[1:], turn[:-1]),
        narrow_ratio=np.maximum(narrow[1:], narrow[:-1]),
    )


def step_time_s(speed_mps: float, profile: PatientProfile) -> float:
    """LIP step time (Eq. 25)."""
    return float(min(max(profile.lip_base_period_s / max(speed_mps, 1e-3) ** 0.36, 0.25), 0.9))


def step_half_width(turn_rate: np.ndarray, profile: PatientProfile) -> np.ndarray:
    """Turn-compensated foot placement half-width (Eq. 26), lateral share = |sin(dpsi)|."""
    return profile.step_half_width_m * (1.0 + np.abs(np.sin(np.asarray(turn_rate))))
