from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MapConfig:
    """Local, world-anchored 2.5D traversability grid around the user."""

    cell_size_m: float = 0.10
    size_m: float = 10.0
    recenter_distance_m: float = 2.0
    max_range_m: float = 4.5
    floor_tolerance_m: float = 0.05
    obstacle_min_height_m: float = 0.08
    obstacle_max_height_m: float = 1.90
    dropoff_depth_m: float = 0.12
    min_obstacle_points: int = 2
    min_floor_points: int = 1
    log_odds_hit: float = 0.9
    log_odds_free: float = -0.45
    log_odds_min: float = -2.0
    log_odds_max: float = 3.0
    occupied_threshold: float = 0.6
    free_threshold: float = -0.3
    decay_time_constant_s: float = 20.0
    initial_eye_height_m: float = 1.6
    floor_smoothing: float = 0.15

    @property
    def cells(self) -> int:
        return int(round(self.size_m / self.cell_size_m))


@dataclass
class ValueConfig:
    """Training-free critic: discounted geodesic value over the traversability grid."""

    discount_per_m: float = 0.85
    body_radius_m: float = 0.25
    narrow_cost_gain: float = 2.0
    narrow_cost_length_m: float = 0.20
    unknown_cell_cost: float = 1.0
    goal_min_distance_m: float = 1.0
    goal_preferred_distance_m: float = 4.5
    goal_near_penalty: float = 2.0
    goal_alignment_cost_m: float = 3.0
    goal_max_angle_deg: float = 135.0
    lookahead_m: float = 4.0
    path_smoothing_window: int = 5
    path_sample_step_m: float = 0.10
    turn_window_m: float = 1.0
    turn_deadband_deg: float = 15.0
    width_ray_max_m: float = 1.5
    comfort_width_m: float = 1.1
    min_width_m: float = 0.6
    field_turn_probe_cells: int = 5


@dataclass
class PatientProfile:
    """Parameters of the Go/Explore/No-go (GEN) model from Bhattacharya et al. (2026).

    delta_lim=None means an unclamped TD error (healthy control). The paper uses
    0.008 for PD non-FOG and 0.001 for PD-FOG with a reference clamp of 0.01.
    """

    name: str = "pd_fog"
    delta_lim: float | None = 0.001
    delta_lim_ref: float = 0.01
    go_amplitude: float = 3.0
    nogo_amplitude: float = 3.0
    go_slope: float = 1.0
    nogo_slope: float = -10.0
    go_threshold: float = 0.0
    go_scaling: float = 0.25
    nogo_scaling: float = 0.25
    nogo_threshold_shift: float = 0.15
    value_canyon_gain: float = 0.08
    gen_step_m: float = 0.5
    cue_step_length_m: float = 0.55
    step_half_width_m: float = 0.10
    cue_speed_mps: float = 0.8
    lip_base_period_s: float = 0.5

    @property
    def fog_factor(self) -> float:
        if self.delta_lim is None:
            return 0.0
        return self.delta_lim_ref / self.delta_lim


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "healthy": {"name": "healthy", "delta_lim": None},
    "pd_nonfog": {"name": "pd_nonfog", "delta_lim": 0.008},
    "pd_fog": {"name": "pd_fog", "delta_lim": 0.001},
}


def make_profile(preset: str, **overrides: Any) -> PatientProfile:
    if preset not in PROFILE_PRESETS:
        raise ValueError(f"Unknown profile preset {preset!r}; use one of {sorted(PROFILE_PRESETS)}")
    return PatientProfile(**{**PROFILE_PRESETS[preset], **overrides})


@dataclass
class CueConfig:
    """Hysteresis and footstep-placement settings for the cue controller."""

    on_risk: float = 0.6
    off_risk: float = 0.35
    on_hold_s: float = 0.2
    off_hold_s: float = 1.0
    min_active_s: float = 1.5
    trigger_distance_m: float = 3.0
    initiation_distance_m: float = 1.5
    stationary_speed_mps: float = 0.12
    extend_past_hazard_m: float = 1.5
    first_step_offset_m: float = 0.35
    max_footsteps: int = 8
    replan_deviation_m: float = 0.35
    replan_hold_s: float = 0.5
    footstep_min_clearance_m: float = 0.15
    audio_enabled: bool = True
    visual_enabled: bool = True


@dataclass
class EngineConfig:
    map: MapConfig = field(default_factory=MapConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    profile: PatientProfile = field(default_factory=PatientProfile)
    cue: CueConfig = field(default_factory=CueConfig)
    depth_stride: int = 2
    heading_speed_threshold_mps: float = 0.25
    heading_time_constant_s: float = 0.5
    velocity_time_constant_s: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        data = dict(data or {})
        profile_data = dict(data.pop("profile", {}) or {})
        preset = profile_data.pop("preset", None)
        profile = (
            make_profile(preset, **profile_data) if preset else _build(PatientProfile, profile_data)
        )
        return cls(
            map=_build(MapConfig, data.pop("map", {})),
            value=_build(ValueConfig, data.pop("value", {})),
            profile=profile,
            cue=_build(CueConfig, data.pop("cue", {})),
            **data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


def _build(kind: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    known = {f.name for f in fields(kind)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {kind.__name__} keys: {sorted(unknown)}")
    instance = kind(**values)
    _validate(instance)
    return instance


def _validate(instance: Any) -> None:
    if not is_dataclass(instance):
        return
    for f in fields(instance):
        value = getattr(instance, f.name)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{type(instance).__name__}.{f.name} must be finite.")
