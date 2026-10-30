from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from cafog.spatial.config import CueConfig, PatientProfile
from cafog.spatial.gen import RiskProfile, step_half_width, step_time_s
from cafog.spatial.geometry import right_of
from cafog.spatial.path import PredictedPath


class CuePhase(StrEnum):
    IDLE = "IDLE"
    ARMING = "ARMING"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"


class HazardKind(StrEnum):
    TURN = "TURN"
    NARROW = "NARROW"
    INITIATION = "INITIATION"


@dataclass(frozen=True)
class Footstep:
    position: np.ndarray
    yaw_rad: float
    side: int
    index: int


@dataclass
class Hazard:
    kind: HazardKind
    distance_m: float
    risk: float
    turn_deg: float
    width_m: float


@dataclass
class CueOutput:
    phase: CuePhase
    footsteps: list[Footstep] = field(default_factory=list)
    metronome_bpm: float | None = None
    hazard: Hazard | None = None
    reason: str = ""

    @property
    def active(self) -> bool:
        return self.phase in (CuePhase.ACTIVE, CuePhase.RELEASING)


def plan_footsteps(
    path: PredictedPath,
    end_s: float,
    profile: PatientProfile,
    cfg: CueConfig,
    first_side: int = -1,
) -> list[Footstep]:
    """Alternating footsteps along the path with turn-widened step width (paper Eq. 26)."""
    stride = profile.cue_step_length_m
    end_s = min(end_s, path.length_m)
    arcs = np.arange(cfg.first_step_offset_m, end_s + 1e-9, stride)[: cfg.max_footsteps]
    steps: list[Footstep] = []
    side = first_side
    for i, s in enumerate(arcs):
        k = int(np.clip(np.searchsorted(path.arc_m, s), 0, len(path.arc_m) - 1))
        if not path.known_free[k] or path.clearance_m[k] < cfg.footstep_min_clearance_m:
            break
        tangent = path.tangent_at(s)
        ahead = path.tangent_at(s + stride)
        dpsi = np.arctan2(tangent[0] * ahead[1] - tangent[1] * ahead[0], float(tangent @ ahead))
        half_width = float(step_half_width(dpsi, profile))
        pos = path.point_at(s) + side * half_width * right_of(tangent)
        steps.append(
            Footstep(
                position=pos, yaw_rad=float(np.arctan2(tangent[1], tangent[0])), side=side, index=i
            )
        )
        side = -side
    return steps


def footsteps_deviation(steps: list[Footstep], path: PredictedPath) -> float:
    if not steps:
        return np.inf
    pts = np.array([s.position for s in steps])
    d = np.linalg.norm(pts[:, None, :] - path.points[None, :, :], axis=-1).min(axis=1)
    return float(np.mean(d))


class CueController:
    """Turns a risk profile into a stable, minimal cue (footsteps, optional metronome)."""

    def __init__(self, cfg: CueConfig, profile: PatientProfile):
        self.cfg = cfg
        self.profile = profile
        self.phase = CuePhase.IDLE
        self.footsteps: list[Footstep] = []
        self._since: float | None = None
        self._activated_at = 0.0
        self._deviating_since: float | None = None
        self._hazard: Hazard | None = None

    def reset(self) -> None:
        self.__init__(self.cfg, self.profile)

    def _detect(self, risk: RiskProfile, path: PredictedPath, speed: float) -> Hazard | None:
        cfg = self.cfg
        stationary = speed < cfg.stationary_speed_mps
        reach = cfg.initiation_distance_m if stationary else cfg.trigger_distance_m
        k = risk.first_above(cfg.on_risk, reach)
        if k is None:
            return None
        s = float(risk.arc_m[k])
        region = (risk.arc_m >= s) & (risk.arc_m <= s + cfg.extend_past_hazard_m)
        kind = (
            HazardKind.TURN
            if risk.turn_ratio[region].max() >= risk.narrow_ratio[region].max()
            else HazardKind.NARROW
        )
        if stationary:
            kind = HazardKind.INITIATION
        on_path = (path.arc_m >= s - 0.5) & (path.arc_m <= s + cfg.extend_past_hazard_m)
        return Hazard(
            kind,
            s,
            float(risk.risk[region].max()),
            float(path.turn_deg[on_path].max()),
            float(path.width_m[on_path].min()),
        )

    def update(
        self,
        risk: RiskProfile | None,
        path: PredictedPath | None,
        user_xz: np.ndarray,
        speed_mps: float,
        t: float,
    ) -> CueOutput:
        cfg = self.cfg
        if risk is None or path is None:
            return self._release(t, "no traversable route ahead", force=True)

        hazard = self._detect(risk, path, speed_mps)
        reach = cfg.trigger_distance_m + cfg.extend_past_hazard_m
        level = risk.max_within(reach)

        if self.phase in (CuePhase.IDLE, CuePhase.ARMING):
            if hazard is None:
                self.phase, self._since = CuePhase.IDLE, None
                return CueOutput(self.phase, reason="spatial demand within tolerance")
            if self.phase == CuePhase.IDLE:
                self.phase, self._since = CuePhase.ARMING, t
            if t - self._since >= cfg.on_hold_s:
                self._activate(path, hazard, t)
            return self._output(f"{hazard.kind.value} risk {hazard.risk:.2f}")

        self._hazard = hazard or self._hazard
        self._prune(user_xz)
        self._maybe_replan(path, t)

        if level <= cfg.off_risk and t - self._activated_at >= cfg.min_active_s:
            if self.phase != CuePhase.RELEASING:
                self.phase, self._since = CuePhase.RELEASING, t
            elif t - self._since >= cfg.off_hold_s:
                return self._release(t, "demand resolved")
        elif self.phase == CuePhase.RELEASING:
            self.phase = CuePhase.ACTIVE

        if not self.footsteps:
            self.footsteps = self._plan(path)
        return self._output("guiding through high-demand space")

    def _activate(self, path: PredictedPath, hazard: Hazard, t: float) -> None:
        self.phase = CuePhase.ACTIVE
        self._activated_at = t
        self._hazard = hazard
        self.footsteps = self._plan(path)

    def _plan(self, path: PredictedPath) -> list[Footstep]:
        end = (self._hazard.distance_m if self._hazard else 0.0) + self.cfg.extend_past_hazard_m
        return plan_footsteps(path, end, self.profile, self.cfg)

    def _prune(self, user_xz: np.ndarray) -> None:
        kept = []
        for step in self.footsteps:
            heading = np.array([np.cos(step.yaw_rad), np.sin(step.yaw_rad)])
            if float((step.position - user_xz) @ heading) > 0.1:
                kept.append(step)
        self.footsteps = kept

    def _maybe_replan(self, path: PredictedPath, t: float) -> None:
        if footsteps_deviation(self.footsteps, path) <= self.cfg.replan_deviation_m:
            self._deviating_since = None
            return
        if self._deviating_since is None:
            self._deviating_since = t
        elif t - self._deviating_since >= self.cfg.replan_hold_s:
            self.footsteps = self._plan(path)
            self._deviating_since = None

    def _release(self, t: float, reason: str, force: bool = False) -> CueOutput:
        if (
            force
            and self.phase == CuePhase.ACTIVE
            and t - self._activated_at < self.cfg.min_active_s
        ):
            return self._output("holding cue: route temporarily lost")
        self.phase = CuePhase.IDLE
        self.footsteps = []
        self._since = None
        self._hazard = None
        return CueOutput(self.phase, reason=reason)

    def _output(self, reason: str) -> CueOutput:
        active = self.phase in (CuePhase.ACTIVE, CuePhase.RELEASING)
        bpm = None
        if active and self.cfg.audio_enabled:
            bpm = 60.0 / step_time_s(self.profile.cue_speed_mps, self.profile)
        return CueOutput(
            phase=self.phase,
            footsteps=list(self.footsteps) if active and self.cfg.visual_enabled else [],
            metronome_bpm=bpm,
            hazard=self._hazard if active else None,
            reason=reason,
        )
