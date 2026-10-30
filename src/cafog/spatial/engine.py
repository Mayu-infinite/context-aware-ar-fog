from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from cafog.spatial.config import EngineConfig
from cafog.spatial.cues import CueController, CueOutput
from cafog.spatial.gen import RiskProfile, assess_path
from cafog.spatial.geometry import CameraIntrinsics, depth_to_world_points, ground_heading
from cafog.spatial.occupancy import CellCounts, TraversabilityMap
from cafog.spatial.path import PredictedPath, predict_path
from cafog.spatial.value import ValueField, compute_value_field


@dataclass
class DepthFrame:
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    camera_to_world: np.ndarray
    timestamp_s: float
    color: np.ndarray | None = None


@dataclass
class EngineResult:
    timestamp_s: float
    user_xz: np.ndarray
    heading_xz: np.ndarray
    speed_mps: float
    field: ValueField
    path: PredictedPath | None
    risk: RiskProfile | None
    cue: CueOutput
    points_world: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)


class EgoMotion:
    """Smoothed walking direction and speed from the head pose."""

    def __init__(self, speed_threshold: float, heading_tau: float, velocity_tau: float):
        self.speed_threshold = speed_threshold
        self.heading_tau = heading_tau
        self.velocity_tau = velocity_tau
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2)
        self.heading = np.array([0.0, 1.0])
        self._t: float | None = None

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def update(self, position_xz: np.ndarray, gaze_xz: np.ndarray | None, t: float) -> None:
        position_xz = np.asarray(position_xz, dtype=np.float64)
        if self.position is None or self._t is None:
            self.position, self._t = position_xz, t
            if gaze_xz is not None:
                self.heading = gaze_xz
            return
        dt = t - self._t
        if dt <= 0:
            return
        a_v = 1.0 - math.exp(-dt / self.velocity_tau)
        self.velocity = (1 - a_v) * self.velocity + a_v * (position_xz - self.position) / dt
        self.position, self._t = position_xz, t

        target = self.velocity / self.speed if self.speed >= self.speed_threshold else gaze_xz
        if target is not None:
            a_h = 1.0 - math.exp(-dt / self.heading_tau)
            blended = (1 - a_h) * self.heading + a_h * target
            norm = np.linalg.norm(blended)
            if norm > 1e-6:
                self.heading = blended / norm


class SpatialCueEngine:
    """Depth + head pose in, value heatmap + footstep cue out. No labels, no training."""

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.map: TraversabilityMap | None = None
        self.ego = EgoMotion(
            self.config.heading_speed_threshold_mps,
            self.config.heading_time_constant_s,
            self.config.velocity_time_constant_s,
        )
        self.cues = CueController(self.config.cue, self.config.profile)

    def reset(self) -> None:
        self.__init__(self.config)

    def process(self, frame: DepthFrame) -> EngineResult:
        t0 = time.perf_counter()
        pose = np.asarray(frame.camera_to_world, dtype=np.float64)
        eye = pose[:3, 3]
        points = depth_to_world_points(
            frame.depth_m,
            frame.intrinsics,
            pose,
            stride=self.config.depth_stride,
            max_range_m=self.config.map.max_range_m,
        )
        t1 = time.perf_counter()
        user_xz = eye[[0, 2]]
        if self.map is None:
            self.map = TraversabilityMap(self.config.map, center_xz=tuple(user_xz))
        self.map.recenter(user_xz)
        self.map.integrate_points(points, float(eye[1]), frame.timestamp_s)
        t2 = time.perf_counter()
        return self._plan(frame.timestamp_s, user_xz, ground_heading(pose), points, t0, t1, t2)

    def process_counts(
        self,
        counts: CellCounts,
        user_xz: np.ndarray,
        gaze_xz: np.ndarray | None,
        timestamp_s: float,
    ) -> EngineResult:
        """Entry point when another component (e.g. the Quest GPU pass) already binned depth.

        Counts must be binned against the current `self.map.origin`; the grid recentres
        afterwards, so read the origin again before binning the next frame.
        """
        t0 = time.perf_counter()
        if self.map is None:
            self.map = TraversabilityMap(self.config.map, center_xz=tuple(user_xz))
        self.map.integrate_counts(counts, timestamp_s)
        self.map.recenter(np.asarray(user_xz))
        t2 = time.perf_counter()
        return self._plan(timestamp_s, np.asarray(user_xz), gaze_xz, np.zeros((0, 3)), t0, t0, t2)

    def _plan(self, t, user_xz, gaze_xz, points, t0, t1, t2) -> EngineResult:
        cfg = self.config
        self.ego.update(user_xz, gaze_xz, t)
        heading = self.ego.heading
        field = compute_value_field(self.map, user_xz, heading, cfg.value, cfg.profile)
        t3 = time.perf_counter()
        path = predict_path(self.map, field, user_xz, heading, cfg.value)
        risk = assess_path(path, cfg.profile, cfg.value) if path is not None else None
        t4 = time.perf_counter()
        cue = self.cues.update(risk, path, user_xz, self.ego.speed, t)
        t5 = time.perf_counter()
        return EngineResult(
            timestamp_s=t,
            user_xz=np.asarray(user_xz, dtype=np.float64),
            heading_xz=heading.copy(),
            speed_mps=self.ego.speed,
            field=field,
            path=path,
            risk=risk,
            cue=cue,
            points_world=points,
            timings_ms={
                "backproject": 1e3 * (t1 - t0),
                "mapping": 1e3 * (t2 - t1),
                "value_field": 1e3 * (t3 - t2),
                "path_and_risk": 1e3 * (t4 - t3),
                "cue": 1e3 * (t5 - t4),
                "total": 1e3 * (t5 - t0),
            },
        )
