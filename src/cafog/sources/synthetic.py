"""Procedural floor plans and a ray-cast depth renderer for headset-free testing.

The turning scenes mirror the environment of Bhattacharya et al. (2026): a corridor whose
second segment is rotated by the turn angle, with total walking distance kept similar.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from cafog.spatial.config import EngineConfig
from cafog.spatial.engine import DepthFrame, SpatialCueEngine
from cafog.spatial.geometry import CameraIntrinsics, look_at_pose, pixel_rays, right_of

WALL_HEIGHT_M = 2.5
SIM_INTRINSICS = CameraIntrinsics.from_fov(240, 180, 90.0)


@dataclass
class Scene:
    name: str
    segments: np.ndarray
    heights: np.ndarray
    route: np.ndarray
    hazards: list[tuple[str, np.ndarray]] = field(default_factory=list)

    def route_arc(self) -> np.ndarray:
        return np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(self.route, axis=0), axis=1))]
        )


def _direction(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


def offset_polyline(points: np.ndarray, offset: float) -> np.ndarray:
    """Mitered parallel offset; positive offset is to the left of travel."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    normals = -right_of(seg)
    out = [pts[0] + offset * normals[0]]
    for i in range(1, len(pts) - 1):
        n = normals[i - 1] + normals[i]
        n /= np.linalg.norm(n)
        out.append(pts[i] + n * offset / max(float(n @ normals[i]), 0.2))
    out.append(pts[-1] + offset * normals[-1])
    return np.array(out)


def polyline_segments(points: np.ndarray) -> np.ndarray:
    return np.stack([points[:-1], points[1:]], axis=1)


def _box(center: tuple[float, float], size: tuple[float, float]) -> np.ndarray:
    cx, cz = center
    hx, hz = size[0] / 2, size[1] / 2
    c = np.array([[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]])
    return polyline_segments(np.vstack([c, c[:1]]))


def _scene(name, walls, route, extra=(), extra_heights=(), hazards=()) -> Scene:
    parts = [np.asarray(walls, dtype=np.float64).reshape(-1, 2, 2), *extra]
    heights = [np.full(len(parts[0]), WALL_HEIGHT_M)]
    heights += [np.full(len(e), h) for e, h in zip(extra, extra_heights, strict=True)]
    return Scene(
        name,
        np.concatenate(parts),
        np.concatenate(heights),
        np.asarray(route, dtype=np.float64),
        list(hazards),
    )


def corridor_turn(
    turn_deg: float,
    width_m: float = 1.4,
    before_m: float = 6.0,
    after_m: float = 6.0,
    overrun_m: float = 4.0,
) -> Scene:
    """Corridor with one turn of `turn_deg` degrees; 180 builds a U-turn with a thin divider.

    The walls continue `overrun_m` past the end of the walk so the corridor never opens
    onto the renderer's unbounded floor while the walker is still inside it.
    """
    start = np.array([0.0, -1.0])
    corner = np.array([0.0, before_m])
    psi0 = math.pi / 2
    if turn_deg >= 179.0:
        gap = width_m + 0.2
        p2 = corner + gap * _direction(psi0 - math.pi / 2)
        exit_dir = _direction(psi0 - math.pi)
        centre = np.array([start, corner, p2, p2 + (after_m + overrun_m) * exit_dir])
        walk_end = p2 + after_m * exit_dir
        apex = 0.5 * (corner + p2)
    else:
        exit_dir = _direction(psi0 - math.radians(turn_deg))
        centre = np.array([start, corner, corner + (after_m + overrun_m) * exit_dir])
        walk_end = corner + after_m * exit_dir
        apex = corner
    left = offset_polyline(centre, width_m / 2)
    right = offset_polyline(centre, -width_m / 2)
    walls = list(polyline_segments(left)) + list(polyline_segments(right))
    walls.append(np.array([left[0], right[0]]))
    route = np.array([start + np.array([0.0, 0.4]), *centre[1:-1], walk_end])
    hazards = [("turn", apex)] if turn_deg >= 30.0 else []
    return _scene(f"turn_{int(round(turn_deg))}", walls, route, hazards=hazards)


def doorway(
    door_width_m: float,
    corridor_width_m: float = 2.0,
    door_at_m: float = 5.0,
    length_m: float = 14.0,
    wall_thickness_m: float = 0.12,
) -> Scene:
    hw = corridor_width_m / 2
    hd = door_width_m / 2
    z0, z1 = door_at_m, door_at_m + wall_thickness_m
    walls = [
        [[-hw, -1.0], [-hw, length_m]],
        [[hw, -1.0], [hw, length_m]],
        [[-hw, -1.0], [hw, -1.0]],
    ]
    for sign in (-1.0, 1.0):
        outer, inner = sign * hw, sign * hd
        walls += [
            [[outer, z0], [inner, z0]],
            [[outer, z1], [inner, z1]],
            [[inner, z0], [inner, z1]],
        ]
    route = np.array([[0.0, -0.6], [0.0, length_m - 4.0]])
    return _scene(
        f"door_{int(round(door_width_m * 100))}cm",
        walls,
        route,
        hazards=[("narrow", np.array([0.0, z0]))],
    )


def cluttered_room() -> Scene:
    walls = polyline_segments(np.array([[-3, -1], [3, -1], [3, 9], [-3, 9], [-3, -1]], float))
    centres = [(0.6, 2.5), (-1.2, 4.5), (0.9, 6.0)]
    boxes = [_box(c, s) for c, s in zip(centres, [(0.8, 0.6), (1.4, 0.5), (1.6, 0.6)], strict=True)]
    route = np.array(
        [[0.0, -0.5], [0.0, 1.6], [-0.6, 3.2], [0.0, 4.5], [-0.2, 5.6], [-1.2, 7.0], [-1.2, 8.3]]
    )
    hazards = [("obstacle", np.array(c)) for c in centres]
    return _scene(
        "cluttered_room",
        walls,
        route,
        extra=boxes,
        extra_heights=[0.75, 0.9, 0.45],
        hazards=hazards,
    )


PAPER_TURN_ANGLES = (0, 60, 90, 120, 180)


def render_depth(
    scene: Scene, intrinsics: CameraIntrinsics, pose: np.ndarray, max_range_m: float = 8.0
) -> np.ndarray:
    """Metric z-depth (OpenCV camera) of floor and extruded wall segments; inf for no hit."""
    _, rays_cam = pixel_rays(intrinsics, 1)
    rays_cam = rays_cam.reshape(-1, 3).astype(np.float64)
    d = rays_cam @ pose[:3, :3].T
    o = pose[:3, 3]
    best = np.full(len(d), np.inf)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(d[:, 1] < -1e-9, -o[1] / d[:, 1], np.inf)
    best = np.minimum(best, t_floor)
    if len(scene.segments) == 0:
        best[best > max_range_m] = np.inf
        return best.reshape(intrinsics.height, intrinsics.width).astype(np.float32)

    a = scene.segments[:, 0]
    e = scene.segments[:, 1] - a
    dx, dz = d[:, 0:1], d[:, 2:3]
    denom = dx * e[None, :, 1] - dz * e[None, :, 0]
    wx = a[None, :, 0] - o[0]
    wz = a[None, :, 1] - o[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * e[None, :, 1] - wz * e[None, :, 0]) / denom
        u = (wx * dz - wz * dx) / denom
    y = o[1] + t * d[:, 1:2]
    hit = (
        (np.abs(denom) > 1e-12)
        & (t > 1e-6)
        & (u >= 0)
        & (u <= 1)
        & (y >= 0)
        & (y <= scene.heights[None, :])
    )
    t_wall = np.where(hit, t, np.inf).min(axis=1)
    best = np.minimum(best, t_wall)
    best[best > max_range_m] = np.inf
    return best.reshape(intrinsics.height, intrinsics.width).astype(np.float32)


def walk(
    scene: Scene,
    *,
    speed_mps: float = 0.8,
    fps: float = 10.0,
    eye_height_m: float = 1.6,
    pitch_deg: float = 20.0,
    look_ahead_m: float = 0.8,
    intrinsics: CameraIntrinsics | None = None,
    pause_s: float = 0.0,
    noise_std_m: float = 0.0,
    seed: int = 0,
) -> Iterator[DepthFrame]:
    """Walk along the scene route, yielding depth frames like a headset would."""
    intrinsics = intrinsics or SIM_INTRINSICS
    rng = np.random.default_rng(seed)
    route = scene.route
    arc = scene.route_arc()
    total = float(arc[-1])
    dt = 1.0 / fps
    t = 0.0
    s = 0.0
    while s <= total:
        pos2 = np.array([np.interp(s, arc, route[:, 0]), np.interp(s, arc, route[:, 1])])
        ahead_s = min(s + look_ahead_m, total)
        ahead = np.array(
            [np.interp(ahead_s, arc, route[:, 0]), np.interp(ahead_s, arc, route[:, 1])]
        )
        heading = ahead - pos2
        if np.linalg.norm(heading) < 1e-6:
            heading = route[-1] - route[-2]
        pose = look_at_pose(np.array([pos2[0], eye_height_m, pos2[1]]), heading, pitch_deg)
        depth = render_depth(scene, intrinsics, pose)
        if noise_std_m > 0:
            finite = np.isfinite(depth)
            depth[finite] += rng.normal(0.0, noise_std_m, finite.sum()) * depth[finite]
        yield DepthFrame(depth_m=depth, intrinsics=intrinsics, camera_to_world=pose, timestamp_s=t)
        t += dt
        if t > pause_s:
            s += speed_mps * dt


@dataclass
class FrameRecord:
    t: float
    user_xz: np.ndarray
    speed_mps: float
    risk_ahead: float
    phase: str
    footsteps: int
    hazard_distance_m: float | None
    scene_hazard_distance_m: float | None
    latency_ms: float


def run_scene(
    scene: Scene, config: EngineConfig, on_result=None, **walk_kwargs
) -> list[FrameRecord]:
    engine = SpatialCueEngine(config)
    records = []
    for frame in walk(scene, **walk_kwargs):
        result = engine.process(frame)
        if on_result is not None:
            on_result(frame, result, engine)
        scene_d = (
            min(float(np.linalg.norm(p - result.user_xz)) for _, p in scene.hazards)
            if scene.hazards
            else None
        )
        hazard = result.cue.hazard
        records.append(
            FrameRecord(
                t=frame.timestamp_s,
                user_xz=result.user_xz,
                speed_mps=result.speed_mps,
                risk_ahead=result.risk.max_within(config.cue.trigger_distance_m)
                if result.risk is not None
                else float("nan"),
                phase=result.cue.phase.value,
                footsteps=len(result.cue.footsteps),
                hazard_distance_m=None if hazard is None else hazard.distance_m,
                scene_hazard_distance_m=scene_d,
                latency_ms=result.timings_ms["total"],
            )
        )
    return records


def summarize_run(records: list[FrameRecord], false_cue_margin_m: float = 5.0) -> dict:
    """Cue timing relative to the scene's known hazard; activations far from any hazard count
    as false cues."""
    active = [r for r in records if r.phase in ("ACTIVE", "RELEASING")]
    first = active[0] if active else None
    false_frames = sum(
        1
        for r in active
        if r.scene_hazard_distance_m is None or r.scene_hazard_distance_m > false_cue_margin_m
    )
    latencies = np.array([r.latency_ms for r in records])
    risks = np.array([r.risk_ahead for r in records])
    return {
        "frames": len(records),
        "active_frames": len(active),
        "false_cue_frames": false_frames,
        "first_cue_distance_m": None if first is None else first.scene_hazard_distance_m,
        "max_risk": float(np.nanmax(risks)) if np.isfinite(risks).any() else float("nan"),
        "latency_mean_ms": float(latencies.mean()),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }
