"""Researcher views: value heatmaps and projected footsteps. Patients only ever see footsteps."""

from __future__ import annotations

import cv2
import numpy as np

from cafog.spatial.cues import Footstep
from cafog.spatial.engine import DepthFrame, EngineResult
from cafog.spatial.geometry import CameraIntrinsics
from cafog.spatial.occupancy import OCCUPIED, UNKNOWN, TraversabilityMap

FOOT_LENGTH_M = 0.26
FOOT_WIDTH_M = 0.10
LEFT_COLOR = (255, 200, 60)
RIGHT_COLOR = (60, 160, 255)


def footprint_outline(step: Footstep, n: int = 14) -> np.ndarray:
    """Shoe-shaped outline (x, z) for a footstep, toe pointing along the walking direction."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    along = 0.5 * FOOT_LENGTH_M * np.cos(t)
    across = 0.5 * FOOT_WIDTH_M * np.sin(t) * np.where(np.cos(t) > 0, 1.0, 0.75)
    c, s = np.cos(step.yaw_rad), np.sin(step.yaw_rad)
    forward = np.array([c, s])
    lateral = np.array([-s, c])
    return step.position + along[:, None] * forward + across[:, None] * lateral


def value_rgb(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.where(mask, values, np.nan)
    finite = np.isfinite(v)
    if finite.any():
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        norm = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        norm = np.zeros_like(values)
    norm = np.nan_to_num(norm)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def bev_image(
    result: EngineResult, grid: TraversabilityMap, *, which: str = "pd", px_per_cell: int = 6
) -> np.ndarray:
    field = result.field
    values = field.value_pd if which == "pd" else field.value_hc
    reachable = field.traversable & np.isfinite(field.cost_to_go)
    img = value_rgb(values, reachable)
    img[~reachable] = (70, 70, 70)
    img[field.states == UNKNOWN] = (30, 30, 30)
    img[field.states == OCCUPIED] = (235, 235, 235)
    img = cv2.resize(img, None, fx=px_per_cell, fy=px_per_cell, interpolation=cv2.INTER_NEAREST)

    def to_px(xz: np.ndarray) -> np.ndarray:
        rel = (np.asarray(xz) - grid.origin) / grid.config.cell_size_m * px_per_cell
        return np.stack([rel[..., 0], rel[..., 1]], axis=-1).astype(np.int32)

    if result.path is not None:
        cv2.polylines(img, [to_px(result.path.points)], False, (255, 255, 255), 1, cv2.LINE_AA)
    for step in result.cue.footsteps:
        color = LEFT_COLOR if step.side < 0 else RIGHT_COLOR
        cv2.fillPoly(img, [to_px(footprint_outline(step))], color, cv2.LINE_AA)
    if result.cue.hazard is not None and result.path is not None:
        cv2.circle(
            img,
            tuple(to_px(result.path.point_at(result.cue.hazard.distance_m))),
            7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    user = to_px(result.user_xz)
    tip = to_px(result.user_xz + 0.5 * result.heading_xz)
    cv2.arrowedLine(img, tuple(user), tuple(tip), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.4)
    # Rotate rather than flip: a flipped top view would mirror left and right turns.
    return np.ascontiguousarray(img[::-1, ::-1])


def project_world(
    points_xyz: np.ndarray, intrinsics: CameraIntrinsics, camera_to_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    world_to_cam = np.linalg.inv(camera_to_world)
    cam = points_xyz @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = cam[:, 2]
    uv = np.stack(
        [
            intrinsics.fx * cam[:, 0] / z + intrinsics.cx,
            intrinsics.fy * cam[:, 1] / z + intrinsics.cy,
        ],
        axis=-1,
    )
    return uv, z


def depth_to_gray(depth_m: np.ndarray, max_m: float = 6.0) -> np.ndarray:
    d = np.nan_to_num(np.clip(depth_m / max_m, 0, 1), nan=1.0, posinf=1.0)
    gray = ((1.0 - d) * 200 + 30).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def camera_overlay(
    frame: DepthFrame,
    result: EngineResult,
    grid: TraversabilityMap,
    *,
    heatmap: bool = True,
    alpha: float = 0.45,
) -> np.ndarray:
    """Floor-projected value heatmap and footsteps in the camera view (no object labels)."""
    base = frame.color.copy() if frame.color is not None else depth_to_gray(frame.depth_m)
    if base.shape[:2] != frame.depth_m.shape:
        base = cv2.resize(base, (frame.depth_m.shape[1], frame.depth_m.shape[0]))
    k = frame.intrinsics
    pose = np.asarray(frame.camera_to_world)
    floor_y = grid.floor_y if grid.floor_y is not None else pose[1, 3] - 1.6

    if heatmap:
        vv, uu = np.indices(frame.depth_m.shape)
        rays = np.stack([(uu - k.cx) / k.fx, (vv - k.cy) / k.fy, np.ones_like(uu, float)], -1)
        d_world = rays @ pose[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (floor_y - pose[1, 3]) / d_world[..., 1]
        sees_floor = (t > 0) & (np.abs(frame.depth_m - t) < 0.08 * np.maximum(t, 1.0))
        hit = pose[:3, 3] + t[..., None] * d_world
        rc = grid.world_to_cell(hit[..., [0, 2]])
        ok = sees_floor & grid.in_bounds(rc)
        reachable = result.field.traversable & np.isfinite(result.field.cost_to_go)
        colors = value_rgb(result.field.value_pd, reachable)
        cell_ok = np.zeros_like(ok)
        cell_ok[ok] = reachable[rc[ok, 0], rc[ok, 1]]
        overlay = base.copy()
        overlay[cell_ok] = colors[rc[cell_ok, 0], rc[cell_ok, 1]]
        base = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    for step in result.cue.footsteps:
        outline = footprint_outline(step)
        pts = np.column_stack([outline[:, 0], np.full(len(outline), floor_y), outline[:, 1]])
        uv, z = project_world(pts, k, pose)
        if np.all(z > 0.1):
            color = LEFT_COLOR if step.side < 0 else RIGHT_COLOR
            cv2.fillPoly(base, [uv.astype(np.int32)], color, cv2.LINE_AA)
    return base


def status_panel(result: EngineResult, width: int, height: int = 70) -> np.ndarray:
    panel = np.zeros((height, width, 3), np.uint8)
    cue = result.cue
    risk = result.risk.max_within(3.0) if result.risk is not None else float("nan")
    lines = [
        f"cue {cue.phase.value}  risk(3m) {risk:.2f}  speed {result.speed_mps:.2f} m/s  "
        f"{result.timings_ms['total']:.1f} ms",
        cue.reason
        + (
            f" | {cue.hazard.kind.value} in {cue.hazard.distance_m:.1f} m, "
            f"turn {cue.hazard.turn_deg:.0f} deg, width {cue.hazard.width_m:.2f} m"
            if cue.hazard
            else ""
        )
        + (f" | metronome {cue.metronome_bpm:.0f} bpm" if cue.metronome_bpm else ""),
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (8, 25 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return panel


def compose(
    frame: DepthFrame, result: EngineResult, grid: TraversabilityMap, height: int = 480
) -> np.ndarray:
    cam = camera_overlay(frame, result, grid)
    cam = cv2.resize(cam, (int(cam.shape[1] * height / cam.shape[0]), height))
    bev = bev_image(result, grid)
    bev = cv2.resize(bev, (height, height), interpolation=cv2.INTER_NEAREST)
    top = np.hstack([cam, bev])
    return np.vstack([top, status_panel(result, top.shape[1])])
