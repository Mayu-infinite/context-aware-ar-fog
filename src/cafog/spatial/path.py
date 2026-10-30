from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cafog.spatial.config import ValueConfig
from cafog.spatial.geometry import right_of
from cafog.spatial.occupancy import FREE, OCCUPIED, TraversabilityMap
from cafog.spatial.value import ValueField, narrowness_ratio, turn_ratio

WIDTH_RAY_STEP_M = 0.05
HEADING_LEAD_M = 0.3


@dataclass
class PredictedPath:
    """Arc-length-resampled route the user is expected to walk next."""

    points: np.ndarray
    arc_m: np.ndarray
    tangents: np.ndarray
    headings: np.ndarray
    turn_deg: np.ndarray
    turn_ratio: np.ndarray
    width_m: np.ndarray
    narrow_ratio: np.ndarray
    cost_to_go: np.ndarray
    cost: np.ndarray
    known_free: np.ndarray
    clearance_m: np.ndarray

    @property
    def length_m(self) -> float:
        return float(self.arc_m[-1]) if self.arc_m.size else 0.0

    @property
    def demand(self) -> np.ndarray:
        return np.maximum(self.turn_ratio, self.narrow_ratio)

    def point_at(self, s: float) -> np.ndarray:
        return np.stack(
            [
                np.interp(s, self.arc_m, self.points[:, 0]),
                np.interp(s, self.arc_m, self.points[:, 1]),
            ],
            axis=-1,
        )

    def tangent_at(self, s: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.arc_m, s), 0, len(self.arc_m) - 1))
        return self.tangents[i]


def trace_route(field: ValueField, max_cells: int) -> list[tuple[int, int]]:
    if field.start_cell is None or not field.has_route:
        return []
    n_c = field.next_cell.shape[1]
    cur = field.start_cell
    route = [cur]
    for _ in range(max_cells):
        nxt = int(field.next_cell[cur])
        if nxt < 0:
            break
        cur = divmod(nxt, n_c)
        route.append(cur)
    return route


def smooth_polyline(points: np.ndarray, window: int) -> np.ndarray:
    if len(points) < 3 or window < 2:
        return points
    half = window // 2
    padded = np.concatenate(
        [np.repeat(points[:1], half, 0), points, np.repeat(points[-1:], half, 0)]
    )
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    out = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=-1)
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def resample_polyline(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    points = points[keep]
    if len(points) < 2:
        return points, np.zeros(len(points))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    samples = np.arange(0.0, arc[-1] + 1e-9, step)
    if samples[-1] < arc[-1] - 1e-6:
        samples = np.append(samples, arc[-1])
    out = np.stack([np.interp(samples, arc, points[:, i]) for i in range(2)], axis=-1)
    return out, samples


def sample_grid(grid: TraversabilityMap, values: np.ndarray, xz: np.ndarray, fill) -> np.ndarray:
    rc = grid.world_to_cell(xz)
    ok = grid.in_bounds(rc)
    out = np.full(len(xz), fill, dtype=np.result_type(values.dtype, type(fill)))
    out[ok] = values[rc[ok, 0], rc[ok, 1]]
    return out


def lateral_width(
    grid: TraversabilityMap,
    states: np.ndarray,
    points: np.ndarray,
    tangents: np.ndarray,
    max_m: float,
) -> np.ndarray:
    """Passage width from two rays cast perpendicular to the path until an occupied cell."""
    steps = np.arange(WIDTH_RAY_STEP_M, max_m + 1e-9, WIDTH_RAY_STEP_M)
    lateral = right_of(tangents)
    total = np.zeros(len(points))
    for side in (1.0, -1.0):
        probes = points[:, None, :] + side * lateral[:, None, :] * steps[None, :, None]
        occ = sample_grid(grid, states, probes.reshape(-1, 2), 0).reshape(len(points), -1)
        hit = occ == OCCUPIED
        first = np.where(hit.any(axis=1), steps[np.argmax(hit, axis=1)] - WIDTH_RAY_STEP_M, max_m)
        total += first
    return total


def windowed_turn(headings: np.ndarray, arc: np.ndarray, window_m: float) -> np.ndarray:
    unwrapped = np.unwrap(headings)
    before = np.interp(arc - 0.5 * window_m, arc, unwrapped)
    after = np.interp(arc + 0.5 * window_m, arc, unwrapped)
    return np.abs(after - before)


def predict_path(
    grid: TraversabilityMap,
    field: ValueField,
    user_xz: np.ndarray,
    heading_xz: np.ndarray,
    cfg: ValueConfig,
) -> PredictedPath | None:
    cs = grid.config.cell_size_m
    route = trace_route(field, max_cells=int(4 * cfg.lookahead_m / cs) + 10)
    if len(route) < 2:
        return None

    centers = grid.cell_to_world(np.array(route))
    lead = np.asarray(user_xz) - HEADING_LEAD_M * np.asarray(heading_xz)
    raw = np.vstack([lead, np.asarray(user_xz), centers[1:]])
    smoothed = smooth_polyline(raw, cfg.path_smoothing_window)
    points, arc = resample_polyline(smoothed, cfg.path_sample_step_m)
    if len(points) < 3:
        return None

    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-9)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    turn = np.degrees(windowed_turn(headings, arc, cfg.turn_window_m))

    # Drop the synthetic lead-in so arc length 0 is the user's position.
    start = int(np.argmin(np.abs(arc - HEADING_LEAD_M)))
    keep = slice(start, None)
    points, tangents, headings, turn = points[keep], tangents[keep], headings[keep], turn[keep]
    arc = arc[keep] - arc[start]
    inside = arc <= cfg.lookahead_m
    points, tangents, headings, turn, arc = (
        points[inside],
        tangents[inside],
        headings[inside],
        turn[inside],
        arc[inside],
    )
    if len(points) < 2:
        return None

    # Rays stop at the first occupied cell, which on average starts half a cell before the
    # surface on each side.
    width = lateral_width(grid, field.states, points, tangents, cfg.width_ray_max_m) + cs
    return PredictedPath(
        points=points,
        arc_m=arc,
        tangents=tangents,
        headings=headings,
        turn_deg=turn,
        turn_ratio=turn_ratio(turn, cfg),
        width_m=width,
        narrow_ratio=narrowness_ratio(width, cfg),
        cost_to_go=sample_grid(grid, field.cost_to_go, points, math.inf),
        cost=sample_grid(grid, field.cost, points, math.inf),
        known_free=sample_grid(grid, field.states, points, -1) == FREE,
        clearance_m=sample_grid(grid, field.clearance_m, points, 0.0),
    )
