from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cafog.spatial.config import MapConfig

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

FLOOR_HISTOGRAM_BIN_M = 0.02
FLOOR_SEARCH_BELOW_EYE_M = (0.5, 2.3)


@dataclass(frozen=True)
class CellCounts:
    """Per-cell point counts for one depth frame (the Quest compute shader produces the same)."""

    obstacle: np.ndarray
    floor: np.ndarray
    dropoff: np.ndarray


def estimate_floor_from_histogram(
    heights: np.ndarray,
    eye_height_y: float,
    *,
    bin_m: float = FLOOR_HISTOGRAM_BIN_M,
    min_fraction: float = 0.02,
    peak_ratio: float = 0.3,
) -> float | None:
    """Lowest strong horizontal-surface peak below the eye, or None if there is no evidence."""
    lo = eye_height_y - FLOOR_SEARCH_BELOW_EYE_M[1]
    hi = eye_height_y - FLOOR_SEARCH_BELOW_EYE_M[0]
    heights = np.asarray(heights, dtype=np.float32)
    heights = heights[(heights >= lo) & (heights < hi)]
    if heights.size == 0:
        return None
    bins = max(1, int(math.ceil((hi - lo) / bin_m)))
    hist, edges = np.histogram(heights, bins=bins, range=(lo, lo + bins * bin_m))
    hist = hist.astype(np.float32)
    smoothed = np.convolve(hist, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
    peak = float(smoothed.max())
    if peak < min_fraction * heights.size:
        return None
    strong = np.flatnonzero(smoothed >= peak_ratio * peak)
    first = int(strong[0])
    end = first
    while end + 1 < smoothed.size and smoothed[end + 1] >= smoothed[end]:
        end += 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[end])


class TraversabilityMap:
    """Rolling log-odds occupancy grid on the ground plane, anchored in world coordinates.

    Cell (row, col) covers world x in [origin_x + col*cs, ...) and z in [origin_z + row*cs, ...).
    Only geometry is used; there are no semantic labels.
    """

    def __init__(self, config: MapConfig | None = None, center_xz: tuple[float, float] = (0, 0)):
        self.config = config or MapConfig()
        n = self.config.cells
        self.log_odds = np.zeros((n, n), dtype=np.float32)
        self.observed = np.zeros((n, n), dtype=bool)
        half = 0.5 * n * self.config.cell_size_m
        self.origin = np.array([center_xz[0] - half, center_xz[1] - half], dtype=np.float64)
        self.floor_y: float | None = None
        self._last_time: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.log_odds.shape  # type: ignore[return-value]

    @property
    def center(self) -> np.ndarray:
        return self.origin + 0.5 * self.config.cells * self.config.cell_size_m

    def world_to_cell(self, xz: np.ndarray) -> np.ndarray:
        """Integer (row, col) indices; may fall outside the grid."""
        xz = np.asarray(xz, dtype=np.float64)
        rel = (xz - self.origin) / self.config.cell_size_m
        return np.stack([np.floor(rel[..., 1]), np.floor(rel[..., 0])], axis=-1).astype(np.int64)

    def cell_to_world(self, rc: np.ndarray) -> np.ndarray:
        rc = np.asarray(rc, dtype=np.float64)
        cs = self.config.cell_size_m
        return np.stack(
            [self.origin[0] + (rc[..., 1] + 0.5) * cs, self.origin[1] + (rc[..., 0] + 0.5) * cs],
            axis=-1,
        )

    def in_bounds(self, rc: np.ndarray) -> np.ndarray:
        n = self.config.cells
        rc = np.asarray(rc)
        return (rc[..., 0] >= 0) & (rc[..., 0] < n) & (rc[..., 1] >= 0) & (rc[..., 1] < n)

    def recenter(self, user_xz: np.ndarray) -> bool:
        """Shift the grid by whole cells so the user stays near its center."""
        offset = np.asarray(user_xz, dtype=np.float64) - self.center
        if np.max(np.abs(offset)) < self.config.recenter_distance_m:
            return False
        shift = np.round(offset / self.config.cell_size_m).astype(int)
        shift_rc = (int(shift[1]), int(shift[0]))
        self.log_odds = _shift(self.log_odds, shift_rc, 0.0)
        self.observed = _shift(self.observed, shift_rc, False)
        self.origin = self.origin + shift * self.config.cell_size_m
        return True

    def update_floor(self, points: np.ndarray, eye_y: float) -> float:
        estimate = estimate_floor_from_histogram(points[:, 1], eye_y) if len(points) else None
        if estimate is not None:
            if self.floor_y is None:
                self.floor_y = estimate
            else:
                a = self.config.floor_smoothing
                self.floor_y = (1 - a) * self.floor_y + a * estimate
        if self.floor_y is None:
            return eye_y - self.config.initial_eye_height_m
        return self.floor_y

    def count_points(self, points: np.ndarray, floor_y: float) -> CellCounts:
        cfg = self.config
        n = cfg.cells
        size = n * n
        if len(points) == 0:
            zeros = np.zeros((n, n), dtype=np.int32)
            return CellCounts(zeros, zeros.copy(), zeros.copy())
        h = points[:, 1] - floor_y
        rc = self.world_to_cell(points[:, [0, 2]])
        inside = self.in_bounds(rc)
        flat = rc[:, 0] * n + rc[:, 1]

        def binned(mask: np.ndarray) -> np.ndarray:
            sel = mask & inside
            return np.bincount(flat[sel], minlength=size).reshape(n, n).astype(np.int32)

        return CellCounts(
            obstacle=binned((h >= cfg.obstacle_min_height_m) & (h <= cfg.obstacle_max_height_m)),
            floor=binned(np.abs(h) <= cfg.floor_tolerance_m),
            dropoff=binned(h <= -cfg.dropoff_depth_m),
        )

    def integrate_counts(self, counts: CellCounts, timestamp_s: float) -> None:
        cfg = self.config
        dt = 0.0 if self._last_time is None else max(0.0, timestamp_s - self._last_time)
        self._last_time = timestamp_s

        hit = (counts.obstacle >= cfg.min_obstacle_points) | (
            counts.dropoff >= cfg.min_obstacle_points
        )
        free = (counts.floor >= cfg.min_floor_points) & ~hit
        seen = hit | free

        if dt > 0 and cfg.decay_time_constant_s > 0:
            decay = math.exp(-dt / cfg.decay_time_constant_s)
            self.log_odds[~seen] *= decay

        self.log_odds[hit] += cfg.log_odds_hit
        self.log_odds[free] += cfg.log_odds_free
        np.clip(self.log_odds, cfg.log_odds_min, cfg.log_odds_max, out=self.log_odds)
        self.observed |= seen

    def integrate_points(self, points: np.ndarray, eye_y: float, timestamp_s: float) -> CellCounts:
        floor_y = self.update_floor(points, eye_y)
        counts = self.count_points(points, floor_y)
        self.integrate_counts(counts, timestamp_s)
        return counts

    def states(self) -> np.ndarray:
        cfg = self.config
        out = np.full(self.shape, UNKNOWN, dtype=np.int8)
        out[self.log_odds >= cfg.occupied_threshold] = OCCUPIED
        out[(self.log_odds <= cfg.free_threshold) & self.observed] = FREE
        return out


def _shift(grid: np.ndarray, shift_rc: tuple[int, int], fill) -> np.ndarray:
    """Content moves opposite to the origin shift; vacated cells are filled."""
    out = np.full_like(grid, fill)
    dr, dc = shift_rc
    n_r, n_c = grid.shape
    src_r = slice(max(dr, 0), n_r + min(dr, 0))
    dst_r = slice(max(-dr, 0), n_r + min(-dr, 0))
    src_c = slice(max(dc, 0), n_c + min(dc, 0))
    dst_c = slice(max(-dc, 0), n_c + min(-dc, 0))
    if src_r.start < src_r.stop and src_c.start < src_c.stop:
        out[dst_r, dst_c] = grid[src_r, src_c]
    return out
