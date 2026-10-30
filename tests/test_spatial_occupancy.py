import numpy as np
import pytest

from cafog.spatial.config import MapConfig
from cafog.spatial.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    CellCounts,
    TraversabilityMap,
    estimate_floor_from_histogram,
)


def test_floor_estimate_prefers_lowest_strong_surface_over_table():
    rng = np.random.default_rng(0)
    floor = rng.normal(0.02, 0.01, 4000)
    table = rng.normal(0.75, 0.01, 3000)
    estimate = estimate_floor_from_histogram(np.concatenate([floor, table]), eye_height_y=1.6)
    assert estimate == pytest.approx(0.02, abs=0.03)


def test_floor_estimate_none_without_evidence():
    assert estimate_floor_from_histogram(np.array([1.5, 1.55]), eye_height_y=1.6) is None


def test_points_classified_into_floor_obstacle_and_dropoff():
    grid = TraversabilityMap(MapConfig(size_m=4.0), center_xz=(0, 0))
    pts = np.array(
        [[0.05, 0.0, 0.05]] * 3 + [[1.05, 0.5, 0.05]] * 3 + [[-1.05, -0.5, 0.05]] * 3,
        dtype=np.float32,
    )
    counts = grid.count_points(pts, floor_y=0.0)
    for xz, attr in [
        ((0.05, 0.05), "floor"),
        ((1.05, 0.05), "obstacle"),
        ((-1.05, 0.05), "dropoff"),
    ]:
        r, c = grid.world_to_cell(np.array(xz))
        assert getattr(counts, attr)[r, c] == 3


def test_integration_marks_free_and_occupied_and_obstacle_wins():
    grid = TraversabilityMap(MapConfig(size_m=2.0), center_xz=(0, 0))
    n = grid.config.cells
    obstacle = np.zeros((n, n), np.int32)
    floor = np.zeros((n, n), np.int32)
    floor[5, 5] = floor[6, 6] = 5
    obstacle[6, 6] = 5
    grid.integrate_counts(CellCounts(obstacle, floor, np.zeros_like(floor)), 0.0)
    states = grid.states()
    assert states[5, 5] == FREE
    assert states[6, 6] == OCCUPIED
    assert states[0, 0] == UNKNOWN


def test_recenter_keeps_world_anchored_content():
    grid = TraversabilityMap(MapConfig(size_m=4.0, recenter_distance_m=1.0), center_xz=(0, 0))
    target = np.array([0.55, 0.35])
    r, c = grid.world_to_cell(target)
    grid.log_odds[r, c] = 3.0
    assert grid.recenter(np.array([1.6, -1.2]))
    r2, c2 = grid.world_to_cell(target)
    assert (r2, c2) != (r, c)
    assert grid.log_odds[r2, c2] == 3.0
    assert np.count_nonzero(grid.log_odds) == 1


def test_unobserved_cells_decay_towards_unknown():
    cfg = MapConfig(size_m=2.0, decay_time_constant_s=1.0)
    grid = TraversabilityMap(cfg)
    n = cfg.cells
    hit = np.zeros((n, n), np.int32)
    hit[3, 3] = 5
    zeros = np.zeros_like(hit)
    grid.integrate_counts(CellCounts(hit, zeros, zeros), 0.0)
    grid.integrate_counts(CellCounts(zeros, zeros, zeros), 5.0)
    assert abs(grid.log_odds[3, 3]) < 0.05
