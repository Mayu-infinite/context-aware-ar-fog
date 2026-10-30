import math

import numpy as np

from cafog.spatial.config import MapConfig, ValueConfig, make_profile
from cafog.spatial.occupancy import FREE, OCCUPIED, TraversabilityMap
from cafog.spatial.value import (
    clearance_map,
    compute_value_field,
    narrowness_ratio,
    solve_cost_to_go,
    traversal_cost,
    turn_ratio,
)


def corridor_map(width_cells: int = 12, opening: int | None = None) -> TraversabilityMap:
    """Straight free corridor along +z with walls; optional narrow opening in a cross wall."""
    grid = TraversabilityMap(MapConfig(size_m=6.0), center_xz=(0, 0))
    n = grid.config.cells
    grid.observed[:] = True
    grid.log_odds[:] = 3.0
    mid = n // 2
    lo, hi = mid - width_cells // 2, mid + width_cells // 2
    grid.log_odds[:, lo:hi] = -2.0
    if opening is not None:
        grid.log_odds[n // 2, lo:hi] = 3.0
        grid.log_odds[n // 2, mid - opening // 2 : mid + (opening + 1) // 2] = -2.0
    return grid


def test_cost_to_go_matches_grid_distance_in_open_space():
    cost = np.ones((20, 20), np.float32)
    goal = np.full((20, 20), np.inf)
    goal[0, 0] = 0.0
    ctg, nxt = solve_cost_to_go(cost, goal, cell_size_m=0.1)
    assert ctg[0, 10] == np.float64(ctg[0, 10])
    assert math.isclose(ctg[0, 10], 1.0, rel_tol=1e-6)
    assert math.isclose(ctg[10, 10], 10 * 0.1 * math.sqrt(2), rel_tol=1e-6)
    assert nxt[0, 0] == -1
    assert nxt[0, 1] == 0


def test_walls_block_and_corners_are_not_cut():
    cost = np.ones((5, 5), np.float32)
    cost[1, 1] = np.inf
    goal = np.full((5, 5), np.inf)
    goal[0, 0] = 0.0
    _, nxt = solve_cost_to_go(cost, goal, 0.1)
    assert nxt[1, 2] != 0  # cannot slip diagonally past the blocked cell
    cost[0, 1] = cost[1, 0] = np.inf
    ctg, _ = solve_cost_to_go(cost, goal, 0.1)
    assert np.isinf(ctg[2, 2])


def test_clearance_and_cost_respect_body_radius():
    states = np.full((30, 30), FREE, np.int8)
    states[:, 0] = OCCUPIED
    clr = clearance_map(states, 0.1)
    assert clr[5, 1] < clr[5, 10]
    cost = traversal_cost(states, clr, ValueConfig(body_radius_m=0.25))
    assert np.isinf(cost[5, 1]) and np.isinf(cost[5, 0])
    assert cost[5, 3] > cost[5, 20] >= 1.0


def test_value_increases_along_route_to_goal_ahead():
    grid = corridor_map()
    field = compute_value_field(
        grid, np.array([0.0, -2.0]), np.array([0.0, 1.0]), ValueConfig(), make_profile("healthy")
    )
    assert field.has_route
    r, c = field.start_cell
    values = [field.value_hc[r, c]]
    cur = (r, c)
    for _ in range(15):
        nxt = int(field.next_cell[cur])
        if nxt < 0:
            break
        cur = divmod(nxt, field.next_cell.shape[1])
        values.append(field.value_hc[cur])
    assert np.all(np.diff(values) > 0)
    assert np.allclose(field.value_pd, field.value_hc)


def test_goals_behind_the_user_are_excluded():
    grid = corridor_map()
    field = compute_value_field(
        grid, np.array([0.0, -2.0]), np.array([0.0, 1.0]), ValueConfig(), make_profile("pd_fog")
    )
    rows = np.flatnonzero(field.goal_mask.any(axis=1))
    user_row = grid.world_to_cell(np.array([0.0, -2.0]))[0]
    assert rows.min() > user_row


def test_pd_value_canyon_at_narrow_opening():
    grid = corridor_map(width_cells=16, opening=7)
    user = np.array([0.0, -2.0])
    pd = compute_value_field(
        grid, user, np.array([0.0, 1.0]), ValueConfig(), make_profile("pd_fog")
    )
    n = grid.config.cells
    door_row, mid = n // 2, n // 2
    assert pd.demand[door_row, mid] > 0.5
    assert pd.value_pd[door_row, mid] < pd.value_hc[door_row, mid]


def test_demand_ratios_follow_paper_angle_ratio():
    cfg = ValueConfig(turn_deadband_deg=15.0)
    assert turn_ratio(np.array(10.0), cfg) == 0.0
    assert turn_ratio(np.array(90.0), cfg) == 1.0
    assert turn_ratio(np.array(180.0), cfg) == 2.0
    assert narrowness_ratio(np.array(cfg.comfort_width_m), cfg) == 0.0
    assert narrowness_ratio(np.array(cfg.min_width_m), cfg) == 1.0
