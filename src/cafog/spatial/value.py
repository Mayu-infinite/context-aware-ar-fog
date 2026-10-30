"""Training-free spatial critic.

The healthy critic is the exact optimal value of a discounted navigation problem on the
current map: reward 1 on reaching a goal ahead, cost-weighted distance D(s) to it, and
V_HC(s) = gamma ** D(s). Dijkstra solves this in closed form, so nothing is learned.

The Parkinsonian critic follows Bhattacharya et al. (2026): a clamped TD error leaves a
"value canyon" where the spatial demand (turn sharpness, narrowness) is high. We model it
as V_PD(s) = V_HC(s) * exp(-kappa * r(s)), with kappa proportional to the FOG factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from cafog.spatial.config import PatientProfile, ValueConfig
from cafog.spatial.occupancy import FREE, OCCUPIED, UNKNOWN, TraversabilityMap

NEIGHBOURS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int64
)


@dataclass
class ValueField:
    states: np.ndarray
    clearance_m: np.ndarray
    cost: np.ndarray
    traversable: np.ndarray
    cost_to_go: np.ndarray
    next_cell: np.ndarray
    goal_mask: np.ndarray
    value_hc: np.ndarray
    demand: np.ndarray
    value_pd: np.ndarray
    start_cell: tuple[int, int] | None
    discount_per_m: float

    @property
    def has_route(self) -> bool:
        if self.start_cell is None:
            return False
        return bool(np.isfinite(self.cost_to_go[self.start_cell]))


def clearance_map(states: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Distance (m) from each cell centre to the nearest occupied surface."""
    occupied = states == OCCUPIED
    if not occupied.any():
        return np.full(states.shape, np.inf, dtype=np.float32)
    edt = ndimage.distance_transform_edt(~occupied).astype(np.float32) * cell_size_m
    return np.maximum(edt - 0.5 * cell_size_m, 0.0)


def traversal_cost(states: np.ndarray, clearance_m: np.ndarray, cfg: ValueConfig) -> np.ndarray:
    """Per-metre traversal cost; inf where the body does not fit."""
    margin = np.minimum(clearance_m, 10.0) - cfg.body_radius_m
    cost = 1.0 + cfg.narrow_cost_gain * np.exp(-np.maximum(margin, 0.0) / cfg.narrow_cost_length_m)
    cost = cost + np.where(states == UNKNOWN, cfg.unknown_cell_cost, 0.0)
    cost[(states == OCCUPIED) | (margin < 0)] = np.inf
    return cost.astype(np.float32)


def nearest_traversable(traversable: np.ndarray, rc: tuple[int, int], radius: int) -> tuple | None:
    r0, c0 = rc
    n_r, n_c = traversable.shape
    best, best_d = None, math.inf
    for r in range(max(0, r0 - radius), min(n_r, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(n_c, c0 + radius + 1)):
            if traversable[r, c]:
                d = (r - r0) ** 2 + (c - c0) ** 2
                if d < best_d:
                    best, best_d = (r, c), d
    return best


def goal_costs(
    grid: TraversabilityMap,
    traversable: np.ndarray,
    states: np.ndarray,
    user_xz: np.ndarray,
    heading_xz: np.ndarray,
    cfg: ValueConfig,
) -> np.ndarray:
    """Initial cost of each candidate goal cell (inf if not a goal).

    Goals are reachable-looking cells at the edge of what is known (frontier or map border)
    ahead of the user. Misalignment with the walking direction and being closer than the
    preferred distance add cost, so the route reaches as far as the evidence allows.
    """
    n = states.shape[0]
    rows, cols = np.indices(states.shape)
    centers = grid.cell_to_world(np.stack([rows, cols], axis=-1))
    rel = centers - user_xz
    dist = np.linalg.norm(rel, axis=-1)
    cos_angle = (rel @ heading_xz) / np.maximum(dist, 1e-6)

    unknown = states == UNKNOWN
    near_unknown = ndimage.binary_dilation(unknown, structure=np.ones((3, 3), bool))
    border = np.zeros_like(unknown)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    frontier = (states == FREE) & (near_unknown | border)

    candidate = (
        frontier
        & traversable
        & (dist >= cfg.goal_min_distance_m)
        & (cos_angle >= math.cos(math.radians(cfg.goal_max_angle_deg)))
    )
    out = np.full((n, n), np.inf, dtype=np.float64)
    shortfall = np.maximum(0.0, cfg.goal_preferred_distance_m - dist[candidate])
    out[candidate] = (
        cfg.goal_alignment_cost_m * (1.0 - cos_angle[candidate]) + cfg.goal_near_penalty * shortfall
    )
    return out


def solve_cost_to_go(
    cost: np.ndarray, goal_cost: np.ndarray, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-goal Dijkstra on an 8-connected grid.

    Returns (cost_to_go, next_cell) where next_cell is the flat index of the next cell on
    the optimal route to a goal (-1 at goals and unreachable cells).
    """
    n_r, n_c = cost.shape
    size = n_r * n_c
    flat_cost = cost.ravel()
    finite = np.isfinite(flat_cost)
    rows, cols = np.indices(cost.shape)
    rows, cols = rows.ravel(), cols.ravel()

    src_list, dst_list, w_list = [], [], []
    for dr, dc in NEIGHBOURS:
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (r2 < n_r) & (c2 >= 0) & (c2 < n_c)
        a = np.flatnonzero(ok)
        b = r2[ok] * n_c + c2[ok]
        keep = finite[a] & finite[b]
        if dr != 0 and dc != 0:
            corner1 = rows[a] * n_c + c2[ok]
            corner2 = r2[ok] * n_c + cols[a]
            keep &= finite[corner1] & finite[corner2]
        a, b = a[keep], b[keep]
        step = cell_size_m * math.hypot(dr, dc)
        src_list.append(a)
        dst_list.append(b)
        w_list.append(step * 0.5 * (flat_cost[a] + flat_cost[b]))

    goals = np.flatnonzero(np.isfinite(goal_cost.ravel()) & finite)
    super_source = size
    src_list.append(np.full(goals.size, super_source))
    dst_list.append(goals)
    w_list.append(goal_cost.ravel()[goals] + 1e-9)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    w = np.concatenate(w_list)
    graph = csr_matrix((w, (src, dst)), shape=(size + 1, size + 1))
    # Grid edges are symmetric, so the distance from the super source equals the cost-to-go
    # and each predecessor is the next cell on the route towards a goal.
    dist, pred = dijkstra(graph, directed=True, indices=super_source, return_predecessors=True)
    cost_to_go = dist[:size].reshape(n_r, n_c)
    next_cell = pred[:size].astype(np.int64)
    next_cell[(next_cell == super_source) | (next_cell < 0)] = -1
    return cost_to_go, next_cell.reshape(n_r, n_c)


def demand_ratio_field(
    next_cell: np.ndarray,
    clearance_m: np.ndarray,
    cell_size_m: float,
    cfg: ValueConfig,
) -> np.ndarray:
    """Per-cell spatial demand r(s) = max(turn ratio, narrowness ratio) along the optimal route."""
    n_r, n_c = next_cell.shape
    size = n_r * n_c
    flat_next = next_cell.ravel()
    idx = np.arange(size)

    def advance(start: np.ndarray, steps: int) -> np.ndarray:
        cur = start.copy()
        for _ in range(steps):
            nxt = flat_next[cur]
            cur = np.where(nxt >= 0, nxt, cur)
        return cur

    k = max(1, cfg.field_turn_probe_cells)
    p1 = advance(idx, k)
    p2 = advance(p1, k)
    r0, c0 = np.divmod(idx, n_c)
    r1, c1 = np.divmod(p1, n_c)
    r2, c2 = np.divmod(p2, n_c)
    d1 = np.stack([c1 - c0, r1 - r0], axis=-1).astype(np.float64)
    d2 = np.stack([c2 - c1, r2 - r1], axis=-1).astype(np.float64)
    n1 = np.linalg.norm(d1, axis=-1)
    n2 = np.linalg.norm(d2, axis=-1)
    valid = (n1 > 0) & (n2 > 0)
    cos_t = np.where(valid, np.sum(d1 * d2, axis=-1) / np.maximum(n1 * n2, 1e-9), 1.0)
    turn = turn_ratio(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))), cfg)

    width = 2.0 * np.minimum(clearance_m.ravel(), 10.0)
    narrow = narrowness_ratio(width, cfg)
    return np.maximum(turn, narrow).reshape(n_r, n_c)


def turn_ratio(turn_deg: np.ndarray, cfg: ValueConfig) -> np.ndarray:
    """Generalised angle ratio (paper Eq. 29): 1.0 at a 90 degree turn, 0 inside the deadband."""
    span = 90.0 - cfg.turn_deadband_deg
    return np.clip((np.asarray(turn_deg) - cfg.turn_deadband_deg) / span, 0.0, 2.0)


def narrowness_ratio(width_m: np.ndarray, cfg: ValueConfig) -> np.ndarray:
    span = max(cfg.comfort_width_m - cfg.min_width_m, 1e-6)
    return np.clip((cfg.comfort_width_m - np.asarray(width_m)) / span, 0.0, 1.5)


def value_canyon_kappa(profile: PatientProfile) -> float:
    return profile.value_canyon_gain * profile.fog_factor


def compute_value_field(
    grid: TraversabilityMap,
    user_xz: np.ndarray,
    heading_xz: np.ndarray,
    cfg: ValueConfig,
    profile: PatientProfile,
) -> ValueField:
    cs = grid.config.cell_size_m
    states = grid.states()
    clearance = clearance_map(states, cs)
    cost = traversal_cost(states, clearance, cfg)
    traversable = np.isfinite(cost)

    user_rc = tuple(int(v) for v in grid.world_to_cell(np.asarray(user_xz)))
    start = None
    if grid.in_bounds(np.array(user_rc)):
        start = nearest_traversable(traversable, user_rc, radius=int(round(0.5 / cs)))

    g_cost = goal_costs(grid, traversable, states, np.asarray(user_xz), heading_xz, cfg)
    cost_to_go, next_cell = solve_cost_to_go(cost, g_cost, cs)

    log_gamma = math.log(cfg.discount_per_m)
    value_hc = np.where(np.isfinite(cost_to_go), np.exp(log_gamma * cost_to_go), 0.0)
    demand = demand_ratio_field(next_cell, clearance, cs, cfg)
    value_pd = value_hc * np.exp(-value_canyon_kappa(profile) * demand)

    return ValueField(
        states=states,
        clearance_m=clearance,
        cost=cost,
        traversable=traversable,
        cost_to_go=cost_to_go,
        next_cell=next_cell,
        goal_mask=np.isfinite(g_cost),
        value_hc=value_hc.astype(np.float32),
        demand=demand.astype(np.float32),
        value_pd=value_pd.astype(np.float32),
        start_cell=start,
        discount_per_m=cfg.discount_per_m,
    )
