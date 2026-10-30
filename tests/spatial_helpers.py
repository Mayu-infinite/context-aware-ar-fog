import numpy as np

from cafog.spatial.config import ValueConfig
from cafog.spatial.path import PredictedPath
from cafog.spatial.value import narrowness_ratio, turn_ratio


def straight_path(length=4.0, turn_deg_at=None, width=2.0) -> PredictedPath:
    """Synthetic predicted path along +z; `turn_deg_at=(s, deg)` adds a turn bump at arc s."""
    cfg = ValueConfig()
    arc = np.arange(0, length + 1e-9, 0.1)
    turn = np.zeros_like(arc)
    if turn_deg_at is not None:
        s, deg = turn_deg_at
        turn = deg * np.exp(-(((arc - s) / 0.4) ** 2))
    widths = np.full_like(arc, width)
    return PredictedPath(
        points=np.stack([np.zeros_like(arc), arc], -1),
        arc_m=arc,
        tangents=np.tile([0.0, 1.0], (len(arc), 1)),
        headings=np.full_like(arc, np.pi / 2),
        turn_deg=turn,
        turn_ratio=turn_ratio(turn, cfg),
        width_m=widths,
        narrow_ratio=narrowness_ratio(widths, cfg),
        cost_to_go=12.0 - arc,
        cost=np.ones_like(arc),
        known_free=np.ones_like(arc, bool),
        clearance_m=widths / 2,
    )
