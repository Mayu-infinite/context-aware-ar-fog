from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthGrid:
    """Per-cell relative-depth statistics; not metric distance or traversability."""

    median: np.ndarray
    spread: np.ndarray
    valid_fraction: np.ndarray
    image_width: int
    image_height: int
    cell_width: int
    cell_height: int

    @property
    def rows(self) -> int:
        return int(self.median.shape[0])

    @property
    def cols(self) -> int:
        return int(self.median.shape[1])


def summarize_depth_grid(
    depth: np.ndarray,
    cell_width: int = 80,
    cell_height: int = 80,
) -> DepthGrid:
    """Summarize finite depth predictions in image-space cells.

    `median` and `spread` retain the model's relative output scale.
    `spread` is the median absolute deviation (MAD).
    Cells with no finite samples have NaN statistics.
    """
    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("Depth must be a non-empty 2D array.")

    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("Cell dimensions must be positive.")

    height, width = depth.shape
    cols = (width + cell_width - 1) // cell_width
    rows = (height + cell_height - 1) // cell_height

    medians = np.full((rows, cols), np.nan, dtype=np.float32)
    spreads = np.full((rows, cols), np.nan, dtype=np.float32)
    valid_fractions = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        y1 = row * cell_height
        y2 = min(y1 + cell_height, height)

        for col in range(cols):
            x1 = col * cell_width
            x2 = min(x1 + cell_width, width)

            cell = depth[y1:y2, x1:x2]
            valid = cell[np.isfinite(cell)]
            valid_fractions[row, col] = valid.size / cell.size

            if valid.size == 0:
                continue

            median = float(np.median(valid))
            medians[row, col] = median
            spreads[row, col] = float(np.median(np.abs(valid - median)))

    return DepthGrid(
        median=medians,
        spread=spreads,
        valid_fraction=valid_fractions,
        image_width=width,
        image_height=height,
        cell_width=cell_width,
        cell_height=cell_height,
    )
