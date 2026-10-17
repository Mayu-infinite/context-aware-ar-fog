
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = -1
NO_OBJECT_EVIDENCE = 0
OBJECT_EVIDENCE = 1


@dataclass(frozen=True)
class SpatialGrid:
    """Image-space grid containing object evidence, not traversability."""

    cells: np.ndarray
    image_width: int
    image_height: int
    cell_width: int
    cell_height: int

    @property
    def rows(self) -> int:
        return int(self.cells.shape[0])

    @property
    def cols(self) -> int:
        return int(self.cells.shape[1])

    @property
    def object_evidence_cells(self) -> int:
        return int(np.count_nonzero(self.cells == OBJECT_EVIDENCE))

    @property
    def unknown_cells(self) -> int:
        return int(np.count_nonzero(self.cells == UNKNOWN))

    @property
    def no_object_evidence_cells(self) -> int:
        return int(np.count_nonzero(self.cells == NO_OBJECT_EVIDENCE))


def create_spatial_grid(
    image_width: int,
    image_height: int,
    cell_width: int = 80,
    cell_height: int = 80,
) -> SpatialGrid:
    """Create a grid whose cells are all unknown initially."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("Cell dimensions must be positive.")

    cols = (image_width + cell_width - 1) // cell_width
    rows = (image_height + cell_height - 1) // cell_height

    cells = np.full((rows, cols), UNKNOWN, dtype=np.int8)

    return SpatialGrid(
        cells=cells,
        image_width=image_width,
        image_height=image_height,
        cell_width=cell_width,
        cell_height=cell_height,
    )


def add_object_box(
    grid: SpatialGrid,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    """Mark grid cells overlapped by a detected image-space bounding box."""
    coords = np.array([x1, y1, x2, y2], dtype=float)

    if not np.isfinite(coords).all():
        raise ValueError("Bounding-box coordinates must be finite.")

    if x2 < x1 or y2 < y1:
        raise ValueError("Bounding-box maximum coordinates must be >= minimum.")

    # Clip to the image boundaries.
    x1 = float(np.clip(x1, 0, grid.image_width))
    x2 = float(np.clip(x2, 0, grid.image_width))
    y1 = float(np.clip(y1, 0, grid.image_height))
    y2 = float(np.clip(y2, 0, grid.image_height))

    if x2 <= x1 or y2 <= y1:
        return

    col_start = int(x1 // grid.cell_width)
    col_end = int(np.ceil(x2 / grid.cell_width)) - 1
    row_start = int(y1 // grid.cell_height)
    row_end = int(np.ceil(y2 / grid.cell_height)) - 1

    col_start = int(np.clip(col_start, 0, grid.cols - 1))
    col_end = int(np.clip(col_end, 0, grid.cols - 1))
    row_start = int(np.clip(row_start, 0, grid.rows - 1))
    row_end = int(np.clip(row_end, 0, grid.rows - 1))

    grid.cells[row_start : row_end + 1, col_start : col_end + 1] = (
        OBJECT_EVIDENCE
    )


def mark_no_object_evidence(grid: SpatialGrid) -> None:
    """Mark remaining cells as having no detected-object evidence.

    This is not proof that these cells are empty or traversable.
    Call this only after processing all detections for the frame.
    """
    grid.cells[grid.cells == UNKNOWN] = NO_OBJECT_EVIDENCE


def grid_summary(grid: SpatialGrid) -> dict[str, int | float]:
    """Return simple counts and proportions for diagnostics."""
    total = int(grid.cells.size)

    return {
        "rows": grid.rows,
        "cols": grid.cols,
        "total_cells": total,
        "object_evidence_cells": grid.object_evidence_cells,
        "unknown_cells": grid.unknown_cells,
        "no_object_evidence_cells": grid.no_object_evidence_cells,
        "object_evidence_fraction": (
            grid.object_evidence_cells / total if total else 0.0
        ),
    }
