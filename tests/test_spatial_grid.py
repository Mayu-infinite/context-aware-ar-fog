
import numpy as np
import pytest

from cafog.perception.spatial_grid import (
    NO_OBJECT_EVIDENCE,
    OBJECT_EVIDENCE,
    UNKNOWN,
    add_object_box,
    create_spatial_grid,
    grid_summary,
    mark_no_object_evidence,
)


def test_grid_starts_unknown():
    grid = create_spatial_grid(320, 240, 80, 80)

    assert grid.cells.shape == (3, 4)
    assert np.all(grid.cells == UNKNOWN)


def test_object_box_marks_overlapping_cells():
    grid = create_spatial_grid(320, 240, 80, 80)

    add_object_box(grid, 0, 0, 80, 80)

    assert grid.cells[0, 0] == OBJECT_EVIDENCE
    assert np.count_nonzero(grid.cells == OBJECT_EVIDENCE) == 1


def test_object_box_spanning_cells():
    grid = create_spatial_grid(320, 240, 80, 80)

    add_object_box(grid, 70, 70, 170, 170)

    assert np.count_nonzero(grid.cells == OBJECT_EVIDENCE) == 9


def test_box_is_clipped_to_image():
    grid = create_spatial_grid(320, 240, 80, 80)

    add_object_box(grid, -20, -20, 50, 50)

    assert grid.cells[0, 0] == OBJECT_EVIDENCE
    assert np.count_nonzero(grid.cells == OBJECT_EVIDENCE) == 1


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError):
        create_spatial_grid(0, 240)

    with pytest.raises(ValueError):
        create_spatial_grid(320, 240, 0, 80)


def test_non_finite_box_raises():
    grid = create_spatial_grid(320, 240, 80, 80)

    with pytest.raises(ValueError):
        add_object_box(grid, np.nan, 0, 80, 80)


def test_mark_no_evidence_keeps_object_cells():
    grid = create_spatial_grid(160, 160, 80, 80)
    add_object_box(grid, 0, 0, 80, 80)

    mark_no_object_evidence(grid)

    assert grid.cells[0, 0] == OBJECT_EVIDENCE
    assert np.all(grid.cells[1:, :] == NO_OBJECT_EVIDENCE)
    assert np.all(grid.cells[0, 1:] == NO_OBJECT_EVIDENCE)


def test_grid_summary():
    grid = create_spatial_grid(160, 160, 80, 80)
    add_object_box(grid, 0, 0, 80, 80)

    summary = grid_summary(grid)

    assert summary["total_cells"] == 4
    assert summary["object_evidence_cells"] == 1
    assert summary["unknown_cells"] == 3
    assert summary["object_evidence_fraction"] == 0.25
