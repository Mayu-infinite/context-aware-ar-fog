import numpy as np
import pytest

from cafog.perception.depth_grid import summarize_depth_grid


def test_depth_grid_shape_and_median():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)

    grid = summarize_depth_grid(depth, cell_width=2, cell_height=2)

    assert grid.median.shape == (2, 2)
    assert grid.median[0, 0] == pytest.approx(2.5)
    assert grid.median[1, 1] == pytest.approx(12.5)
    assert np.all(grid.valid_fraction == 1.0)


def test_spread_is_median_absolute_deviation():
    depth = np.array([[1, 2], [3, 100]], dtype=np.float32)

    grid = summarize_depth_grid(depth, cell_width=2, cell_height=2)

    assert grid.median[0, 0] == pytest.approx(2.5)
    assert grid.spread[0, 0] == pytest.approx(1.0)


def test_non_finite_values_are_ignored():
    depth = np.array([[1, np.nan], [3, np.inf]], dtype=np.float32)

    grid = summarize_depth_grid(depth, cell_width=2, cell_height=2)

    assert grid.median[0, 0] == pytest.approx(2.0)
    assert grid.valid_fraction[0, 0] == pytest.approx(0.5)


def test_cell_without_finite_values_stays_unknown():
    depth = np.full((2, 2), np.nan, dtype=np.float32)

    grid = summarize_depth_grid(depth, cell_width=2, cell_height=2)

    assert np.isnan(grid.median[0, 0])
    assert np.isnan(grid.spread[0, 0])
    assert grid.valid_fraction[0, 0] == 0.0


def test_invalid_depth_shape_raises():
    with pytest.raises(ValueError):
        summarize_depth_grid(np.ones(5, dtype=np.float32))


@pytest.mark.parametrize(
    ("cell_width", "cell_height"),
    [(0, 2), (2, 0), (-1, 2)],
)
def test_invalid_cell_size_raises(cell_width, cell_height):
    with pytest.raises(ValueError):
        summarize_depth_grid(
            np.ones((4, 4), dtype=np.float32),
            cell_width=cell_width,
            cell_height=cell_height,
        )


def test_non_divisible_image_dimensions():
    grid = summarize_depth_grid(
        np.ones((5, 7), dtype=np.float32),
        cell_width=4,
        cell_height=3,
    )

    assert grid.median.shape == (2, 2)
    assert np.all(grid.valid_fraction == 1.0)