"""Coloured voxel reconstruction of everything the headset has seen (for review, not for cueing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

KEY_BITS = 21
KEY_OFFSET = 1 << (KEY_BITS - 1)


class VoxelMap:
    def __init__(self, voxel_size_m: float = 0.05):
        self.voxel_size_m = voxel_size_m
        self.keys = np.zeros(0, dtype=np.int64)
        self.sums = np.zeros((0, 3), dtype=np.float64)
        self.color_sums = np.zeros((0, 3), dtype=np.float64)
        self.counts = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.keys.size)

    def _encode(self, points: np.ndarray) -> np.ndarray:
        idx = np.floor(points / self.voxel_size_m).astype(np.int64) + KEY_OFFSET
        mask = (1 << KEY_BITS) - 1
        return (
            ((idx[:, 0] & mask) << (2 * KEY_BITS))
            | ((idx[:, 1] & mask) << KEY_BITS)
            | (idx[:, 2] & mask)
        )

    def add(self, points: np.ndarray, colors_bgr: np.ndarray | None = None) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return
        colors = (
            np.full((len(points), 3), 180.0)
            if colors_bgr is None
            else np.asarray(colors_bgr, dtype=np.float64)
        )
        keys = np.concatenate([self.keys, self._encode(points)])
        uniq, inverse = np.unique(keys, return_inverse=True)
        n = len(uniq)
        old = len(self.keys)

        def merge(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
            out = np.zeros((n, existing.shape[1]))
            np.add.at(out, inverse[:old], existing)
            np.add.at(out, inverse[old:], new)
            return out

        counts = np.zeros(n, dtype=np.int64)
        np.add.at(counts, inverse[:old], self.counts)
        np.add.at(counts, inverse[old:], 1)
        self.sums = merge(self.sums, points)
        self.color_sums = merge(self.color_sums, colors)
        self.counts = counts
        self.keys = uniq

    def points_and_colors(self, min_count: int = 2) -> tuple[np.ndarray, np.ndarray]:
        keep = self.counts >= min_count
        c = self.counts[keep, None]
        return self.sums[keep] / c, (self.color_sums[keep] / c).astype(np.uint8)

    def export_ply(self, path: str | Path, min_count: int = 2) -> int:
        points, colors_bgr = self.points_and_colors(min_count)
        vertex = np.empty(
            len(points),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
        )
        vertex["x"], vertex["y"], vertex["z"] = points.T
        vertex["r"], vertex["g"], vertex["b"] = colors_bgr[:, 2], colors_bgr[:, 1], colors_bgr[:, 0]
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        with Path(path).open("wb") as handle:
            handle.write(header.encode("ascii"))
            handle.write(vertex.tobytes())
        return len(points)
