"""Replay ScanNet RGB-D sequences (https://github.com/ScanNet/ScanNet) through the engine.

Expects the layout written by ScanNet's SensReader (`reader.py --export_depth_images
--export_color_images --export_poses --export_intrinsics`):

    scene0000_00/
        color/0.jpg ...           depth/0.png ...  (uint16, millimetres)
        pose/0.txt ...            (4x4 camera-to-world, z-up world)
        intrinsic/intrinsic_depth.txt
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from cafog.spatial.engine import DepthFrame
from cafog.spatial.geometry import CameraIntrinsics

SCANNET_FPS = 30.0
Z_UP_TO_Y_UP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def read_intrinsics(scene_dir: Path, width: int, height: int) -> CameraIntrinsics:
    k = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_depth.txt")
    return CameraIntrinsics(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        width=width,
        height=height,
    )


def frame_ids(scene_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in (scene_dir / "depth").glob("*.png"))


def iter_scannet(
    scene_dir: str | Path,
    *,
    every: int = 3,
    start: int = 0,
    limit: int | None = None,
    with_color: bool = True,
) -> Iterator[DepthFrame]:
    scene_dir = Path(scene_dir)
    ids = frame_ids(scene_dir)[start::every]
    if limit is not None:
        ids = ids[:limit]
    if not ids:
        raise FileNotFoundError(f"No depth frames under {scene_dir / 'depth'}")

    intrinsics = None
    for i in ids:
        pose = np.loadtxt(scene_dir / "pose" / f"{i}.txt")
        if not np.all(np.isfinite(pose)):
            continue
        depth_raw = cv2.imread(str(scene_dir / "depth" / f"{i}.png"), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth = depth_raw.astype(np.float32) / 1000.0
        depth[depth_raw == 0] = np.nan
        if intrinsics is None:
            intrinsics = read_intrinsics(scene_dir, depth.shape[1], depth.shape[0])
        color = None
        if with_color:
            color_path = scene_dir / "color" / f"{i}.jpg"
            if color_path.exists():
                color = cv2.resize(cv2.imread(str(color_path)), (depth.shape[1], depth.shape[0]))
        yield DepthFrame(
            depth_m=depth,
            intrinsics=intrinsics,
            camera_to_world=Z_UP_TO_Y_UP @ pose,
            timestamp_s=i / SCANNET_FPS,
            color=color,
        )
