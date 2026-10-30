"""Camera and ground-plane geometry.

World frame: right-handed, y up, ground plane spanned by (x, z).
Camera frame: OpenCV convention (x right, y down, z forward).
Poses are 4x4 camera-to-world transforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled(self, width: int, height: int) -> CameraIntrinsics:
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
        )

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_deg: float) -> CameraIntrinsics:
        fx = 0.5 * width / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        return cls(
            fx=fx, fy=fx, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0, width=width, height=height
        )


def pixel_rays(intrinsics: CameraIntrinsics, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled pixel coordinates (v, u) and camera rays with z == 1."""
    vs = np.arange(0, intrinsics.height, stride, dtype=np.float32)
    us = np.arange(0, intrinsics.width, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    rays = np.stack(
        [
            (uu - intrinsics.cx) / intrinsics.fx,
            (vv - intrinsics.cy) / intrinsics.fy,
            np.ones_like(uu),
        ],
        axis=-1,
    )
    return np.stack([vv, uu], axis=-1).astype(np.int32), rays


def depth_to_world_points(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_to_world: np.ndarray,
    *,
    stride: int = 1,
    max_range_m: float | None = None,
    min_range_m: float = 0.1,
    return_pixels: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Back-project a metric z-depth image into world points of shape (N, 3).

    With return_pixels=True also returns the (row, col) pixel of every point.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth shape {depth.shape} does not match intrinsics "
            f"({intrinsics.height}, {intrinsics.width})."
        )
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera_to_world must be a 4x4 matrix.")

    pixels, rays = pixel_rays(intrinsics, stride)
    z = depth[pixels[..., 0], pixels[..., 1]]
    valid = np.isfinite(z) & (z > min_range_m)
    if max_range_m is not None:
        valid &= z < max_range_m
    cam = rays[valid] * z[valid, None]
    world = (cam @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    return (world, pixels[valid]) if return_pixels else world


def ground_heading(camera_to_world: np.ndarray) -> np.ndarray | None:
    """Unit (x, z) direction of the camera's optical axis projected on the ground."""
    forward = np.asarray(camera_to_world, dtype=np.float64)[:3, 2]
    planar = np.array([forward[0], forward[2]])
    norm = float(np.linalg.norm(planar))
    if norm < 0.2:
        return None
    return planar / norm


def right_of(direction: np.ndarray) -> np.ndarray:
    """Right-hand lateral (x, z) vector for a walking direction in a y-up right-handed world."""
    d = np.asarray(direction, dtype=np.float64)
    return np.stack([-d[..., 1], d[..., 0]], axis=-1)


def heading_angle(direction: np.ndarray) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    return np.arctan2(d[..., 1], d[..., 0])


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def look_at_pose(position: np.ndarray, heading_xz: np.ndarray, pitch_deg: float) -> np.ndarray:
    """Camera-to-world pose for a camera at `position` facing `heading_xz`, pitched down."""
    h = np.asarray(heading_xz, dtype=np.float64)
    h = h / np.linalg.norm(h)
    pitch = math.radians(pitch_deg)
    forward = np.array([h[0] * math.cos(pitch), -math.sin(pitch), h[1] * math.cos(pitch)])
    up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose
