"""Single RGB camera (phone, webcam, video file) -> metric depth + pose for the engine.

A headset measures depth and head pose directly; a phone camera does not, so:
  * depth comes from a pretrained metric depth network (no training or fine-tuning),
  * gravity and camera height come from the floor plane fitted in every frame (drift-free),
  * heading and walking motion come from frame-to-frame visual odometry (ORB + PnP on the
    previous frame's metric depth), re-levelled on the floor plane each frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from cafog.spatial.geometry import CameraIntrinsics

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class MetricDepthModel:
    """Pretrained monocular metric depth (metres, OpenCV z-depth) at the requested size."""

    def __init__(
        self,
        model_id: str = DEFAULT_DEPTH_MODEL,
        *,
        input_size: int = 308,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        self.input_size = int(round(input_size / 14) * 14)

    def __call__(self, bgr: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        """Depth map of shape (out_height, out_width) for out_size = (width, height)."""
        torch = self._torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            images=rgb,
            return_tensors="pt",
            size={"height": self.input_size, "width": self.input_size},
        ).to(self.device)
        with torch.inference_mode():
            predicted = self.model(**inputs).predicted_depth
            depth = torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=(out_size[1], out_size[0]),
                mode="bilinear",
                align_corners=False,
            )
        return depth[0, 0].float().cpu().numpy()


def up_hint_from_pitch(pitch_down_deg: float) -> np.ndarray:
    """World 'up' in OpenCV camera coordinates for a camera pitched down by the given angle."""
    p = math.radians(pitch_down_deg)
    return np.array([0.0, -math.cos(p), -math.sin(p)])


@dataclass(frozen=True)
class FloorPlane:
    normal_cam: np.ndarray  # unit, pointing up, in camera coordinates
    height_m: float  # camera height above the floor
    inlier_fraction: float


def fit_floor_plane(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    up_hint_cam: np.ndarray,
    *,
    stride: int = 4,
    max_tilt_deg: float = 30.0,
    inlier_m: float = 0.04,
    inlier_per_m: float = 0.02,
    max_range_m: float = 8.0,
    iterations: int = 150,
    min_inlier_fraction: float = 0.08,
    height_range_m: tuple[float, float] = (0.3, 2.2),
    below_penalty: int = 3,
    seed: int = 0,
) -> FloorPlane | None:
    """RANSAC plane through the lower image whose normal is close to the expected 'up'.

    The inlier band widens with range (inlier_m + inlier_per_m * z) because monocular depth
    error grows with distance; a level phone only sees the floor from ~3-4 m onwards.
    Hypotheses are penalised for points below them: the floor is the lowest surface.
    """
    h, w = depth_m.shape
    vs, us = np.mgrid[int(0.4 * h) : h : stride, 0:w:stride]
    z = depth_m[vs, us]
    ok = np.isfinite(z) & (z > 0.3) & (z < max_range_m)
    if ok.sum() < 50:
        return None
    z, us, vs = z[ok], us[ok], vs[ok]
    pts = np.stack(
        [(us - intrinsics.cx) / intrinsics.fx * z, (vs - intrinsics.cy) / intrinsics.fy * z, z],
        axis=1,
    )
    up = np.asarray(up_hint_cam, dtype=np.float64)
    up = up / np.linalg.norm(up)
    cos_tol = math.cos(math.radians(max_tilt_deg))
    band = inlier_m + inlier_per_m * z
    rng = np.random.default_rng(seed)
    best_score, best_count, best_mask = 0, 0, None
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n @ up < 0:
            n = -n
        if n @ up < cos_tol:
            continue
        d = -float(n @ a)
        if not height_range_m[0] <= d <= height_range_m[1]:
            continue
        signed = pts @ n + d
        mask = np.abs(signed) < band
        score = int(mask.sum()) - below_penalty * int((signed < -band).sum())
        if score > best_score:
            best_score, best_count, best_mask = score, int(mask.sum()), mask
    if best_mask is None or best_count < min_inlier_fraction * len(pts):
        return None

    inliers = pts[best_mask]
    centroid = inliers.mean(axis=0)
    n = np.linalg.svd(inliers - centroid, full_matrices=False)[2][-1]
    if n @ up < 0:
        n = -n
    height = float(-(n @ centroid))
    if not height_range_m[0] <= height <= height_range_m[1]:
        return None
    return FloorPlane(n, height, best_count / len(pts))


def levelled_rotation(normal_cam: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation with world y = floor normal and world z = camera forward on ground.

    Its rows are the world axes expressed in camera coordinates.
    """
    n = normal_cam / np.linalg.norm(normal_cam)
    forward = np.array([0.0, 0.0, 1.0]) - n[2] * n
    forward /= np.linalg.norm(forward)
    right = np.cross(n, forward)
    return np.stack([right, n, forward])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


@dataclass
class OdometryStatus:
    tracking: bool
    floor_found: bool
    inliers: int
    camera_height_m: float | None
    depth_scale: float = 1.0


class FloorAnchoredOdometry:
    """Camera-to-world pose from metric depth + RGB, with the floor at world y = 0.

    With camera_height_m set, the network depth is rescaled every frame so that the fitted
    floor lies that far below the camera (ground-plane scale recovery); the caller must
    multiply its depth by status.depth_scale. Without it the network's metric scale is kept.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        camera_height_m: float | None = None,
        scale_smoothing: float = 0.15,
        initial_pitch_deg: float = 20.0,
        acquire_tilt_deg: float = 40.0,
        track_tilt_deg: float = 20.0,
        features: int = 1200,
        min_inliers: int = 20,
        max_step_m: float = 0.6,
        height_smoothing: float = 0.3,
    ):
        self.K = intrinsics
        self.k_mat = np.array(
            [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]],
            dtype=np.float64,
        )
        self.up_hint = up_hint_from_pitch(initial_pitch_deg)
        self.acquire_tilt_deg = acquire_tilt_deg
        self.track_tilt_deg = track_tilt_deg
        self.orb = cv2.ORB_create(nfeatures=features, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_inliers = min_inliers
        self.max_step_m = max_step_m
        self.height_smoothing = height_smoothing
        self.camera_height_m = camera_height_m
        self.scale_smoothing = scale_smoothing
        self.depth_scale = 1.0
        self._height_range = (0.3, 5.0) if camera_height_m else (0.3, 2.2)
        self.pose: np.ndarray | None = None
        self._prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def update(
        self, gray: np.ndarray, depth_m: np.ndarray
    ) -> tuple[np.ndarray | None, OdometryStatus]:
        """gray and depth_m must share the intrinsics' resolution."""
        hint, tilt = (
            (self.up_hint, self.acquire_tilt_deg)
            if self.pose is None
            else (self.pose[1, :3], self.track_tilt_deg)
        )
        floor = fit_floor_plane(
            depth_m, self.K, hint, max_tilt_deg=tilt, height_range_m=self._height_range
        )
        if floor is not None and self.camera_height_m:
            frame_scale = self.camera_height_m / floor.height_m
            a = self.scale_smoothing if self.pose is not None else 1.0
            self.depth_scale = float(
                np.exp((1 - a) * np.log(self.depth_scale) + a * np.log(frame_scale))
            )
        if floor is not None:
            floor = FloorPlane(
                floor.normal_cam, floor.height_m * self.depth_scale, floor.inlier_fraction
            )
        depth_m = depth_m * self.depth_scale
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        tracking, inliers = False, 0

        if self.pose is None:
            if floor is None:
                return None, OdometryStatus(False, False, 0, None, self.depth_scale)
            pose = np.eye(4)
            pose[:3, :3] = levelled_rotation(floor.normal_cam)
            pose[1, 3] = floor.height_m
            self.pose = pose
            tracking = True
        else:
            tracked = self._track(keypoints, descriptors)
            if tracked is not None:
                self.pose, inliers = tracked
                tracking = True
            if floor is not None:
                self._relevel(floor)

        if descriptors is not None and len(keypoints) > 0:
            uv = np.array([kp.pt for kp in keypoints], dtype=np.float64)
            self._prev = (uv, descriptors, depth_m)
        status = OdometryStatus(
            tracking,
            floor is not None,
            inliers,
            floor.height_m if floor else None,
            self.depth_scale,
        )
        return self.pose.copy(), status

    def _track(self, keypoints, descriptors) -> tuple[np.ndarray, int] | None:
        if self._prev is None or descriptors is None or len(keypoints) < self.min_inliers:
            return None
        prev_uv, prev_desc, prev_depth = self._prev
        pairs = self.matcher.knnMatch(prev_desc, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < self.min_inliers:
            return None
        src = prev_uv[[m.queryIdx for m in good]]
        dst = np.array([keypoints[m.trainIdx].pt for m in good], dtype=np.float64)
        u = np.clip(np.round(src[:, 0]).astype(int), 0, prev_depth.shape[1] - 1)
        v = np.clip(np.round(src[:, 1]).astype(int), 0, prev_depth.shape[0] - 1)
        z = prev_depth[v, u]
        ok = np.isfinite(z) & (z > 0.3) & (z < 6.0)
        if ok.sum() < self.min_inliers:
            return None
        cam = np.stack(
            [
                (src[ok, 0] - self.K.cx) / self.K.fx * z[ok],
                (src[ok, 1] - self.K.cy) / self.K.fy * z[ok],
                z[ok],
            ],
            axis=1,
        )
        world = cam @ self.pose[:3, :3].T + self.pose[:3, 3]

        world_to_cam = np.linalg.inv(self.pose)
        rvec, _ = cv2.Rodrigues(world_to_cam[:3, :3])
        tvec = world_to_cam[:3, 3].reshape(3, 1)
        found, rvec, tvec, inl = cv2.solvePnPRansac(
            world,
            dst[ok],
            self.k_mat,
            None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.99,
        )
        if not found or inl is None or len(inl) < self.min_inliers:
            return None
        r_cw, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = r_cw.T
        pose[:3, 3] = (-r_cw.T @ tvec).ravel()
        if np.linalg.norm(pose[[0, 2], 3] - self.pose[[0, 2], 3]) > self.max_step_m:
            return None
        return pose, int(len(inl))

    def _relevel(self, floor: FloorPlane) -> None:
        r_wc = self.pose[:3, :3]
        correction = rotation_between(r_wc @ floor.normal_cam, np.array([0.0, 1.0, 0.0]))
        self.pose[:3, :3] = correction @ r_wc
        a = self.height_smoothing
        self.pose[1, 3] = (1 - a) * self.pose[1, 3] + a * floor.height_m
