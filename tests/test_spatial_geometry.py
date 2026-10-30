import numpy as np
import pytest

from cafog.sources.synthetic import Scene, render_depth
from cafog.spatial.geometry import (
    CameraIntrinsics,
    depth_to_world_points,
    ground_heading,
    look_at_pose,
    right_of,
)


def test_look_at_pose_is_a_proper_rotation_facing_heading():
    pose = look_at_pose(np.array([1.0, 1.6, 2.0]), np.array([0.0, 1.0]), pitch_deg=20)
    rot = pose[:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    assert np.allclose(ground_heading(pose), [0.0, 1.0])
    assert pose[1, 2] < 0  # optical axis points down


def test_right_of_matches_camera_x_axis():
    heading = np.array([0.6, 0.8])
    pose = look_at_pose(np.zeros(3), heading, pitch_deg=0)
    cam_right = pose[[0, 2], 0]
    assert np.allclose(right_of(heading), cam_right)


def test_backprojected_floor_points_lie_on_floor():
    k = CameraIntrinsics.from_fov(64, 48, 90)
    pose = look_at_pose(np.array([0.0, 1.6, 0.0]), np.array([0.0, 1.0]), pitch_deg=30)
    empty = Scene("empty", np.zeros((0, 2, 2)), np.zeros(0), np.zeros((2, 2)))
    depth = render_depth(empty, k, pose)
    points = depth_to_world_points(depth, k, pose, max_range_m=6.0)
    assert len(points) > 100
    assert np.allclose(points[:, 1], 0.0, atol=1e-4)
    assert np.all(points[:, 2] > 0)


def test_depth_shape_mismatch_rejected():
    k = CameraIntrinsics.from_fov(64, 48, 90)
    with pytest.raises(ValueError):
        depth_to_world_points(np.ones((10, 10)), k, np.eye(4))
