import math

import numpy as np

from cafog.sources.monocular import (
    FloorAnchoredOdometry,
    fit_floor_plane,
    levelled_rotation,
    rotation_between,
    up_hint_from_pitch,
)
from cafog.sources.synthetic import SIM_INTRINSICS, corridor_turn, walk


def first_frame(pitch_deg=20.0, eye=1.6):
    return next(iter(walk(corridor_turn(90), pitch_deg=pitch_deg, eye_height_m=eye)))


def test_floor_plane_recovers_height_and_tilt():
    frame = first_frame(pitch_deg=20.0, eye=1.6)
    plane = fit_floor_plane(frame.depth_m, SIM_INTRINSICS, up_hint_from_pitch(30.0))
    assert plane is not None
    assert abs(plane.height_m - 1.6) < 0.03
    tilt = math.degrees(math.acos(-plane.normal_cam[1]))
    assert abs(tilt - 20.0) < 1.5


def test_levelled_rotation_maps_floor_normal_to_world_up():
    n = up_hint_from_pitch(25.0)
    r = levelled_rotation(n)
    assert np.allclose(r @ n, [0, 1, 0], atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)
    forward_world = r @ np.array([0.0, 0.0, 1.0])
    assert abs(forward_world[0]) < 1e-9 and forward_world[2] > 0


def test_rotation_between_aligns_vectors():
    a = np.array([0.1, 0.9, 0.2])
    a /= np.linalg.norm(a)
    b = np.array([0.0, 1.0, 0.0])
    assert np.allclose(rotation_between(a, b) @ a, b, atol=1e-9)


def test_odometry_initialises_on_the_floor():
    frame = first_frame()
    odo = FloorAnchoredOdometry(SIM_INTRINSICS)
    gray = np.zeros((SIM_INTRINSICS.height, SIM_INTRINSICS.width), np.uint8)
    pose, status = odo.update(gray, frame.depth_m)
    assert status.floor_found and pose is not None
    assert abs(pose[1, 3] - 1.6) < 0.03
    # Camera looks down by ~20 degrees: its optical axis has a negative world-y component.
    forward = pose[:3, 2]
    assert abs(math.degrees(math.asin(-forward[1])) - 20.0) < 1.5


def test_known_camera_height_fixes_depth_scale():
    frame = first_frame(pitch_deg=25.0, eye=1.4)
    odo = FloorAnchoredOdometry(SIM_INTRINSICS, camera_height_m=1.4)
    gray = np.zeros((SIM_INTRINSICS.height, SIM_INTRINSICS.width), np.uint8)
    pose, status = odo.update(gray, frame.depth_m * 2.0)
    assert pose is not None
    assert abs(status.depth_scale - 0.5) < 0.02
    assert abs(pose[1, 3] - 1.4) < 0.03
