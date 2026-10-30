import cv2
import numpy as np

from cafog.sources.scannet import Z_UP_TO_Y_UP, iter_scannet
from cafog.sources.synthetic import SIM_INTRINSICS, corridor_turn, walk
from cafog.spatial import EngineConfig, SpatialCueEngine, make_profile
from cafog.spatial.voxels import VoxelMap


def write_fake_scannet(root, frames):
    for sub in ("depth", "pose", "intrinsic", "color"):
        (root / sub).mkdir(parents=True)
    k = SIM_INTRINSICS
    intrinsic = np.eye(4)
    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = k.fx, k.fy, k.cx, k.cy
    np.savetxt(root / "intrinsic" / "intrinsic_depth.txt", intrinsic)
    to_z_up = np.linalg.inv(Z_UP_TO_Y_UP)
    for i, frame in enumerate(frames):
        mm = np.nan_to_num(frame.depth_m * 1000.0, nan=0, posinf=0).astype(np.uint16)
        cv2.imwrite(str(root / "depth" / f"{i * 3}.png"), mm)
        cv2.imwrite(str(root / "color" / f"{i * 3}.jpg"), np.full((*mm.shape, 3), 128, np.uint8))
        np.savetxt(root / "pose" / f"{i * 3}.txt", to_z_up @ frame.camera_to_world)


def test_scannet_replay_matches_direct_engine(tmp_path):
    frames = list(walk(corridor_turn(90), fps=10))
    write_fake_scannet(tmp_path, frames)
    replayed = list(iter_scannet(tmp_path, every=1))
    assert len(replayed) == len(frames)
    assert np.allclose(replayed[5].camera_to_world, frames[5].camera_to_world, atol=1e-6)
    assert replayed[0].color is not None

    engine = SpatialCueEngine(EngineConfig(profile=make_profile("pd_fog")))
    phases = [engine.process(f).cue.phase.value for f in replayed]
    assert "ACTIVE" in phases


def test_voxel_map_merges_and_exports(tmp_path):
    vox = VoxelMap(0.1)
    pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vox.add(pts, np.array([[0, 0, 255], [0, 0, 255], [255, 0, 0]]))
    vox.add(pts[:1], np.array([[0, 0, 255]]))
    assert len(vox) == 2
    points, colors = vox.points_and_colors(min_count=2)
    assert len(points) == 1 and tuple(colors[0]) == (0, 0, 255)
    assert vox.export_ply(tmp_path / "m.ply", min_count=1) == 2
    assert (tmp_path / "m.ply").read_bytes().startswith(b"ply")
