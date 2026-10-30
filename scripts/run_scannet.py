"""Replay a ScanNet scene through the label-free cue engine.

Outputs a researcher video (camera view with floor heatmap + footsteps, top-down value map),
a per-frame CSV log, and a coloured 3D point model (PLY) of the scanned space.

    uv run python scripts/run_scannet.py data/scannet/scene0000_00 --profile pd_fog
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from cafog.sources.scannet import SCANNET_FPS, iter_scannet
from cafog.spatial import EngineConfig, SpatialCueEngine, make_profile
from cafog.spatial.geometry import depth_to_world_points
from cafog.spatial.viz import compose
from cafog.spatial.voxels import VoxelMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--profile", default="pd_fog")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--every", type=int, default=3, help="use every Nth frame (30 fps source)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/exp006_scannet/results"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.04)
    args = parser.parse_args()

    config = EngineConfig.from_yaml(args.config) if args.config else EngineConfig()
    config.profile = make_profile(args.profile)
    engine = SpatialCueEngine(config)
    voxels = VoxelMap(args.voxel_size)

    out = args.out / args.scene.name
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = []
    for frame in iter_scannet(args.scene, every=args.every, limit=args.limit):
        result = engine.process(frame)
        points, pixels = depth_to_world_points(
            frame.depth_m,
            frame.intrinsics,
            frame.camera_to_world,
            stride=2,
            max_range_m=config.map.max_range_m,
            return_pixels=True,
        )
        colors = frame.color[pixels[:, 0], pixels[:, 1]] if frame.color is not None else None
        voxels.add(points, colors)

        hazard = result.cue.hazard
        rows.append(
            {
                "t": round(frame.timestamp_s, 3),
                "x": round(float(result.user_xz[0]), 3),
                "z": round(float(result.user_xz[1]), 3),
                "speed": round(result.speed_mps, 3),
                "risk_3m": round(result.risk.max_within(3.0), 3) if result.risk else "",
                "phase": result.cue.phase.value,
                "footsteps": len(result.cue.footsteps),
                "hazard": hazard.kind.value if hazard else "",
                "hazard_m": round(hazard.distance_m, 2) if hazard else "",
                "turn_deg": round(hazard.turn_deg) if hazard else "",
                "width_m": round(hazard.width_m, 2) if hazard else "",
                "latency_ms": round(result.timings_ms["total"], 2),
            }
        )
        if not args.no_video:
            panel = compose(frame, result, engine.map)
            if writer is None:
                h, w = panel.shape[:2]
                writer = cv2.VideoWriter(
                    str(out / "cues.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    SCANNET_FPS / args.every,
                    (w, h),
                )
            writer.write(panel)

    if writer is not None:
        writer.release()
    with (out / "frames.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    n_points = voxels.export_ply(out / "model.ply")
    latency = np.array([r["latency_ms"] for r in rows])
    active = sum(r["phase"] in ("ACTIVE", "RELEASING") for r in rows)
    print(
        f"{len(rows)} frames, cue active in {active}, latency mean {latency.mean():.1f} ms "
        f"p95 {np.percentile(latency, 95):.1f} ms, 3D model {n_points} voxels -> {out}"
    )


if __name__ == "__main__":
    main()
