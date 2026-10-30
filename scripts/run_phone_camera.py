"""Run the label-free cue engine on a phone camera, webcam or recorded walking video.

    # live phone stream (IP Webcam app on Android)
    uv run python scripts/run_phone_camera.py http://192.168.1.23:8080/video
    # recorded video
    uv run python scripts/run_phone_camera.py data/raw/videos/video01.mp4 --every 3

Shows the camera view with the value heatmap on the floor and the footstep cue, a top-down
map and a status panel. Writes cues.mp4, frames.csv and (with --ply) a coloured 3D model.
Keys: q / Esc quit, space pause.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from cafog.sources.monocular import DEFAULT_DEPTH_MODEL, FloorAnchoredOdometry, MetricDepthModel
from cafog.sources.stream import is_live, iter_frames
from cafog.spatial import DepthFrame, EngineConfig, SpatialCueEngine, make_profile
from cafog.spatial.geometry import CameraIntrinsics, depth_to_world_points
from cafog.spatial.viz import compose
from cafog.spatial.voxels import VoxelMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source", help="stream URL, webcam index (0) or video file")
    parser.add_argument("--profile", default="pd_fog", help="healthy | pd_nonfog | pd_fog")
    parser.add_argument(
        "--config", type=Path, default=None, help="engine YAML (profile is overridden)"
    )
    parser.add_argument(
        "--hfov", type=float, default=66.0, help="camera horizontal field of view (deg)"
    )
    parser.add_argument("--width", type=int, default=320, help="processing width (px)")
    parser.add_argument(
        "--pitch",
        type=float,
        default=20.0,
        help="downward tilt guess (deg); the floor is searched within +-40 deg of it",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.35,
        help="camera height above the floor (m) used to fix the depth scale; 0 = trust network",
    )
    parser.add_argument("--input-size", type=int, default=308, help="depth network input (px)")
    parser.add_argument("--model", default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--every", type=int, default=1, help="video files: use every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/exp008_phone_camera/results"))
    parser.add_argument("--name", default=None, help="output sub-folder (default: source name)")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--ply", action="store_true", help="also export a coloured 3D model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EngineConfig.from_yaml(args.config) if args.config else EngineConfig()
    config.profile = make_profile(args.profile)
    engine = SpatialCueEngine(config)

    print(f"Loading depth model {args.model} ...", flush=True)
    depth_model = MetricDepthModel(args.model, input_size=args.input_size)
    print(f"Depth model on {depth_model.device}", flush=True)

    name = args.name or ("live" if is_live(args.source) else Path(args.source).stem)
    out = args.out / f"{name}_{args.profile}"
    out.mkdir(parents=True, exist_ok=True)
    voxels = VoxelMap(0.05) if args.ply else None

    intrinsics: CameraIntrinsics | None = None
    odometry: FloorAnchoredOdometry | None = None
    writer = None
    rows: list[dict] = []
    paused = False
    t_first: float | None = None
    for i, (bgr, stamp) in enumerate(iter_frames(args.source, every=args.every)):
        if args.max_frames is not None and i >= args.max_frames:
            break
        t_first = stamp if t_first is None else t_first
        t = stamp - t_first
        h = round(bgr.shape[0] * args.width / bgr.shape[1])
        small = cv2.resize(bgr, (args.width, h), interpolation=cv2.INTER_AREA)
        if intrinsics is None:
            intrinsics = CameraIntrinsics.from_fov(args.width, h, args.hfov)
            odometry = FloorAnchoredOdometry(
                intrinsics,
                camera_height_m=args.camera_height or None,
                initial_pitch_deg=args.pitch,
            )

        t0 = time.perf_counter()
        depth = depth_model(small, (args.width, h))
        t1 = time.perf_counter()
        pose, status = odometry.update(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), depth)
        t2 = time.perf_counter()
        depth = depth * status.depth_scale
        if pose is None:
            panel = small.copy()
            cv2.putText(
                panel,
                "Point the camera at the floor ahead",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            if not show(panel, args):
                break
            continue

        frame = DepthFrame(depth, intrinsics, pose, t, color=small)
        result = engine.process(frame)
        t3 = time.perf_counter()
        if voxels is not None:
            pts, px = depth_to_world_points(
                depth, intrinsics, pose, stride=2, max_range_m=4.0, return_pixels=True
            )
            voxels.add(pts, small[px[:, 0], px[:, 1]])

        hazard = result.cue.hazard
        rows.append(
            {
                "t": round(t, 3),
                "x": round(float(result.user_xz[0]), 3),
                "z": round(float(result.user_xz[1]), 3),
                "camera_height_m": round(float(pose[1, 3]), 3),
                "tracking": int(status.tracking),
                "floor_found": int(status.floor_found),
                "depth_scale": round(status.depth_scale, 3),
                "speed": round(result.speed_mps, 3),
                "risk_3m": round(result.risk.max_within(3.0), 3) if result.risk else "",
                "phase": result.cue.phase.value,
                "footsteps": len(result.cue.footsteps),
                "hazard": hazard.kind.value if hazard else "",
                "hazard_m": round(hazard.distance_m, 2) if hazard else "",
                "turn_deg": round(hazard.turn_deg) if hazard else "",
                "width_m": round(hazard.width_m, 2) if hazard else "",
                "depth_ms": round(1e3 * (t1 - t0), 1),
                "odometry_ms": round(1e3 * (t2 - t1), 1),
                "engine_ms": round(1e3 * (t3 - t2), 1),
            }
        )

        panel = compose(frame, result, engine.map)
        cv2.putText(
            panel,
            f"depth {1e3 * (t1 - t0):.0f} ms  odo {1e3 * (t2 - t1):.0f} ms  "
            f"engine {1e3 * (t3 - t2):.0f} ms  track {'ok' if status.tracking else 'LOST'}",
            (8, panel.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if writer is None:
            fps = 5.0 if is_live(args.source) else 30.0 / args.every
            writer = cv2.VideoWriter(
                str(out / "cues.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (panel.shape[1], panel.shape[0]),
            )
        writer.write(panel)
        if not args.no_window:
            cv2.imshow("cafog", panel)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if not rows:
        print("No frames with a detected floor; nothing written.")
        return
    with (out / "frames.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    if voxels is not None:
        print(f"3D model: {voxels.export_ply(out / 'model.ply')} voxels")
    report(rows, out)


def show(panel: np.ndarray, args: argparse.Namespace) -> bool:
    if args.no_window:
        return True
    cv2.imshow("cafog", panel)
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)


def report(rows: list[dict], out: Path) -> None:
    n = len(rows)
    col = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    total = col("depth_ms") + col("odometry_ms") + col("engine_ms")
    active = [r for r in rows if r["phase"] in ("ACTIVE", "RELEASING")]
    print(f"\n{n} frames -> {out}")
    print(
        f"  tracking ok {col('tracking').mean():.0%}, floor found {col('floor_found').mean():.0%}, "
        f"camera height median {np.median(col('camera_height_m')):.2f} m, depth scale median "
        f"{np.median(col('depth_scale')):.2f} (IQR {np.percentile(col('depth_scale'), 25):.2f}-"
        f"{np.percentile(col('depth_scale'), 75):.2f})"
    )
    print(
        f"  latency per frame: depth {np.median(col('depth_ms')):.0f} ms, odometry "
        f"{np.median(col('odometry_ms')):.0f} ms, engine {np.median(col('engine_ms')):.0f} ms, "
        f"total {np.median(total):.0f} ms (~{1000 / max(np.median(total), 1):.1f} fps)"
    )
    print(f"  cue active in {len(active)}/{n} frames")
    onsets = [
        r
        for prev, r in zip([None, *rows], rows, strict=False)
        if r["phase"] == "ACTIVE" and (prev is None or prev["phase"] not in ("ACTIVE", "RELEASING"))
    ]
    for r in onsets:
        print(
            f"  cue at t={r['t']:.1f}s: {r['hazard']} {r['hazard_m']} m ahead, "
            f"turn {r['turn_deg']} deg, width {r['width_m']} m"
        )


if __name__ == "__main__":
    main()
