"""Headset-free evaluation of the spatial cue engine on procedural scenes.

Reproduces the turning conditions of Bhattacharya et al. (2026) (0, 60, 90, 120, 180 deg)
plus doorways of decreasing width, for healthy / PD non-FOG / PD-FOG profiles.

    uv run python scripts/run_simulation.py --videos
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from cafog.sources.synthetic import (
    PAPER_TURN_ANGLES,
    cluttered_room,
    corridor_turn,
    doorway,
    run_scene,
    summarize_run,
    walk,
)
from cafog.spatial import EngineConfig, SpatialCueEngine, make_profile
from cafog.spatial.gen import assess_path
from cafog.spatial.viz import bev_image, compose

PROFILES = ("healthy", "pd_nonfog", "pd_fog")
DOOR_WIDTHS = (0.7, 0.8, 0.9, 1.0, 1.2)


def build_config(profile: str, config_path: Path | None) -> EngineConfig:
    config = EngineConfig.from_yaml(config_path) if config_path else EngineConfig()
    config.profile = make_profile(profile)
    return config


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def value_figure(out: Path, config_path: Path | None) -> None:
    """Value maps 2.4 m before each turn: healthy critic vs. the PD-FOG value canyon."""
    rows = []
    for angle in PAPER_TURN_ANGLES:
        engine = SpatialCueEngine(build_config("pd_fog", config_path))
        for frame in walk(corridor_turn(angle)):
            result = engine.process(frame)
            if result.user_xz[1] >= 3.6:
                break
        tiles = [bev_image(result, engine.map, which=w, px_per_cell=4) for w in ("hc", "pd")]
        for tile, label in zip(tiles, ("healthy V", "PD-FOG V"), strict=True):
            cv2.putText(
                tile,
                f"{angle} deg  {label}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out), np.vstack(rows))


def canyon_figure(out: Path, config_path: Path | None) -> None:
    """Value and predicted speed along the planned route, as in the paper's value-canyon plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(PAPER_TURN_ANGLES), figsize=(3.2 * len(PAPER_TURN_ANGLES), 5.5))
    for col, angle in enumerate(PAPER_TURN_ANGLES):
        config = build_config("pd_fog", config_path)
        engine = SpatialCueEngine(config)
        for frame in walk(corridor_turn(angle)):
            result = engine.process(frame)
            if result.user_xz[1] >= 4.0:
                break
        if result.path is None:
            continue
        for profile, color in zip(PROFILES, ("tab:green", "tab:orange", "tab:red"), strict=True):
            risk = assess_path(result.path, make_profile(profile), config.value)
            ax_v, ax_s = axes[0, col], axes[1, col]
            if profile == "healthy":
                ax_v.plot(risk.arc_m, risk.value_hc, color=color, label="healthy")
            else:
                ax_v.plot(risk.arc_m, risk.value_pd, color=color, label=profile)
            ax_s.plot(risk.arc_m, risk.speed_pd, color=color, label=profile)
        axes[0, col].set_title(f"{angle} deg turn")
        axes[1, col].set_xlabel("distance ahead along route (m)")
    axes[0, 0].set_ylabel("value V")
    axes[1, 0].set_ylabel("GEN speed (a.u.)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/exp005_spatial_value/results")
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--videos", action="store_true", help="write per-scene PD-FOG videos")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = [corridor_turn(a) for a in PAPER_TURN_ANGLES]
    scenes += [doorway(w) for w in DOOR_WIDTHS]
    scenes.append(cluttered_room())

    summary_path = args.out / "simulation_summary.csv"
    rows = []
    for scene in scenes:
        for profile in PROFILES:
            frames: list[np.ndarray] = []

            def capture(frame, result, engine, frames=frames):
                frames.append(compose(frame, result, engine.map))

            record = args.videos and profile == "pd_fog"
            records = run_scene(
                scene, build_config(profile, args.config), capture if record else None, fps=args.fps
            )
            row = {"scene": scene.name, "profile": profile, **summarize_run(records)}
            rows.append(row)
            print(
                f"{scene.name:15s} {profile:10s} max_risk={row['max_risk']:.2f} "
                f"first_cue={row['first_cue_distance_m']} active={row['active_frames']}"
                f"/{row['frames']} latency={row['latency_mean_ms']:.1f} ms",
                flush=True,
            )
            if frames:
                write_video(args.out / f"{scene.name}_{profile}.mp4", frames, args.fps)

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    value_figure(args.out / "value_maps_hc_vs_pdfog.png", args.config)
    canyon_figure(args.out / "value_canyon_along_route.png", args.config)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
