"""Record per-frame grid counts and Python engine outputs for the C# parity check.

The C# core (unity/CafogQuest/Assets/Cafog/Runtime/Core) replays the same counts and must
reproduce the cue decisions. Python's world is right-handed; the replay switches the C# core
to that handedness (Float2.Handedness = -1) instead of mirroring the trace, because the grid
is centred on a cell corner and mirroring would flip how equal-cost routes are tie-broken.

    uv run python scripts/export_parity_trace.py
    dotnet run -c Release --project unity/CafogQuest/Tools/CoreParity -- \
        experiments/exp007_csharp_parity/results
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from cafog.sources.synthetic import corridor_turn, doorway, walk
from cafog.spatial import EngineConfig, SpatialCueEngine, make_profile
from cafog.spatial.geometry import depth_to_world_points, ground_heading
from cafog.spatial.occupancy import TraversabilityMap

PHASES = {"IDLE": 0, "ARMING": 1, "ACTIVE": 2, "RELEASING": 3}


def export(scene, out: Path, profile: str = "pd_fog") -> None:
    config = EngineConfig(profile=make_profile(profile))
    engine = SpatialCueEngine(config)
    n = config.map.cells
    frames = list(walk(scene))
    with out.open("wb") as fh:
        fh.write(struct.pack("<ii", n, len(frames)))
        for frame in frames:
            pose = np.asarray(frame.camera_to_world, dtype=np.float64)
            eye = pose[:3, 3]
            user = eye[[0, 2]]
            points = depth_to_world_points(
                frame.depth_m,
                frame.intrinsics,
                pose,
                stride=config.depth_stride,
                max_range_m=config.map.max_range_m,
            )
            if engine.map is None:
                engine.map = TraversabilityMap(config.map, center_xz=tuple(user))
            engine.map.recenter(user)
            counts = engine.map.integrate_points(points, float(eye[1]), frame.timestamp_s)
            gaze = ground_heading(pose)
            result = engine._plan(frame.timestamp_s, user, gaze, points, 0.0, 0.0, 0.0)

            gx, gz = gaze if gaze is not None else (np.nan, np.nan)
            origin = engine.map.origin
            fh.write(
                struct.pack(
                    "<8d",
                    frame.timestamp_s,
                    user[0],
                    user[1],
                    gx,
                    gz,
                    origin[0],
                    origin[1],
                    engine.map.floor_y if engine.map.floor_y is not None else np.nan,
                )
            )
            for grid in (counts.obstacle, counts.floor, counts.dropoff):
                fh.write(np.ascontiguousarray(grid, dtype="<i4").tobytes())

            cue = result.cue
            first = cue.footsteps[0].position if cue.footsteps else (np.nan, np.nan)
            fh.write(
                struct.pack(
                    "<ii5d",
                    PHASES[cue.phase.value],
                    len(cue.footsteps),
                    result.risk.max_within(3.0) if result.risk else np.nan,
                    result.path.length_m if result.path else np.nan,
                    cue.hazard.distance_m if cue.hazard else np.nan,
                    first[0],
                    first[1],
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/exp007_csharp_parity/results")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = [corridor_turn(0), corridor_turn(90), corridor_turn(180), doorway(0.8)]
    for scene in scenes:
        export(scene, args.out / f"{scene.name}.bin")
        print(f"wrote {args.out / scene.name}.bin")


if __name__ == "__main__":
    main()
