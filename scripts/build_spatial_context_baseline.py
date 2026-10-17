
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ANNOTATIONS_DIR = Path("data/annotations")
DETECTIONS_FILE = ANNOTATIONS_DIR / "detection_baseline.csv"
SCENES_FILE = ANNOTATIONS_DIR / "scene_annotations.csv"
OUTPUT_FILE = ANNOTATIONS_DIR / "spatial_context_baseline.csv"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def frame_index_from_path(frame_file: str) -> int | None:
    """Extract the sampled source-frame index from a frame filename."""
    match = re.search(r"frame_(\d+)", Path(frame_file).stem)
    return int(match.group(1)) if match else None


def classify_object_position(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> str:
    """Estimate an object's image region from its bounding-box centre."""
    center_x = ((x1 + x2) / 2) / IMAGE_WIDTH

    if center_x < 1 / 3:
        return "left"
    if center_x > 2 / 3:
        return "right"
    return "centre"


def main() -> None:
    if not DETECTIONS_FILE.exists():
        raise SystemExit(f"Missing input file: {DETECTIONS_FILE}")

    if not SCENES_FILE.exists():
        raise SystemExit(f"Missing input file: {SCENES_FILE}")

    detections = pd.read_csv(DETECTIONS_FILE)
    scenes = pd.read_csv(SCENES_FILE)

    scenes["frame_index"] = scenes["frame_file"].map(
        frame_index_from_path
    )

    if scenes["frame_index"].isna().any():
        print("WARNING: Some scene annotation filenames could not be parsed.")

    rows: list[dict[str, object]] = []

    for (video_id, frame_index), group in detections.groupby(
        ["video_id", "frame_index"], sort=True
    ):
        scene_match = scenes[
            (scenes["video_id"] == video_id)
            & (scenes["frame_index"] == frame_index)
        ]

        if scene_match.empty:
            scene_type = "unknown"
            annotation_path_status = "unknown"
            annotation_confidence = "unknown"
        else:
            scene = scene_match.iloc[0]
            scene_type = scene["scene_type"]
            annotation_path_status = scene["path_status"]
            annotation_confidence = scene["confidence"]

        objects = group[group["class_name"] != "none"].copy()

        positions: list[str] = []
        classes: list[str] = []
        central_lower_objects = 0

        for _, obj in objects.iterrows():
            coordinates = [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]

            if pd.isna(coordinates).any():
                continue

            x1, y1, x2, y2 = map(float, coordinates)

            positions.append(
                classify_object_position(x1, y1, x2, y2)
            )
            classes.append(str(obj["class_name"]))

            center_x = ((x1 + x2) / 2) / IMAGE_WIDTH
            bottom_y = y2 / IMAGE_HEIGHT

            # A visual warning flag only; this does not establish
            # that the object physically blocks the walking path.
            if 0.35 <= center_x <= 0.65 and bottom_y >= 0.55:
                central_lower_objects += 1

        if objects.empty:
            detection_status = "no_objects_detected"
            potential_obstruction = "unknown"
            interpretation = (
                "No objects detected at the configured threshold; "
                "free space cannot be inferred."
            )
        else:
            detection_status = "objects_detected"

            if central_lower_objects:
                potential_obstruction = "possible_central_view_obstacle"
                interpretation = (
                    "One or more detected boxes occupy the central, "
                    "lower image region; verify against the original frame "
                    "and depth before making a guidance decision."
                )
            else:
                potential_obstruction = "not_flagged_by_simple_heuristic"
                interpretation = (
                    "The simple central-region heuristic was not triggered; "
                    "this does not mean the route is clear."
                )

        rows.append(
            {
                "video_id": video_id,
                "frame_index": int(frame_index),
                "scene_type": scene_type,
                "annotation_path_status": annotation_path_status,
                "annotation_confidence": annotation_confidence,
                "detection_status": detection_status,
                "detected_object_count": len(objects),
                "detected_classes": ";".join(sorted(set(classes))),
                "object_positions": ";".join(sorted(set(positions))),
                "central_lower_object_count": central_lower_objects,
                "potential_obstruction": potential_obstruction,
                "path_traversability": "unknown_requires_geometry_and_depth",
                "interpretation": interpretation,
            }
        )

    result = pd.DataFrame(rows)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"Sampled frames summarized: {len(result)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nFrames by scene type:")
    print(result["scene_type"].value_counts().to_string())
    print("\nDetection status:")
    print(result["detection_status"].value_counts().to_string())
    print("\nPotential obstruction flags:")
    print(result["potential_obstruction"].value_counts().to_string())
    print("\nFirst 10 rows:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
