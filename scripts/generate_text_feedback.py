
from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/annotations/spatial_context_baseline.csv")
OUTPUT_FILE = Path("data/annotations/text_feedback_baseline.csv")


def generate_feedback(row: pd.Series) -> dict[str, str]:
    """Generate a conservative, research-only text feedback candidate."""

    scene = str(row["scene_type"]).strip().lower()
    obstruction = str(row["potential_obstruction"]).strip().lower()
    detection_status = str(row["detection_status"]).strip().lower()

    if detection_status == "no_objects_detected":
        return {
            "feedback": (
                "Unable to assess the path from current detections. "
                "Obtain more scene information."
            ),
            "action": "NO_AUTOMATIC_ACTION",
            "reason": "No objects detected; free space remains unknown.",
            "evidence_level": "insufficient",
        }

    if obstruction == "possible_central_view_obstacle":
        return {
            "feedback": (
                "A possible obstacle appears near the centre of the view. "
                "Verify its location and the available clearance."
            ),
            "action": "VERIFY_POTENTIAL_OBSTACLE",
            "reason": (
                "A detected bounding box overlaps the central, lower "
                "image region; physical obstruction is unconfirmed."
            ),
            "evidence_level": "limited",
        }

    if scene == "stairwell":
        return {
            "feedback": (
                "Stairs or a landing may be present. "
                "The current system cannot assess safe passage."
            ),
            "action": "NO_AUTOMATIC_ACTION",
            "reason": "Stair geometry and traversability are not estimated.",
            "evidence_level": "insufficient",
        }

    if scene == "doorway":
        return {
            "feedback": (
                "A doorway may be present. Opening width and clearance "
                "have not been measured."
            ),
            "action": "NO_AUTOMATIC_ACTION",
            "reason": "Doorway passage is not verified geometrically.",
            "evidence_level": "insufficient",
        }

    if scene == "corridor":
        return {
            "feedback": (
                "A corridor may be present, but the walkable route "
                "has not been established."
            ),
            "action": "NO_AUTOMATIC_ACTION",
            "reason": "No validated free-space or route estimate is available.",
            "evidence_level": "insufficient",
        }

    return {
        "feedback": (
            "Scene detected, but there is insufficient evidence "
            "to recommend a direction."
        ),
        "action": "NO_AUTOMATIC_ACTION",
        "reason": "Scene geometry and route are not validated.",
        "evidence_level": "insufficient",
    }


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing input file: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)

    required_columns = {
        "video_id",
        "frame_index",
        "scene_type",
        "detection_status",
        "potential_obstruction",
        "path_traversability",
    }

    missing = required_columns - set(data.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {sorted(missing)}")

    feedback_rows: list[dict[str, object]] = []

    for _, row in data.iterrows():
        result = generate_feedback(row)

        feedback_rows.append(
            {
                "video_id": row["video_id"],
                "frame_index": int(row["frame_index"]),
                "scene_type": row["scene_type"],
                "feedback": result["feedback"],
                "action": result["action"],
                "reason": result["reason"],
                "evidence_level": result["evidence_level"],
                "research_only": True,
            }
        )

    result_df = pd.DataFrame(feedback_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Frames processed: {len(result_df)}")
    print(f"Feedback CSV: {OUTPUT_FILE}")
    print("\nProposed feedback by action:")
    print(result_df["action"].value_counts().to_string())
    print("\nExample feedback messages:")
    print(
        result_df[
            ["video_id", "frame_index", "feedback", "action"]
        ].head(10).to_string(index=False)
    )

    print(
        "\nSAFETY: These are research prototypes, not validated "
        "instructions for patients."
    )


if __name__ == "__main__":
    main()
