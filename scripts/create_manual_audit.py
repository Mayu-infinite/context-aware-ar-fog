
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SPATIAL_PATH = ROOT / "data/annotations/spatial_context_baseline.csv"
FEEDBACK_PATH = ROOT / "data/annotations/text_feedback_baseline.csv"
OUTPUT_PATH = ROOT / "data/annotations/manual_frame_audit.csv"


def main() -> None:
    spatial = pd.read_csv(SPATIAL_PATH)
    feedback = pd.read_csv(FEEDBACK_PATH)

    key_columns = ["video_id", "frame_index"]

    for name, frame in [
        ("spatial annotations", spatial),
        ("text feedback", feedback),
    ]:
        missing = set(key_columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

        if frame.duplicated(key_columns).any():
            raise ValueError(
                f"{name} contains duplicate video/frame keys."
            )

    merged = spatial.merge(
        feedback,
        on=key_columns,
        how="outer",
        suffixes=("_spatial", "_feedback"),
        indicator=True,
        validate="one_to_one",
    )

    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        examples = merged.loc[
            unmatched, key_columns + ["_merge"]
        ].to_dict(orient="records")
        raise ValueError(
            f"Spatial and feedback frame keys do not match: {examples[:5]}"
        )

    merged = merged.drop(columns="_merge")

    # Preserve the original predictions and evidence.
    # Human reviewers fill in the audit columns later.
    merged["review_obstacle_visible"] = ""
    merged["review_obstacle_in_walking_region"] = ""
    merged["review_prompt_appropriate"] = ""
    merged["review_missed_potential_obstacle"] = ""
    merged["review_image_quality"] = ""
    merged["review_notes"] = ""

    columns_first = [
        "video_id",
        "frame_index",
        "scene_type_spatial",
        "detection_status",
        "detected_object_count",
        "detected_classes",
        "object_positions",
        "central_lower_object_count",
        "potential_obstruction",
        "path_traversability",
        "feedback",
        "action",
        "reason",
        "evidence_level",
        "review_obstacle_visible",
        "review_obstacle_in_walking_region",
        "review_prompt_appropriate",
        "review_missed_potential_obstacle",
        "review_image_quality",
        "review_notes",
    ]

    # Handle columns that may differ across versions of the baseline.
    columns_first = [
        column for column in columns_first if column in merged.columns
    ]
    remaining = [
        column for column in merged.columns if column not in columns_first
    ]
    audit = merged[columns_first + remaining]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_PATH, index=False)

    print(f"Manual audit CSV created: {OUTPUT_PATH}")
    print(f"Frames available for review: {len(audit)}")
    print()
    print("Suggested review labels:")
    print("- visible / not_visible / uncertain")
    print("- yes / no / uncertain")
    print("- appropriate / inappropriate / uncertain")
    print("- yes / no / uncertain")
    print("- good / blurry / obstructed / other")
    print()
    print("No predictions were changed. Review fields are blank.")


if __name__ == "__main__":
    main()
