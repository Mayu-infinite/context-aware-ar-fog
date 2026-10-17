from __future__ import annotations

import argparse
import csv
from pathlib import Path

CSV_PATH = Path("data/annotations/scene_annotations.csv")

# Preliminary suggestions based on the contact sheet.
# These are not verified ground-truth labels.
SCENE_DEFAULTS = {
    "video01": {
        "scene_type": "office",
        "path_status": "uncertain",
        "obstacle_position": "multiple",
        "confidence": "low",
        "notes": "Preliminary: office furniture visible; inspect individual frame.",
    },
    "video02": {
        "scene_type": "corridor",
        "path_status": "uncertain",
        "obstacle_position": "centre",
        "confidence": "low",
        "notes": "Preliminary: corridor with closed door at far end; inspect clearance.",
    },
    "video03": {
        "scene_type": "stairwell",
        "path_status": "uncertain",
        "obstacle_position": "multiple",
        "confidence": "low",
        "notes": "Preliminary: landing/stairs and railing visible; do not infer safe passage.",
    },
    "video04": {
        "scene_type": "entrance",
        "path_status": "uncertain",
        "obstacle_position": "multiple",
        "confidence": "low",
        "notes": "Preliminary: entrance area with signboard, plants and seating.",
    },
    "video05": {
        "scene_type": "doorway",
        "path_status": "uncertain",
        "obstacle_position": "centre",
        "confidence": "low",
        "notes": "Preliminary: doorway changes across frames; inspect opening and clearance.",
    },
    "video06": {
        "scene_type": "office",
        "path_status": "uncertain",
        "obstacle_position": "multiple",
        "confidence": "low",
        "notes": "Preliminary: office desks and chairs with a visible aisle.",
    },
}

FIELDS = [
    "video_id",
    "frame_file",
    "scene_type",
    "path_status",
    "obstacle_position",
    "confidence",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill blank scene annotations with preliminary suggestions."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing annotation values too.",
    )
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"Annotation CSV not found: {CSV_PATH}")

    with CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise SystemExit("CSV has no annotation rows.")

    updated = 0
    unknown = 0

    for row in rows:
        video_id = row.get("video_id", "")
        suggestions = SCENE_DEFAULTS.get(video_id)

        if suggestions is None:
            unknown += 1
            continue

        for field, value in suggestions.items():
            if args.overwrite or not row.get(field, "").strip():
                row[field] = value

        updated += 1

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows updated with preliminary suggestions: {updated}")
    print(f"Rows with unknown video IDs: {unknown}")
    print(f"Saved: {CSV_PATH}")
    print("Review every suggested label before using it for evaluation.")


if __name__ == "__main__":
    main()
