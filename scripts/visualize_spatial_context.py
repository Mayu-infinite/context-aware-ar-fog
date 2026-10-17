
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ANNOTATIONS_DIR = Path("data/annotations")
FRAMES_DIR = Path("data/interim/frames")
OUTPUT_DIR = Path("data/processed/spatial_context")

BASELINE_FILE = ANNOTATIONS_DIR / "spatial_context_baseline.csv"
DETECTIONS_FILE = ANNOTATIONS_DIR / "detection_baseline.csv"
SCENES_FILE = ANNOTATIONS_DIR / "scene_annotations.csv"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the baseline, detections, and scene annotations."""
    for path in (BASELINE_FILE, DETECTIONS_FILE, SCENES_FILE):
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")

    baseline = pd.read_csv(BASELINE_FILE)
    detections = pd.read_csv(DETECTIONS_FILE)
    scenes = pd.read_csv(SCENES_FILE)

    return baseline, detections, scenes


def find_frame(video_id: str, frame_index: int) -> Path:
    """Find the extracted image corresponding to a sampled frame."""
    return FRAMES_DIR / video_id / f"frame_{frame_index:06d}.jpg"


def draw_frame_diagnostic(
    video_id: str,
    frame_index: int,
    detections: pd.DataFrame,
    baseline_row: pd.Series,
    scene_type: str,
    output_path: Path,
) -> None:
    """Draw detected boxes and a simple image-space diagnostic."""
    frame_path = find_frame(video_id, frame_index)

    if not frame_path.exists():
        print(f"WARNING: Missing frame image: {frame_path}")
        return

    image = plt.imread(frame_path)

    fig, (ax_image, ax_regions) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    ax_image.imshow(image)
    ax_image.set_title(f"{video_id}, frame {frame_index}\nScene: {scene_type}")
    ax_image.set_xlim(0, IMAGE_WIDTH)
    ax_image.set_ylim(IMAGE_HEIGHT, 0)
    ax_image.set_xlabel("Image x-coordinate (pixels)")
    ax_image.set_ylabel("Image y-coordinate (pixels)")

    frame_detections = detections[
        (detections["video_id"] == video_id)
        & (detections["frame_index"] == frame_index)
        & (detections["class_name"] != "none")
    ]

    for _, detection in frame_detections.iterrows():
        coords = [
            detection["x1"],
            detection["y1"],
            detection["x2"],
            detection["y2"],
        ]

        if pd.isna(coords).any():
            continue

        x1, y1, x2, y2 = map(float, coords)

        box = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )
        ax_image.add_patch(box)

        label = (
            f"{detection['class_name']} "
            f"({detection['confidence']:.2f})"
        )
        ax_image.text(
            x1,
            max(12, y1 - 5),
            label,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8},
        )

    ax_regions.set_title("Image-space diagnostic")
    ax_regions.set_xlim(0, 3)
    ax_regions.set_ylim(0, 1)
    ax_regions.set_xticks([0.5, 1.5, 2.5])
    ax_regions.set_xticklabels(["Left", "Centre", "Right"])
    ax_regions.set_yticks([])
    ax_regions.set_xlabel("Horizontal image region")

    # A neutral grid, not a map of physical walking space.
    for x in range(3):
        ax_regions.add_patch(
            Rectangle(
                (x, 0),
                1,
                1,
                fill=False,
                linewidth=1,
            )
        )

    positions = str(baseline_row["object_positions"]).split(";")
    positions = [p for p in positions if p and p != "nan"]

    for position in positions:
        region_index = {"left": 0, "centre": 1, "right": 2}.get(position)

        if region_index is not None:
            ax_regions.text(
                region_index + 0.5,
                0.65,
                "Detected\nobject",
                ha="center",
                va="center",
                fontsize=9,
            )

    obstruction = str(baseline_row["potential_obstruction"])
    detection_status = str(baseline_row["detection_status"])

    summary = (
        f"Detected objects: {baseline_row['detected_object_count']}\n"
        f"Flag: {obstruction}\n"
        f"Detection status: {detection_status}\n\n"
        "Traversability: UNKNOWN\n"
        "Depth/geometry: NOT EVALUATED"
    )

    ax_regions.text(
        0.03,
        -0.17,
        summary,
        transform=ax_regions.transAxes,
        va="top",
        fontsize=9,
    )

    fig.suptitle(
        "Spatial context baseline — diagnostic only",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_summary_heatmap(baseline: pd.DataFrame, output_path: Path) -> None:
    """Plot counts of detections in left, centre, and right image regions."""
    region_counts = {"left": 0, "centre": 0, "right": 0}

    for positions in baseline["object_positions"].fillna(""):
        for position in positions.split(";"):
            if position in region_counts:
                region_counts[position] += 1

    labels = list(region_counts)
    values = [region_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values)

    ax.bar_label(bars, padding=3)
    ax.set_title("Detected-object locations across sampled frames")
    ax.set_xlabel("Horizontal image region")
    ax.set_ylabel("Object-location instances")
    ax.text(
        0.02,
        0.97,
        "Image-space counts only; not free-space or safety estimates.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline, detections, scenes = load_data()

    required_columns = {
        "video_id",
        "frame_index",
        "scene_type",
        "object_positions",
        "potential_obstruction",
        "detection_status",
        "detected_object_count",
    }

    missing = required_columns - set(baseline.columns)
    if missing:
        raise SystemExit(
            f"Baseline CSV is missing columns: {sorted(missing)}"
        )

    scenes_by_frame = scenes.copy()
    scenes_by_frame["frame_index"] = (
        scenes_by_frame["frame_file"]
        .str.extract(r"frame_(\d+)", expand=False)
        .astype("Int64")
    )

    generated = 0

    for _, row in baseline.iterrows():
        video_id = str(row["video_id"])
        frame_index = int(row["frame_index"])

        matching_scene = scenes_by_frame[
            (scenes_by_frame["video_id"] == video_id)
            & (scenes_by_frame["frame_index"] == frame_index)
        ]

        scene_type = (
            str(matching_scene.iloc[0]["scene_type"])
            if not matching_scene.empty
            else str(row["scene_type"])
        )

        output_path = OUTPUT_DIR / f"{video_id}_frame_{frame_index:06d}.png"

        draw_frame_diagnostic(
            video_id=video_id,
            frame_index=frame_index,
            detections=detections,
            baseline_row=row,
            scene_type=scene_type,
            output_path=output_path,
        )

        if output_path.exists():
            generated += 1

    summary_path = OUTPUT_DIR / "object_region_summary.png"
    draw_summary_heatmap(baseline, summary_path)

    print(f"Frame diagnostics generated: {generated}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Region summary: {summary_path}")
    print(
        "\nImportant: these plots visualize 2D detections. "
        "They do not establish free space, depth, or safe routes."
    )


if __name__ == "__main__":
    main()
