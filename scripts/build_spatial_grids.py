
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cafog.perception.spatial_grid import (
    OBJECT_EVIDENCE,
    add_object_box,
    create_spatial_grid,
    grid_summary,
)

ANNOTATIONS_DIR = Path("data/annotations")
DETECTIONS_FILE = ANNOTATIONS_DIR / "detection_baseline.csv"
OUTPUT_DIR = Path("data/processed/spatial_grids")

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CELL_WIDTH = 80
CELL_HEIGHT = 80


def main() -> None:
    if not DETECTIONS_FILE.exists():
        raise SystemExit(f"Missing input file: {DETECTIONS_FILE}")

    detections = pd.read_csv(DETECTIONS_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cell_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for (video_id, frame_index), group in detections.groupby(
        ["video_id", "frame_index"], sort=True
    ):
        grid = create_spatial_grid(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            CELL_WIDTH,
            CELL_HEIGHT,
        )

        objects = group[group["class_name"] != "none"]

        for _, detection in objects.iterrows():
            coords = [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            ]

            if pd.isna(coords).any():
                continue

            add_object_box(grid, *map(float, coords))

        for row in range(grid.rows):
            for col in range(grid.cols):
                cell_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": int(frame_index),
                        "grid_row": row,
                        "grid_col": col,
                        "cell_value": int(grid.cells[row, col]),
                        "meaning": (
                            "object_evidence"
                            if grid.cells[row, col] == OBJECT_EVIDENCE
                            else "unknown"
                        ),
                    }
                )

        summary = grid_summary(grid)
        summary_rows.append(
            {
                "video_id": video_id,
                "frame_index": int(frame_index),
                **summary,
                "detected_object_count": len(objects),
            }
        )

        figure_path = (
            OUTPUT_DIR / f"{video_id}_frame_{int(frame_index):06d}.png"
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(
            grid.cells,
            interpolation="nearest",
            aspect="auto",
            vmin=-1,
            vmax=1,
            cmap="viridis",
        )

        ax.set_title(
            f"Image-space object evidence: {video_id}, frame {frame_index}"
        )
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
        ax.set_xticks(range(grid.cols))
        ax.set_yticks(range(grid.rows))

        colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 1])
        colorbar.ax.set_yticklabels(["Unknown", "Object evidence"])

        ax.text(
            0.01,
            -0.18,
            "Not a free-space map. Depth and traversability are unknown.",
            transform=ax.transAxes,
            fontsize=9,
        )

        fig.tight_layout()
        fig.savefig(figure_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    cells_file = OUTPUT_DIR / "spatial_grid_cells.csv"
    summary_file = OUTPUT_DIR / "spatial_grid_summary.csv"

    pd.DataFrame(cell_rows).to_csv(cells_file, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    print(f"Frames processed: {len(summary_rows)}")
    print(f"Cell records: {len(cell_rows)}")
    print(f"Grid images: {OUTPUT_DIR}")
    print(f"Cell data: {cells_file}")
    print(f"Frame summary: {summary_file}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\nMean object-evidence fraction:")
        print(
            f"{summary_df['object_evidence_fraction'].mean():.3f}"
        )
        print("\nTotal object-evidence cells across frames:")
        print(int(summary_df["object_evidence_cells"].sum()))


if __name__ == "__main__":
    main()
