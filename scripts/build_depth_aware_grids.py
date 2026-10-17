from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cafog.perception.depth_grid import summarize_depth_grid
from cafog.perception.spatial_grid import (
    OBJECT_EVIDENCE,
    add_object_box,
    create_spatial_grid,
)

ANNOTATIONS = Path("data/annotations/detection_baseline.csv")
DEPTH_DIR = Path("data/processed/depth_demo")
FRAME_DIR = Path("data/interim/frames")
OUTPUT_DIR = Path("data/processed/depth_aware_grids")

CELL_WIDTH = 80
CELL_HEIGHT = 80


def main() -> None:
    if not ANNOTATIONS.exists():
        raise SystemExit(f"Missing detections file: {ANNOTATIONS}")

    detections = pd.read_csv(ANNOTATIONS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    cell_records: list[dict[str, object]] = []

    depth_files = sorted(DEPTH_DIR.glob("*_relative_depth.npy"))
    if not depth_files:
        raise SystemExit(f"No depth arrays found in {DEPTH_DIR}")

    for depth_path in depth_files:
        # Expected: video01_frame_000000_relative_depth.npy
        prefix = "_relative_depth.npy"
        stem = depth_path.name.removesuffix(prefix)
        if "_frame_" not in stem:
            print(f"Skipping unlabelled depth file: {depth_path.name}")
            continue

        video_id, frame_name = stem.split("_frame_", maxsplit=1)
        frame_index = int(frame_name)
        frame_path = FRAME_DIR / video_id / f"frame_{frame_index:06d}.jpg"

        if not frame_path.exists():
            print(f"Skipping missing source frame: {frame_path}")
            continue

        image = Image.open(frame_path).convert("RGB")
        width, height = image.size
        depth = np.load(depth_path)

        if depth.shape != (height, width):
            raise ValueError(
                f"Shape mismatch for {depth_path.name}: "
                f"depth={depth.shape}, image={(height, width)}"
            )

        depth_grid = summarize_depth_grid(
            depth, cell_width=CELL_WIDTH, cell_height=CELL_HEIGHT
        )
        object_grid = create_spatial_grid(
            width, height, CELL_WIDTH, CELL_HEIGHT
        )

        group = detections[
            (detections["video_id"] == video_id)
            & (detections["frame_index"] == frame_index)
            & (detections["class_name"] != "none")
        ]

        for _, detection in group.iterrows():
            coords = [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            ]
            if pd.isna(coords).any():
                continue
            add_object_box(object_grid, *map(float, coords))

        for row in range(depth_grid.rows):
            for col in range(depth_grid.cols):
                cell_records.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "grid_row": row,
                        "grid_col": col,
                        "object_evidence": bool(
                            object_grid.cells[row, col] == OBJECT_EVIDENCE
                        ),
                        "relative_depth_median": float(
                            depth_grid.median[row, col]
                        ),
                        "relative_depth_mad": float(
                            depth_grid.spread[row, col]
                        ),
                        "depth_valid_fraction": float(
                            depth_grid.valid_fraction[row, col]
                        ),
                        "interpretation": (
                            "object_and_relative_depth_evidence"
                            if object_grid.cells[row, col] == OBJECT_EVIDENCE
                            else "relative_depth_only_not_traversability"
                        ),
                    }
                )

        finite = np.isfinite(depth_grid.median)
        summaries.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "depth_file": depth_path.name,
                "detected_object_count": len(group),
                "grid_rows": depth_grid.rows,
                "grid_cols": depth_grid.cols,
                "cells_with_depth": int(finite.sum()),
                "cells_with_object_evidence": int(
                    np.count_nonzero(object_grid.cells == OBJECT_EVIDENCE)
                ),
                "mean_depth_valid_fraction": float(
                    depth_grid.valid_fraction.mean()
                ),
                "interpretation": (
                    "Relative-depth and object evidence only; "
                    "not calibrated distance or traversability."
                ),
            }
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(image)
        axes[0].set_title("Source frame")

        axes[1].imshow(
            depth_grid.median,
            interpolation="nearest",
            cmap="magma",
        )
        axes[1].set_title("Cell median: relative depth")

        overlay = np.zeros(
            (depth_grid.rows, depth_grid.cols, 3), dtype=np.float32
        )
        overlay[:] = [0.35, 0.35, 0.35]
        overlay[object_grid.cells == OBJECT_EVIDENCE] = [1.0, 0.15, 0.1]
        axes[2].imshow(overlay, interpolation="nearest")
        axes[2].set_title("Red = detected object box")

        for ax in axes:
            ax.set_xlabel("Grid column")
            ax.set_ylabel("Grid row")

        fig.suptitle(
            f"{video_id}, frame {frame_index}: "
            "relative depth is not metres or proof of free space"
        )
        fig.tight_layout()
        output_path = OUTPUT_DIR / f"{video_id}_frame_{frame_index:06d}.png"
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")

    if not summaries:
        raise SystemExit("No matching frame/depth pairs were processed.")

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "depth_grid_summary.csv", index=False
    )
    pd.DataFrame(cell_records).to_csv(
        OUTPUT_DIR / "depth_grid_cells.csv", index=False
    )

    print(f"\nProcessed matching frames: {len(summaries)}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Depth values remain relative; no walkability labels are inferred.")


if __name__ == "__main__":
    main()
