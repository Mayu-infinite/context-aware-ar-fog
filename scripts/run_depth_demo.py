
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_NAME = "Intel/dpt-hybrid-midas"

INPUT_FRAMES = [
    Path("data/interim/frames/video01/frame_000000.jpg"),
    Path("data/interim/frames/video02/frame_000000.jpg"),
    Path("data/interim/frames/video05/frame_000000.jpg"),
]

OUTPUT_DIR = Path("data/processed/depth_demo")


def main() -> None:
    missing = [path for path in INPUT_FRAMES if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input frames: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    for frame_path in INPUT_FRAMES:
        print(f"Processing {frame_path}")

        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            prediction = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )

        depth = prediction.squeeze().float().cpu().numpy()

        if not np.isfinite(depth).all():
            raise RuntimeError(f"Non-finite depth values for {frame_path}")

        stem = f"{frame_path.parent.name}_{frame_path.stem}"
        np.save(OUTPUT_DIR / f"{stem}_relative_depth.npy", depth)

        # Normalize only for visualization; this does not produce metres.
        low, high = np.percentile(depth, [2, 98])
        if high <= low:
            high = low + 1e-6

        normalized = np.clip((depth - low) / (high - low), 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original frame")
        axes[0].axis("off")

        heatmap = axes[1].imshow(normalized, cmap="magma")
        axes[1].set_title("Relative depth visualization")
        axes[1].axis("off")
        fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"{frame_path.parent.name} — relative depth, not metres"
        )
        fig.tight_layout()

        output_image = OUTPUT_DIR / f"{stem}_depth_visualization.png"
        fig.savefig(output_image, dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_image}")

    print("\nDepth demo complete.")
    print(f"Results: {OUTPUT_DIR}")
    print("These predictions are relative depth, not calibrated distances.")


if __name__ == "__main__":
    main()

