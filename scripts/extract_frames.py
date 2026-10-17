from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cafog.utils.video import iter_video_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video."
    )

    parser.add_argument(
        "video",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/interim/frames"
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    for frame_index, frame in iter_video_frames(
        args.video,
        stride=args.stride,
    ):
        output_path = (
            args.output
            / f"frame_{frame_index:06d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        count += 1

    print(
        f"Extracted {count} frames to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()