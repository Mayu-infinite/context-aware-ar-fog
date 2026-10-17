from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from cafog.context.rules import ContextEngine
from cafog.perception.detector import YOLODetector
from cafog.policy.cue_policy import CuePolicy
from cafog.realtime.pipeline import RealtimePipeline
from cafog.utils.visualization import (
    draw_detections,
    draw_pipeline_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run scene perception and context "
            "reasoning on a video."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/"
            "exp001_baseline_detection/"
            "results/"
            "annotated.mp4"
        ),
    )

    parser.add_argument(
        "--model",
        default="yolo26n.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = YOLODetector(
        model_path=args.model,
        confidence=args.confidence,
    )

    context_engine = ContextEngine()

    cue_policy = CuePolicy()

    pipeline = RealtimePipeline(
        detector=detector,
        context_engine=context_engine,
        cue_policy=cue_policy,
    )

    capture = cv2.VideoCapture(
        str(args.video)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open {args.video}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    processed = 0
    total_inference_time = 0.0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % args.stride != 0:
                frame_index += 1
                continue

            start = time.perf_counter()

            result = pipeline.process(frame)

            elapsed = (
                time.perf_counter() - start
            )

            total_inference_time += elapsed
            processed += 1

            annotated = draw_detections(
                frame,
                result.detections,
            )

            annotated = draw_pipeline_status(
                annotated,
                result.scene_state,
                result.cue,
            )

            writer.write(annotated)

            frame_index += 1

    finally:
        capture.release()
        writer.release()

    if processed > 0:
        average_ms = (
            total_inference_time
            / processed
            * 1000
        )

        effective_fps = (
            processed
            / total_inference_time
        )

        print(
            f"Processed frames: {processed}"
        )

        print(
            f"Average inference time: "
            f"{average_ms:.2f} ms"
        )

        print(
            f"Effective processing FPS: "
            f"{effective_fps:.2f}"
        )

    print(
        f"Saved annotated video to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()