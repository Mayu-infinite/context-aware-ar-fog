
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2

from cafog.perception.detector import YOLODetector

VIDEO_DIR = Path("data/raw/videos")
OUTPUT_DIR = Path("data/annotations")
FRAME_STRIDE = 60  # Approximately every 2 seconds at 30 FPS


def main() -> None:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        raise SystemExit(f"No MP4 videos found in {VIDEO_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(
        model_path="yolo26n.pt",
        confidence=0.35,
        image_size=640,
        device="0",
    )

    frame_rows: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    frames_processed = 0

    for video_path in videos:
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            print(f"WARNING: Could not open {video_path}")
            continue

        frame_index = 0
        video_detections = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                if frame_index % FRAME_STRIDE == 0:
                    detections = detector.detect(frame)
                    frames_processed += 1
                    video_detections += len(detections)

                    if not detections:
                        frame_rows.append({
                            "video_id": video_path.stem,
                            "frame_index": frame_index,
                            "class_name": "none",
                            "confidence": "",
                            "x1": "",
                            "y1": "",
                            "x2": "",
                            "y2": "",
                        })

                    for detection in detections:
                        class_counts[detection.class_name] += 1

                        frame_rows.append({
                            "video_id": video_path.stem,
                            "frame_index": frame_index,
                            "class_name": detection.class_name,
                            "confidence": round(
                                detection.confidence, 4
                            ),
                            "x1": round(detection.x1, 1),
                            "y1": round(detection.y1, 1),
                            "x2": round(detection.x2, 1),
                            "y2": round(detection.y2, 1),
                        })

                frame_index += 1

        finally:
            capture.release()

        print(
            f"{video_path.name}: "
            f"{frame_index} source frames, "
            f"{video_detections} detections "
            "across sampled frames"
        )

    columns = [
        "video_id",
        "frame_index",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    detections_csv = OUTPUT_DIR / "detection_baseline.csv"

    with detections_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(frame_rows)

    summary_csv = OUTPUT_DIR / "detection_class_counts.csv"

    with summary_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class_name", "detection_count"])
        writer.writerows(sorted(class_counts.items()))

    print(f"\nSampled frames processed: {frames_processed}")
    print(f"Detection details: {detections_csv}")
    print(f"Class counts: {summary_csv}")
    print("\nDetections by class:")

    for class_name, count in class_counts.most_common():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()

