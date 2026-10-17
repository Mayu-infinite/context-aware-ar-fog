
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from textwrap import wrap

import cv2
import numpy as np

from cafog.perception.detector import Detection, YOLODetector

# These classes are candidates for obstacle warnings, not proof of a hazard.
OBSTACLE_LIKE_CLASSES = {
    "chair",
    "couch",
    "bench",
    "bicycle",
    "motorcycle",
    "car",
    "suitcase",
    "dining table",
    "potted plant",
    "refrigerator",
}

GRID_ROWS = 4
GRID_COLS = 5

# Initial engineering heuristics. Tune on labelled data; these are not
# clinically validated thresholds.
MIN_OBSTACLE_CONFIDENCE = 0.45
CENTRAL_X_MIN = 0.30
CENTRAL_X_MAX = 0.70
LOWER_IMAGE_MIN = 0.50
MIN_BOX_AREA_RATIO = 0.008

HISTORY_LENGTH = 5
ACTIVATE_POSITIVES = 3
CLEAR_MAX_POSITIVES = 1
MAX_RESULT_AGE_SECONDS = 2.0


def box_area_ratio(det: Detection, width: int, height: int) -> float:
    box_width = max(0.0, min(float(width), det.x2) - max(0.0, det.x1))
    box_height = max(0.0, min(float(height), det.y2) - max(0.0, det.y1))
    return (box_width * box_height) / max(width * height, 1)


def is_possible_obstacle(
    detections: list[Detection],
    width: int,
    height: int,
) -> bool:
    """Find plausible obstacle-like detections in the lower central view.

    This is an image-space heuristic. It does not estimate real distance,
    floor contact, collision risk, or whether a route is traversable.
    """
    for det in detections:
        if det.class_name not in OBSTACLE_LIKE_CLASSES:
            continue

        if det.confidence < MIN_OBSTACLE_CONFIDENCE:
            continue

        if box_area_ratio(det, width, height) < MIN_BOX_AREA_RATIO:
            continue

        center_x = det.center_x / max(width, 1)
        bottom_y = det.y2 / max(height, 1)

        if (
            CENTRAL_X_MIN <= center_x <= CENTRAL_X_MAX
            and bottom_y >= LOWER_IMAGE_MIN
        ):
            return True

    return False


class FeedbackController:
    """Quiet-by-default warning policy with temporal hysteresis."""

    def __init__(self) -> None:
        self.history: deque[bool] = deque(maxlen=HISTORY_LENGTH)
        self.alert_active = False
        self.last_result_time: float | None = None

    def update(self, possible_obstacle: bool, now: float) -> bool:
        self.history.append(possible_obstacle)
        self.last_result_time = now

        positive_count = sum(self.history)

        if not self.alert_active:
            if (
                len(self.history) == HISTORY_LENGTH
                and positive_count >= ACTIVATE_POSITIVES
            ):
                self.alert_active = True
        else:
            # Require sustained negative evidence before clearing.
            if (
                len(self.history) == HISTORY_LENGTH
                and positive_count <= CLEAR_MAX_POSITIVES
            ):
                self.alert_active = False

        return self.alert_active

    def expire_if_stale(self, now: float) -> bool:
        """Hide an old warning when fresh detector results stop arriving."""
        if (
            self.last_result_time is None
            or now - self.last_result_time > MAX_RESULT_AGE_SECONDS
        ):
            self.alert_active = False
            self.history.clear()

        return self.alert_active


def draw_heatmap(
    frame: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Overlay detected-object evidence, never a safe-path heatmap."""
    height, width = frame.shape[:2]
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    for det in detections:
        x1 = int(np.clip(det.x1, 0, width - 1))
        y1 = int(np.clip(det.y1, 0, height - 1))
        x2 = int(np.clip(det.x2, 0, width - 1))
        y2 = int(np.clip(det.y2, 0, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        c1 = min(GRID_COLS - 1, x1 * GRID_COLS // width)
        c2 = min(GRID_COLS - 1, x2 * GRID_COLS // width)
        r1 = min(GRID_ROWS - 1, y1 * GRID_ROWS // height)
        r2 = min(GRID_ROWS - 1, y2 * GRID_ROWS // height)

        for row in range(r1, r2 + 1):
            for col in range(c1, c2 + 1):
                grid[row, col] += det.confidence

    if grid.max() > 0:
        grid /= grid.max()

    cell_w = width // GRID_COLS
    cell_h = height // GRID_ROWS
    overlay = frame.copy()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = width if col == GRID_COLS - 1 else (col + 1) * cell_w
            y2 = height if row == GRID_ROWS - 1 else (row + 1) * cell_h

            strength = float(grid[row, col])

            # Redder cells indicate more object-detection evidence.
            colour = (
                0,
                int(180 * (1.0 - strength)),
                int(255 * strength),
            )

            if strength > 0:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)

            cv2.rectangle(
                overlay, (x1, y1), (x2, y2), (220, 220, 220), 1
            )

    return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)


def draw_wrapped_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    max_chars: int = 58,
) -> None:
    for line in wrap(text, width=max_chars):
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24


def scale_detections(
    detections: list[Detection],
    scale_x: float,
    scale_y: float,
) -> list[Detection]:
    """Map inference-image boxes back to the displayed frame coordinates."""
    return [
        Detection(
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            x1=det.x1 * scale_x,
            y1=det.y1 * scale_y,
            x2=det.x2 * scale_x,
            y2=det.y2 * scale_y,
        )
        for det in detections
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAFoG live object-evidence and feedback research demo"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Phone camera stream URL")
    source.add_argument("--video", help="Path to a saved video")

    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument(
        "--detect-every",
        type=int,
        default=2,
        help="Run detection every N received frames",
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=640,
        help="Resize inference frames to this width; 0 disables resizing",
    )
    args = parser.parse_args()

    if args.detect_every < 1:
        parser.error("--detect-every must be at least 1")
    if args.inference_width < 0:
        parser.error("--inference-width cannot be negative")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")

    source_path = args.url if args.url else args.video
    cap = cv2.VideoCapture(source_path)

    # Reduce buffering where supported by the capture backend.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise SystemExit(
            "Could not open source. Check the phone URL or video path."
        )

    detector = YOLODetector(
        confidence=args.confidence,
        device=args.device,
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 15.0

    writer = None
    controller = FeedbackController()
    last_detections: list[Detection] = []

    frame_count = 0
    detection_count = 0
    smoothed_display_fps = 0.0
    previous_frame_time = time.perf_counter()
    inference_ms = 0.0

    print("CAFoG research demo started. Press Q or Esc to quit.")
    print("Quiet display means no active warning, NOT a verified clear path.")
    print("Heatmap shows detected-object evidence, not walkability.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame unavailable; stopping.")
                break

            frame_count += 1
            height, width = frame.shape[:2]

            if writer is None and args.save:
                output = Path(args.save)
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise SystemExit(f"Could not create output video: {output}")

            run_detection = (frame_count - 1) % args.detect_every == 0

            if run_detection:
                if args.inference_width > 0 and width > args.inference_width:
                    inference_height = max(
                        1, round(height * args.inference_width / width)
                    )
                    inference_frame = cv2.resize(
                        frame,
                        (args.inference_width, inference_height),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    inference_frame = frame
                    inference_height, _ = frame.shape[:2]

                inference_height, inference_width = inference_frame.shape[:2]
                start_inference = time.perf_counter()
                small_detections = detector.detect(inference_frame)
                inference_ms = (
                    time.perf_counter() - start_inference
                ) * 1000.0

                last_detections = scale_detections(
                    small_detections,
                    width / inference_width,
                    height / inference_height,
                )

                possible_obstacle = is_possible_obstacle(
                    last_detections, width, height
                )
                controller.update(
                    possible_obstacle, time.perf_counter()
                )
                detection_count += 1

            now = time.perf_counter()
            alert_active = controller.expire_if_stale(now)

            display = draw_heatmap(frame.copy(), last_detections)

            for det in last_detections:
                x1 = int(np.clip(det.x1, 0, width - 1))
                y1 = int(np.clip(det.y1, 0, height - 1))
                x2 = int(np.clip(det.x2, 0, width - 1))
                y2 = int(np.clip(det.y2, 0, height - 1))

                cv2.rectangle(
                    display, (x1, y1), (x2, y2), (0, 220, 0), 2
                )
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(
                    display,
                    label,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

            instant_fps = 1.0 / max(now - previous_frame_time, 1e-6)
            smoothed_display_fps = (
                instant_fps
                if smoothed_display_fps == 0
                else 0.9 * smoothed_display_fps + 0.1 * instant_fps
            )
            previous_frame_time = now

            cv2.rectangle(display, (0, 0), (width, 36), (25, 25, 25), -1)
            cv2.putText(
                display,
                "RESEARCH DEMO | Heatmap is not walkability",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Quiet by default: show no feedback panel unless an alert is active.
            if alert_active:
                panel_height = min(100, max(70, height // 4))
                cv2.rectangle(
                    display,
                    (0, height - panel_height),
                    (width, height),
                    (25, 25, 25),
                    -1,
                )
                cv2.putText(
                    display,
                    "VERIFY POSSIBLE OBSTACLE",
                    (10, height - panel_height + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 210, 255),
                    2,
                    cv2.LINE_AA,
                )
                draw_wrapped_text(
                    display,
                    "Possible obstacle in the central view. "
                    "Location and clearance are unverified.",
                    10,
                    height - panel_height + 49,
                    max_chars=max(30, width // 11),
                )

            status = "ALERT" if alert_active else "QUIET"
            stats = (
                f"{status} | objects: {len(last_detections)} | "
                f"inference: {inference_ms:.0f} ms | "
                f"display: {smoothed_display_fps:.1f} FPS"
            )
            cv2.putText(
                display,
                stats,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(display)

            cv2.imshow("CAFoG Live Context Feedback", display)

            if frame_count % 30 == 0:
                print(
                    f"Frame {frame_count}: {status}; "
                    f"objects={len(last_detections)}; "
                    f"inference={inference_ms:.0f} ms"
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Frames received: {frame_count}")
    print(f"Detection runs: {detection_count}")
    print("Finished. A quiet screen does not certify a safe path.")


if __name__ == "__main__":
    main()
