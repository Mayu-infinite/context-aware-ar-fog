from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(".")
DETECTIONS = ROOT / "data/annotations/detection_baseline.csv"
FRAMES_ROOT = ROOT / "data/interim/frames"
OUT = ROOT / "data/processed/curated_object_frames"

IMAGE_OUT = OUT / "images"
ANNOTATED_OUT = OUT / "annotated"
LABEL_OUT = OUT / "labels_yolo"

for folder in (IMAGE_OUT, ANNOTATED_OUT, LABEL_OUT):
    folder.mkdir(parents=True, exist_ok=True)


def frame_path(video_id: str, frame_index: int) -> Path:
    return FRAMES_ROOT / video_id / f"frame_{frame_index:06d}.jpg"


def main() -> None:
    df = pd.read_csv(DETECTIONS)

    required = {
        "video_id", "frame_index", "class_name",
        "confidence", "x1", "y1", "x2", "y2",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing detection CSV columns: {sorted(missing)}")

    df["class_name"] = df["class_name"].fillna("none").astype(str)
    df["video_id"] = df["video_id"].astype(str)
    df["frame_index"] = df["frame_index"].astype(int)

    # Keep actual detections separate from placeholder "none" rows.
    valid = df[
        (df["class_name"].str.lower() != "none")
        & df[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
    ].copy()

    # Stable class-to-ID mapping for YOLO-format pseudo-labels.
    classes = sorted(valid["class_name"].unique())
    class_to_id = {name: i for i, name in enumerate(classes)}

    (OUT / "classes.txt").write_text(
        "".join(f"{name}\n" for name in classes),
        encoding="utf-8",
    )

    # One row per frame that has at least one detection.
    selected = (
        valid.groupby(["video_id", "frame_index"])
        .size()
        .rename("detection_count")
        .reset_index()
        .sort_values(["video_id", "frame_index"])
    )

    # Keep a review list of sampled frames with no detections.
    no_det = df[df["class_name"].str.lower() == "none"][
        ["video_id", "frame_index"]
    ].drop_duplicates()

    no_det["source_frame"] = no_det.apply(
        lambda r: str(frame_path(r["video_id"], int(r["frame_index"]))),
        axis=1,
    )
    no_det["review_status"] = "unreviewed_no_detection"
    no_det.to_csv(OUT / "no_detection_candidates.csv", index=False)

    annotated_paths = []
    detection_rows = []
    frame_rows = []

    for row in selected.itertuples(index=False):
        video_id = str(row.video_id)
        frame_index = int(row.frame_index)
        src = frame_path(video_id, frame_index)

        if not src.exists():
            print(f"WARNING: missing frame: {src}")
            continue

        image = cv2.imread(str(src))
        if image is None:
            print(f"WARNING: could not read: {src}")
            continue

        height, width = image.shape[:2]
        subset = valid[
            (valid["video_id"] == video_id)
            & (valid["frame_index"] == frame_index)
        ]

        stem = f"{video_id}_frame_{frame_index:06d}"
        image_dst = IMAGE_OUT / f"{stem}.jpg"
        annotated_dst = ANNOTATED_OUT / f"{stem}.jpg"
        label_dst = LABEL_OUT / f"{stem}.txt"

        shutil.copy2(src, image_dst)
        annotated = image.copy()
        yolo_lines = []

        for det in subset.itertuples(index=False):
            x1 = max(0, min(width - 1, float(det.x1)))
            y1 = max(0, min(height - 1, float(det.y1)))
            x2 = max(0, min(width - 1, float(det.x2)))
            y2 = max(0, min(height - 1, float(det.y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = class_to_id[str(det.class_name)]
            confidence = float(det.confidence)

            # YOLO format: class_id, x_center, y_center, box_width, box_height
            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            yolo_lines.append(
                f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            label = f"{det.class_name} {confidence:.2f}"
            text_y = max(22, int(y1) - 7)
            cv2.putText(
                annotated,
                label,
                (int(x1), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            detection_rows.append({
                "video_id": video_id,
                "frame_index": frame_index,
                "class_name": str(det.class_name),
                "class_id": class_id,
                "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "image_width": width,
                "image_height": height,
                "label_source": "yolo_pseudo_label_unverified",
            })

        label_dst.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )

        # Explicit frame identifier on the image, even if a label is hard to see.
        cv2.putText(
            annotated,
            f"{video_id} frame {frame_index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if not cv2.imwrite(str(annotated_dst), annotated):
            raise RuntimeError(f"Could not save {annotated_dst}")

        annotated_paths.append(annotated_dst)
        frame_rows.append({
            "video_id": video_id,
            "frame_index": frame_index,
            "source_frame": str(src),
            "image_path": str(image_dst),
            "annotated_path": str(annotated_dst),
            "yolo_label_path": str(label_dst),
            "detection_count": len(yolo_lines),
            "review_status": "unreviewed_pseudo_labels",
        })

    pd.DataFrame(detection_rows).to_csv(
        OUT / "object_detections.csv", index=False
    )
    pd.DataFrame(frame_rows).to_csv(
        OUT / "selected_frames.csv", index=False
    )

    # Contact sheet of all annotated selected frames.
    thumbs = []
    thumb_width = 360

    for path in annotated_paths:
        im = cv2.imread(str(path))
        if im is None:
            continue

        scale = thumb_width / im.shape[1]
        im = cv2.resize(
            im,
            (thumb_width, int(im.shape[0] * scale)),
        )
        thumbs.append((path.name, im))

    if thumbs:
        cell_h = max(im.shape[0] for _, im in thumbs) + 32
        columns = 2
        rows = (len(thumbs) + columns - 1) // columns
        sheet = 255 * __import__("numpy").ones(
            (rows * cell_h, columns * thumb_width, 3),
            dtype="uint8",
        )

        for i, (name, im) in enumerate(thumbs):
            r, c = divmod(i, columns)
            x, y = c * thumb_width, r * cell_h
            cv2.putText(
                sheet, name, (x + 5, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                cv2.LINE_AA,
            )
            sheet[y + 32:y + 32 + im.shape[0], x:x + thumb_width] = im

        cv2.imwrite(str(OUT / "contact_sheet.jpg"), sheet)

    print(f"\nOutput directory: {OUT}")
    print(f"Selected frames: {len(frame_rows)}")
    print(f"Detection labels: {len(detection_rows)}")
    print(f"Classes: {class_to_id}")
    print(f"No-detection frame candidates: {len(no_det)}")
    print("\nReview the contact sheet and verify pseudo-labels manually.")


if __name__ == "__main__":
    main()
