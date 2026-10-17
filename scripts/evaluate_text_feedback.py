
from __future__ import annotations

from pathlib import Path

import pandas as pd

ANNOTATIONS_DIR = Path("data/annotations")
SPATIAL_FILE = ANNOTATIONS_DIR / "spatial_context_baseline.csv"
FEEDBACK_FILE = ANNOTATIONS_DIR / "text_feedback_baseline.csv"
OUTPUT_FILE = Path("data/processed/text_feedback_evaluation.md")

KEY_COLUMNS = ["video_id", "frame_index"]


def markdown_table(frame: pd.DataFrame) -> str:
    """Convert a DataFrame to Markdown without additional dependencies."""
    headers = [str(column) for column in frame.columns]

    def escape(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in frame.itertuples(index=False, name=None):
        values = [escape(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main() -> None:
    for file_path in (SPATIAL_FILE, FEEDBACK_FILE):
        if not file_path.exists():
            raise SystemExit(f"Missing input file: {file_path}")

    spatial = pd.read_csv(SPATIAL_FILE)
    feedback = pd.read_csv(FEEDBACK_FILE)

    for label, frame in (
        ("Spatial annotations", spatial),
        ("Text feedback", feedback),
    ):
        missing_columns = set(KEY_COLUMNS) - set(frame.columns)

        if missing_columns:
            raise SystemExit(
                f"{label} is missing columns: {sorted(missing_columns)}"
            )

        if frame[KEY_COLUMNS].isna().any().any():
            raise SystemExit(f"{label} contains missing frame identifiers.")

        if frame.duplicated(KEY_COLUMNS).any():
            raise SystemExit(f"{label} contains duplicate frame identifiers.")

    spatial_keys = set(map(tuple, spatial[KEY_COLUMNS].to_numpy()))
    feedback_keys = set(map(tuple, feedback[KEY_COLUMNS].to_numpy()))

    spatial_only = spatial_keys - feedback_keys
    feedback_only = feedback_keys - spatial_keys

    if spatial_only or feedback_only:
        raise SystemExit(
            "Frame alignment failed: "
            f"{len(spatial_only)} spatial-only and "
            f"{len(feedback_only)} feedback-only frames."
        )

    merged = spatial.merge(
        feedback,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_spatial", "_feedback"),
        validate="one_to_one",
    )

    total_frames = len(feedback)

    if total_frames == 0:
        raise SystemExit("The feedback CSV contains no frames.")

    action_counts = feedback["action"].value_counts()
    evidence_counts = feedback["evidence_level"].value_counts()
    video_counts = feedback.groupby("video_id").size()

    action_table = action_counts.rename("frames").rename_axis("action").reset_index()
    action_table["percentage"] = (
        action_table["frames"] / total_frames * 100
    ).round(1)

    evidence_table = (
        evidence_counts.rename("frames")
        .rename_axis("evidence_level")
        .reset_index()
    )
    evidence_table["percentage"] = (
        evidence_table["frames"] / total_frames * 100
    ).round(1)

    video_table = (
        video_counts.rename("frames")
        .rename_axis("video_id")
        .reset_index()
    )

    obstruction_table = pd.crosstab(
        merged["potential_obstruction"],
        merged["action"],
    ).rename_axis("potential_obstruction").reset_index()

    if "research_only" in feedback.columns:
        research_only_count = (
            feedback["research_only"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            .sum()
        )
    else:
        research_only_count = 0

    report = [
        "# Text Feedback Baseline Evaluation",
        "",
        (
            "> Research-only evaluation. These results do not establish "
            "clinical safety or effectiveness for Parkinson's disease "
            "Freezing of Gait."
        ),
        "",
        "## Dataset integrity",
        "",
        f"- Spatial annotation rows: {len(spatial)}",
        f"- Feedback rows: {total_frames}",
        f"- Videos: {feedback['video_id'].nunique()}",
        f"- Spatial-only frame keys: {len(spatial_only)}",
        f"- Feedback-only frame keys: {len(feedback_only)}",
        "- Duplicate frame keys: none",
        f"- Rows marked research-only: {research_only_count}",
        "",
        "## Actions",
        "",
        markdown_table(action_table),
        "",
        "## Evidence levels",
        "",
        markdown_table(evidence_table),
        "",
        "## Frames per video",
        "",
        markdown_table(video_table),
        "",
        "## Potential obstruction versus selected action",
        "",
        markdown_table(obstruction_table),
        "",
        "## Interpretation and limitations",
        "",
        (
            "- A central, lower image-region bounding box triggers a "
            "potential-obstacle verification prompt. It does not confirm "
            "a physical obstruction."
        ),
        (
            "- No detected objects means insufficient evidence, not "
            "a confirmed clear path."
        ),
        (
            "- Failure to flag an object does not establish that "
            "a route is traversable."
        ),
        (
            "- Obstacle distances and safe route geometry have not "
            "been calibrated or validated by this baseline."
        ),
        (
            "- These counts do not measure FoG reduction or "
            "clinical effectiveness."
        ),
        "",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")

    print(f"Report saved to: {OUTPUT_FILE}")
    print(f"Frames evaluated: {total_frames}")
    print(f"Videos evaluated: {feedback['video_id'].nunique()}")
    print("Frame-key alignment: passed")

    print("\nAction counts:")
    print(action_counts.to_string())

    print("\nEvidence levels:")
    print(evidence_counts.to_string())


if __name__ == "__main__":
    main()
