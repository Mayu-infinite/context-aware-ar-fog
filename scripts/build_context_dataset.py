from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a context dataset from JSON labels."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/annotations/raw_labels.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/context_dataset.jsonl"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input labels not found: {args.input}"
        )

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Input JSON must contain a list of records."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Wrote {len(records)} records to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()