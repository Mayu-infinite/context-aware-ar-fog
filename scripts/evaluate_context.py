from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect context label distribution."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/context_dataset.jsonl"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}"
        )

    contexts: Counter[str] = Counter()

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            context = record.get(
                "context",
                "UNKNOWN",
            )

            contexts[context] += 1

    print("Context distribution:")
    print()

    for context, count in contexts.most_common():
        print(
            f"{context:25s} {count}"
        )


if __name__ == "__main__":
    main()