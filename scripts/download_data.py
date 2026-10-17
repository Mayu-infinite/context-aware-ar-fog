from __future__ import annotations

from pathlib import Path

DIRECTORIES = [
    Path("data/raw"),
    Path("data/interim"),
    Path("data/processed"),
    Path("data/annotations"),
]


def main() -> None:
    for directory in DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Ready: {directory}"
        )

    print()
    print(
        "Dataset directories are ready."
    )
    print(
        "Place downloaded research datasets "
        "inside data/raw/."
    )


if __name__ == "__main__":
    main()