"""Export an engine YAML profile as JSON for the Quest app (CafogEngineBehaviour.configJson).

uv run python scripts/export_unity_config.py configs/spatial_engine.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cafog.spatial import EngineConfig

DEFAULT_OUT = Path("unity/CafogQuest/Assets/Cafog/Config/cafog_config.json")


def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def to_unity(value: Any) -> Any:
    if isinstance(value, dict):
        return {camel(k): to_unity(v) for k, v in value.items()}
    return value


def unity_config(config: EngineConfig) -> dict[str, Any]:
    data = to_unity(config.to_dict())
    # Unity's JsonUtility has no null; the C# profile treats deltaLim <= 0 as healthy.
    if data["profile"]["deltaLim"] is None:
        data["profile"]["deltaLim"] = 0.0
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = unity_config(EngineConfig.from_yaml(args.config))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} (profile {data['profile']['name']})")


if __name__ == "__main__":
    main()
