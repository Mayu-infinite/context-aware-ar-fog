
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cafog.context.state import ContextType

from .actions import ActionType


@dataclass
class Cue:
    """Output of the cue policy."""

    action: ActionType
    visual_cue: str
    audio_cue: str
    explanation: str
    text_feedback: str = "NONE"


class CuePolicy:
    """Convert environmental context into actions and feedback."""

    def __init__(
        self,
        config_path: str | Path = "configs/cue_policy.yaml",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}

        with self.config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def select(self, context: ContextType) -> Cue:
        context_name = context.value
        config = self.config.get(
            context_name,
            self.config.get("UNKNOWN", {}),
        )

        action_name = config.get("action", "NO_AUTOMATIC_ACTION")
        visual_cue = config.get("visual_cue", "NONE")
        audio_cue = config.get("audio_cue", "NONE")
        text_feedback = config.get("text_feedback", "NONE")

        try:
            action = ActionType(action_name)
        except ValueError:
            action = ActionType.NO_AUTOMATIC_ACTION

        # UNKNOWN must never produce a directional instruction.
        if context == ContextType.UNKNOWN:
            action = ActionType.NO_AUTOMATIC_ACTION
            visual_cue = "NONE"
            audio_cue = "NONE"
            text_feedback = "NONE"

        explanation = (
            f"Context={context_name}; "
            f"Action={action.value}; "
            f"Visual={visual_cue}; "
            f"Audio={audio_cue}; "
            f"Text={text_feedback}"
        )

        return Cue(
            action=action,
            visual_cue=visual_cue,
            audio_cue=audio_cue,
            explanation=explanation,
            text_feedback=text_feedback,
        )
