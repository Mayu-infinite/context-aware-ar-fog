
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cafog.context.state import ContextType
from cafog.planning.value_field import ValueEstimate

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
    """Convert context into conservative, evidence-aware cue decisions.

    This is a research prototype, not a clinically validated mobility aid.
    Value estimates can inform confidence checks but do not certify safety.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/cue_policy.yaml",
        *,
        minimum_value_confidence: float = 0.25,
    ) -> None:
        if (
            not math.isfinite(minimum_value_confidence)
            or not 0.0 <= minimum_value_confidence <= 1.0
        ):
            raise ValueError(
                "minimum_value_confidence must be finite and in [0, 1]."
            )

        self.config_path = Path(config_path)
        self.minimum_value_confidence = minimum_value_confidence
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}

        with self.config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @staticmethod
    def _no_cue(context: ContextType, reason: str) -> Cue:
        return Cue(
            action=ActionType.NO_AUTOMATIC_ACTION,
            visual_cue="NONE",
            audio_cue="NONE",
            text_feedback="NONE",
            explanation=f"Context={context.value}; no cue: {reason}",
        )

    def select(
        self,
        context: ContextType,
        *,
        value_estimate: ValueEstimate | None = None,
        value_change: float | None = None,
        evidence_sufficient: bool = True,
    ) -> Cue:
        """Choose a cue only when context and evidence support it.

        value_change is diagnostic context, not an independent trigger.
        A negative value change never automatically generates a cue.
        evidence_sufficient=False suppresses automatic actions.
        """
        if context == ContextType.UNKNOWN:
            return self._no_cue(context, "context is unknown")

        if not evidence_sufficient:
            return self._no_cue(
                context,
                "scene evidence is insufficient",
            )

        if value_estimate is not None:
            confidence = value_estimate.confidence

            if (
                not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
                or confidence < self.minimum_value_confidence
            ):
                return self._no_cue(
                    context,
                    "value estimate confidence is invalid or low",
                )

        context_config = self.config.get(
            context.value,
            self.config.get("UNKNOWN", {}),
        )

        action_name = context_config.get(
            "action",
            "NO_AUTOMATIC_ACTION",
        )
        visual_cue = context_config.get("visual_cue", "NONE")
        audio_cue = context_config.get("audio_cue", "NONE")
        text_feedback = context_config.get("text_feedback", "NONE")

        try:
            action = ActionType(action_name)
        except (ValueError, TypeError):
            return self._no_cue(
                context,
                "configured action is invalid",
            )

        directional_actions = {
            ActionType.TURN_LEFT,
            ActionType.TURN_RIGHT,
            ActionType.FOLLOW_SAFE_PATH,
            ActionType.CENTER_PATH,
        }

        if action in directional_actions and not evidence_sufficient:
            return self._no_cue(
                context,
                "directional evidence is insufficient",
            )

        explanation = (
            f"Context={context.value}; "
            f"Action={action.value}; "
            f"Visual={visual_cue}; "
            f"Audio={audio_cue}; "
            f"Text={text_feedback}; "
            f"ValueChange={value_change}"
        )

        return Cue(
            action=action,
            visual_cue=visual_cue,
            audio_cue=audio_cue,
            explanation=explanation,
            text_feedback=text_feedback,
        )
