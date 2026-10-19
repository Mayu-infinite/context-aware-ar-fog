from __future__ import annotations

from dataclasses import dataclass

from cafog.planning.value_field import ValueEstimate, value_difference


@dataclass(frozen=True)
class ValueChange:
    """A value change between two sufficiently supported observations."""

    previous_value: float
    current_value: float
    delta: float


class TemporalValueTracker:
    """Track consecutive estimates and reject unsupported value changes.

    Call update(None) when an observation is missing or invalid. This resets
    the comparison baseline so a later observation cannot be compared with
    a stale estimate.

    This tracker does not assess route safety or trigger AR cues.
    """

    def __init__(self, minimum_confidence: float = 0.25) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1].")

        self.minimum_confidence = minimum_confidence
        self._previous: ValueEstimate | None = None

    def reset(self) -> None:
        """Discard the previous estimate."""
        self._previous = None

    def update(self, estimate: ValueEstimate | None) -> ValueChange | None:
        """Update the baseline and return a change only for valid pairs."""
        if estimate is None:
            self.reset()
            return None

        if (
            not 0.0 <= estimate.value <= 1.0
            or not 0.0 <= estimate.confidence <= 1.0
        ):
            self.reset()
            return None

        if estimate.confidence < self.minimum_confidence:
            self.reset()
            return None

        previous = self._previous
        self._previous = estimate

        if previous is None:
            return None

        delta = value_difference(
            previous,
            estimate,
            minimum_confidence=self.minimum_confidence,
        )
        if delta is None:
            return None

        return ValueChange(
            previous_value=previous.value,
            current_value=estimate.value,
            delta=delta,
        )
