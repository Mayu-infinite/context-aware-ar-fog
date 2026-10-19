from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_unit_interval(name: str, value: float | None) -> None:
    """Validate a normalized value when it is available."""
    if value is None:
        return

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class ValueInputs:
    """Normalized evidence describing one candidate spatial state.

    All supplied values must be in [0, 1].
    Higher goal_progress and clearance are preferable.
    Higher uncertainty and turn_complexity are less preferable.

    None means unavailable evidence, not a zero score or a safe path.
    These inputs are engineering features, not clinical measurements.
    """

    goal_progress: float | None = None
    clearance: float | None = None
    uncertainty: float = 1.0
    turn_complexity: float | None = None

    def __post_init__(self) -> None:
        _validate_unit_interval("goal_progress", self.goal_progress)
        _validate_unit_interval("clearance", self.clearance)
        _validate_unit_interval("uncertainty", self.uncertainty)
        _validate_unit_interval("turn_complexity", self.turn_complexity)


@dataclass(frozen=True)
class ValueWeights:
    """Non-negative weights for the interpretable baseline value field."""

    goal_progress: float = 1.0
    clearance: float = 1.5
    uncertainty: float = 2.0
    turn_complexity: float = 1.0

    def __post_init__(self) -> None:
        for name, weight in (
            ("goal_progress", self.goal_progress),
            ("clearance", self.clearance),
            ("uncertainty", self.uncertainty),
            ("turn_complexity", self.turn_complexity),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative.")

        if (
            self.goal_progress
            + self.clearance
            + self.uncertainty
            + self.turn_complexity
            == 0.0
        ):
            raise ValueError("At least one weight must be greater than zero.")


@dataclass(frozen=True)
class ValueEstimate:
    """Value score plus evidence coverage for one spatial state."""

    value: float
    confidence: float
    available_terms: tuple[str, ...]


class ValueField:
    """Calculate a transparent baseline spatial desirability score.

    This is a proposed engineering baseline, not the exact model from
    the referenced research paper. It does not estimate clinical FoG risk,
    certify walkability, or directly decide whether to issue an AR cue.
    """

    def __init__(self, weights: ValueWeights | None = None) -> None:
        self.weights = weights or ValueWeights()

    def estimate(self, inputs: ValueInputs) -> ValueEstimate:
        """Return a normalized score in [0, 1] and evidence coverage.

        Missing goal progress, clearance, and turn complexity are excluded
        from the weighted average. Uncertainty is always included, so
        unavailable spatial evidence defaults to maximum uncertainty.

        A high value is not proof that a route is safe.
        """
        weights = self.weights
        weighted_terms: list[tuple[str, float, float]] = []

        if inputs.goal_progress is not None and weights.goal_progress > 0:
            weighted_terms.append(
                ("goal_progress", inputs.goal_progress, weights.goal_progress)
            )

        if inputs.clearance is not None and weights.clearance > 0:
            weighted_terms.append(("clearance", inputs.clearance, weights.clearance))

        if inputs.turn_complexity is not None and weights.turn_complexity > 0:
            weighted_terms.append(
                (
                    "turn_complexity",
                    1.0 - inputs.turn_complexity,
                    weights.turn_complexity,
                )
            )

        if weights.uncertainty > 0:
            weighted_terms.append(
                ("uncertainty", 1.0 - inputs.uncertainty, weights.uncertainty)
            )

        total_weight = sum(weight for _, _, weight in weighted_terms)
        if total_weight <= 0.0:
            # This can happen if every positively weighted optional term
            # is missing and uncertainty has zero weight.
            return ValueEstimate(
                value=0.0,
                confidence=0.0,
                available_terms=(),
            )

        value = sum(
            score * weight for _, score, weight in weighted_terms
        ) / total_weight

        optional_weight_total = (
            weights.goal_progress
            + weights.clearance
            + weights.turn_complexity
        )
        observed_optional_weight = sum(
            weight
            for name, _, weight in weighted_terms
            if name != "uncertainty"
        )

        confidence = (
            observed_optional_weight / optional_weight_total
            if optional_weight_total > 0.0
            else 0.0
        )

        return ValueEstimate(
            value=float(value),
            confidence=float(confidence),
            available_terms=tuple(name for name, _, _ in weighted_terms),
        )


def value_difference(
    current: ValueEstimate,
    next_state: ValueEstimate,
    *,
    minimum_confidence: float = 0.25,
) -> float | None:
    """Return V(next) - V(current), or None when evidence is too weak.

    The sign convention follows the value-difference definition used in
    the research paper. A negative result is not, by itself, a cue trigger.
    """
    _validate_unit_interval("minimum_confidence", minimum_confidence)

    if (
        current.confidence < minimum_confidence
        or next_state.confidence < minimum_confidence
    ):
        return None

    return float(next_state.value - current.value)
