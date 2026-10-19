import pytest

from cafog.planning.value_field import (
    ValueEstimate,
    ValueField,
    ValueInputs,
    ValueWeights,
    value_difference,
)


def test_high_clearance_and_low_uncertainty_increase_value() -> None:
    field = ValueField()

    favorable = field.estimate(
        ValueInputs(
            goal_progress=0.8,
            clearance=0.9,
            uncertainty=0.1,
            turn_complexity=0.1,
        )
    )
    unfavorable = field.estimate(
        ValueInputs(
            goal_progress=0.8,
            clearance=0.2,
            uncertainty=0.8,
            turn_complexity=0.1,
        )
    )

    assert 0.0 <= unfavorable.value <= 1.0
    assert 0.0 <= favorable.value <= 1.0
    assert favorable.value > unfavorable.value


def test_complex_turn_reduces_value() -> None:
    field = ValueField()

    simple_turn = field.estimate(
        ValueInputs(
            goal_progress=0.5,
            clearance=0.7,
            uncertainty=0.2,
            turn_complexity=0.1,
        )
    )
    complex_turn = field.estimate(
        ValueInputs(
            goal_progress=0.5,
            clearance=0.7,
            uncertainty=0.2,
            turn_complexity=0.9,
        )
    )

    assert simple_turn.value > complex_turn.value


def test_missing_clearance_is_not_reported_as_available() -> None:
    estimate = ValueField().estimate(
        ValueInputs(
            goal_progress=0.5,
            clearance=None,
            uncertainty=1.0,
            turn_complexity=None,
        )
    )

    assert "clearance" not in estimate.available_terms
    assert estimate.confidence < 1.0


def test_all_optional_evidence_missing_has_zero_confidence() -> None:
    estimate = ValueField().estimate(ValueInputs())

    assert estimate.confidence == 0.0
    assert estimate.available_terms == ("uncertainty",)
    assert estimate.value == 0.0


def test_value_difference_uses_next_minus_current() -> None:
    current = ValueEstimate(
        value=0.4,
        confidence=0.8,
        available_terms=("clearance",),
    )
    next_state = ValueEstimate(
        value=0.7,
        confidence=0.8,
        available_terms=("clearance",),
    )

    assert value_difference(current, next_state) == pytest.approx(0.3)
    assert value_difference(next_state, current) == pytest.approx(-0.3)


def test_value_difference_returns_none_when_confidence_is_low() -> None:
    current = ValueEstimate(
        value=0.8,
        confidence=0.1,
        available_terms=(),
    )
    next_state = ValueEstimate(
        value=0.2,
        confidence=0.9,
        available_terms=("clearance",),
    )

    assert value_difference(current, next_state) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("goal_progress", -0.1),
        ("clearance", 1.1),
        ("uncertainty", float("nan")),
        ("turn_complexity", float("inf")),
    ],
)
def test_inputs_reject_invalid_normalized_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        ValueInputs(**{field: value})


def test_weights_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        ValueWeights(clearance=-1.0)


def test_all_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        ValueWeights(
            goal_progress=0.0,
            clearance=0.0,
            uncertainty=0.0,
            turn_complexity=0.0,
        )