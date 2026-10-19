import pytest

from cafog.planning.temporal_value import TemporalValueTracker
from cafog.planning.value_field import ValueEstimate


def estimate(value: float, confidence: float = 0.8) -> ValueEstimate:
    return ValueEstimate(
        value=value,
        confidence=confidence,
        available_terms=("clearance",),
    )


def test_first_estimate_establishes_baseline():
    tracker = TemporalValueTracker()

    assert tracker.update(estimate(0.4)) is None


def test_consecutive_estimates_produce_next_minus_previous():
    tracker = TemporalValueTracker()

    assert tracker.update(estimate(0.4)) is None
    change = tracker.update(estimate(0.7))

    assert change is not None
    assert change.previous_value == pytest.approx(0.4)
    assert change.current_value == pytest.approx(0.7)
    assert change.delta == pytest.approx(0.3)


def test_negative_change_is_preserved_as_diagnostic():
    tracker = TemporalValueTracker()

    tracker.update(estimate(0.8))
    change = tracker.update(estimate(0.3))

    assert change is not None
    assert change.delta == pytest.approx(-0.5)


def test_low_confidence_estimate_resets_baseline():
    tracker = TemporalValueTracker()

    tracker.update(estimate(0.4))
    assert tracker.update(estimate(0.7, confidence=0.1)) is None

    # Must establish a fresh baseline rather than compare against 0.4.
    assert tracker.update(estimate(0.9)) is None


def test_missing_estimate_resets_baseline():
    tracker = TemporalValueTracker()

    tracker.update(estimate(0.4))
    assert tracker.update(None) is None
    assert tracker.update(estimate(0.9)) is None


def test_explicit_reset_discards_previous_estimate():
    tracker = TemporalValueTracker()

    tracker.update(estimate(0.4))
    tracker.reset()

    assert tracker.update(estimate(0.8)) is None


def test_invalid_estimate_resets_baseline():
    tracker = TemporalValueTracker()

    tracker.update(estimate(0.4))
    assert tracker.update(estimate(1.2)) is None
    assert tracker.update(estimate(0.8)) is None