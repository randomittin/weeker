"""T-33 tests — the 013 §8 calibration assertion and prediction mechanics."""

from __future__ import annotations

import random

from atlas.learn.predict import PredictItem, breakeven_p, predict_score


def test_uniform_08_learner_predicted_within_8pct():
    # 013 §8: a learner correct with p=0.8 everywhere. Attempt-all EV per 1-mark
    # question = 0.8 - 0.2*0.25 = 0.75 → true fraction 0.75. Predict within ±8%.
    rng = random.Random(42)
    items = [PredictItem(marks=1.0, p=0.8, covered=True) for _ in range(135)]
    pred = predict_score(items, passing_marks=90, total_marks=135, rng=rng)
    assert abs(pred.expected_fraction - 0.75) <= 0.08
    # The bootstrap band brackets the point estimate.
    assert pred.ci_low <= pred.expected_marks <= pred.ci_high
    assert not pred.suppressed


def test_breakeven_is_one_fifth_at_quarter_negative():
    assert abs(breakeven_p(0.25) - 0.2) < 1e-9


def test_low_probability_questions_are_skipped_not_penalised():
    # p below breakeven → skipped → contributes 0, never a negative drag.
    rng = random.Random(1)
    items = [PredictItem(marks=1.0, p=0.05, covered=True) for _ in range(50)]
    pred = predict_score(items, passing_marks=30, total_marks=50, rng=rng)
    assert abs(pred.expected_marks) < 1e-9
    assert pred.p_pass == 0.0


def test_strong_learner_predicted_to_pass():
    rng = random.Random(5)
    items = [PredictItem(marks=1.0, p=0.95, covered=True) for _ in range(135)]
    pred = predict_score(items, passing_marks=90, total_marks=135, rng=rng)
    assert pred.p_pass > 0.9
    assert pred.expected_fraction > 0.85


def test_coverage_suppression_below_threshold():
    rng = random.Random(9)
    # Only 20% of items covered → below PREDICT_MIN_COVERAGE (0.30) → suppressed.
    items = [PredictItem(marks=1.0, p=0.7, covered=(i < 2)) for i in range(10)]
    pred = predict_score(items, passing_marks=6, total_marks=10, rng=rng)
    assert abs(pred.coverage - 0.2) < 1e-9
    assert pred.suppressed is True
    # Enough coverage → not suppressed.
    items2 = [PredictItem(marks=1.0, p=0.7, covered=(i < 5)) for i in range(10)]
    pred2 = predict_score(items2, passing_marks=6, total_marks=10, rng=rng)
    assert pred2.suppressed is False


def test_negative_marking_reduces_expected_below_raw_probability():
    rng = random.Random(3)
    items = [PredictItem(marks=1.0, p=0.6, covered=True) for _ in range(100)]
    pred = predict_score(items, passing_marks=50, total_marks=100, rng=rng)
    # Raw expected-correct would be 60; negative marking pulls it to 0.6-0.4*0.25=0.5.
    assert abs(pred.expected_fraction - 0.5) <= 0.02


def test_two_mark_questions_weight_more():
    rng = random.Random(7)
    items = [PredictItem(marks=2.0, p=0.8, covered=True) for _ in range(10)]
    pred = predict_score(items, passing_marks=12, total_marks=20, rng=rng)
    # Same 0.75 fraction, but expressed over 20 marks.
    assert abs(pred.expected_fraction - 0.75) <= 0.08
    assert abs(pred.total_marks - 20.0) < 1e-9
