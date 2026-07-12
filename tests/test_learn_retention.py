"""T-30 property tests for the FSRS-lite retention model."""

from __future__ import annotations

from weeker.core import config
from weeker.learn import retention


def test_retrievability_starts_at_one_and_decays():
    assert abs(retention.retrievability(5.0, 0.0) - 1.0) < 1e-9
    r = [retention.retrievability(5.0, d) for d in range(0, 30)]
    for a, b in zip(r, r[1:], strict=False):
        assert b <= a
    assert all(0.0 <= x <= 1.0 for x in r)


def test_retrievability_increases_with_stability():
    # More stable memory → higher recall at the same elapsed time.
    assert retention.retrievability(10.0, 3.0) > retention.retrievability(2.0, 3.0)


def test_stability_monotone_over_three_successes():
    for conf in ("sure", "unsure", "guessing"):
        s = 1.0
        seen = [s]
        for _ in range(3):
            s = retention.stability_update(s, mastery=0.7, correct=True, confidence=conf)
            seen.append(s)
        for a, b in zip(seen, seen[1:], strict=False):
            assert b > a, f"stability not increasing for confidence={conf}"


def test_failure_collapses_stability_and_floors():
    s = 20.0
    s2 = retention.stability_update(s, mastery=0.5, correct=False, confidence="unsure")
    assert s2 < s
    assert s2 >= config.STAB_MIN
    # Floor holds even from a tiny stability.
    assert retention.stability_update(0.6, 0.1, correct=False, confidence="sure") == config.STAB_MIN


def test_confident_wrong_resets_harder_than_guessed_wrong():
    s = 30.0
    sure = retention.stability_update(s, 0.5, correct=False, confidence="sure")
    guess = retention.stability_update(s, 0.5, correct=False, confidence="guessing")
    assert sure < guess


def test_high_mastery_grows_stability_faster():
    lo = retention.stability_update(5.0, mastery=0.1, correct=True, confidence="sure")
    hi = retention.stability_update(5.0, mastery=0.95, correct=True, confidence="sure")
    assert hi > lo


def test_days_until_due_positive_and_scales_with_stability():
    d1 = retention.days_until_due(5.0)
    d2 = retention.days_until_due(10.0)
    assert 0 < d1 < d2
    # At exactly that horizon, retrievability equals the due threshold.
    assert abs(retention.retrievability(5.0, d1) - config.DUE_R_THRESHOLD) < 1e-9
