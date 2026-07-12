"""T-30 property tests for the confidence-weighted Elo ability model."""

from __future__ import annotations

import random

from weeker.core import config
from weeker.learn import elo


def test_mastery_in_unit_interval():
    for theta in [-100, -6, -1, 0, config.MASTERY_TARGET_LOGIT, 1, 6, 100]:
        m = elo.mastery(theta)
        assert 0.0 <= m <= 1.0
    # Centred on the target logit → exactly 0.5 there.
    assert abs(elo.mastery(config.MASTERY_TARGET_LOGIT) - 0.5) < 1e-9
    # Monotone increasing in ability.
    assert elo.mastery(-1) < elo.mastery(0) < elo.mastery(1)


def test_theta_bounded_under_relentless_streak():
    rng = random.Random(1)
    theta = 0.0
    for i in range(2000):
        # A relentlessly-correct learner against easy questions.
        res = elo.update(theta, q_rating=-2.0, attempts=i, q_attempts=i,
                         correct=True, confidence="sure")
        theta = res.theta
        assert -elo.THETA_ABS_MAX <= theta <= elo.THETA_ABS_MAX
    # And a relentlessly-wrong learner stays bounded below.
    theta = 0.0
    for i in range(2000):
        res = elo.update(theta, q_rating=2.0, attempts=i, q_attempts=i,
                         correct=False, confidence="sure")
        theta = res.theta
        assert -elo.THETA_ABS_MAX <= theta <= elo.THETA_ABS_MAX
    _ = rng  # determinism anchor


def test_confident_wrong_moves_more_than_guessed_wrong():
    theta, q = 0.5, 0.0
    sure = elo.update(theta, q, attempts=3, q_attempts=3, correct=False, confidence="sure")
    guess = elo.update(theta, q, attempts=3, q_attempts=3, correct=False, confidence="guessing")
    assert abs(sure.theta_delta) > abs(guess.theta_delta)
    # Both wrong answers push ability down.
    assert sure.theta_delta < 0 and guess.theta_delta < 0


def test_confident_wrong_moves_more_than_unsure_wrong():
    theta, q = 0.2, 0.1
    sure = elo.update(theta, q, attempts=2, q_attempts=2, correct=False, confidence="sure")
    unsure = elo.update(theta, q, attempts=2, q_attempts=2, correct=False, confidence="unsure")
    assert abs(sure.theta_delta) > abs(unsure.theta_delta)


def test_k_decays_with_attempts():
    ks = [elo.k_theta(n) for n in range(0, 50)]
    # Strictly non-increasing, and clamped into [MIN, MAX].
    for a, b in zip(ks, ks[1:], strict=False):
        assert b <= a + 1e-12
    assert ks[0] == config.ELO_K_BASE
    assert all(config.ELO_K_MIN <= k <= config.ELO_K_MAX for k in ks)
    assert elo.k_theta(10_000) == config.ELO_K_MIN


def test_correct_raises_ability_wrong_lowers():
    up = elo.update(0.0, 0.0, 1, 1, correct=True, confidence="sure")
    down = elo.update(0.0, 0.0, 1, 1, correct=False, confidence="sure")
    assert up.theta_delta > 0
    assert down.theta_delta < 0


def test_question_rating_moves_opposite_to_learner():
    # Learner fails → question rating rises (it proved hard).
    res = elo.update(0.0, 0.0, 1, 1, correct=False, confidence="sure")
    assert res.q_rating_delta > 0
    # Learner succeeds → question rating falls.
    res = elo.update(0.0, 0.0, 1, 1, correct=True, confidence="sure")
    assert res.q_rating_delta < 0


def test_expected_score_matches_logistic_gap():
    assert abs(elo.expected(0.0, 0.0) - 0.5) < 1e-9
    assert elo.expected(2.0, 0.0) > 0.8
    assert elo.expected(-2.0, 0.0) < 0.2
