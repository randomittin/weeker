"""Confidence-weighted Elo ability model (013 §2).

Pure, I/O-free functions. A learner's per-concept ability ``theta`` and each
question's ``q_rating`` live on one logit scale; an attempt nudges both by a
learning rate ``K`` that (a) decays with the number of attempts seen and (b) is
scaled by a confidence multiplier so a *confident* wrong answer moves ability
harder than a lucky guess. ``mastery`` maps ability onto ``[0, 1]`` around the
configured target logit. Every tunable comes from :mod:`weeker.core.config`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from weeker.core import config

Confidence = Literal["sure", "unsure", "guessing"]

# Absolute clamp on the logit ability scale. ``6.0`` logits is ~99.75% expected
# score against an average question — far past any real learner — so this is a
# numeric-safety rail, not a tunable (and 6 is a structural literal per the
# single-source lint, so it stays out of config).
THETA_ABS_MAX = 6.0


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one Elo update — new ratings plus the deltas that produced them."""

    theta: float
    q_rating: float
    theta_delta: float
    q_rating_delta: float
    expected: float
    k_theta: float
    k_q: float


def _logistic(x: float) -> float:
    """Numerically stable standard logistic ``1 / (1 + e^-x)``."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def expected(theta: float, q_rating: float) -> float:
    """Probability the learner answers correctly — logistic of the rating gap."""
    return _logistic(theta - q_rating)


def mastery(theta: float) -> float:
    """Map ability onto ``[0, 1]``, centred on ``MASTERY_TARGET_LOGIT``."""
    return _logistic(theta - config.MASTERY_TARGET_LOGIT)


def k_theta(attempts: int) -> float:
    """Ability learning rate: decays from ``ELO_K_BASE`` toward ``ELO_K_MIN``."""
    k = config.ELO_K_BASE / (1 + max(0, attempts))
    return _clamp(k, config.ELO_K_MIN, config.ELO_K_MAX)


def k_q(q_attempts: int) -> float:
    """Question rating learning rate: decays from ``ELO_KQ_BASE`` with exposure."""
    return config.ELO_KQ_BASE / (1 + max(0, q_attempts))


def update(
    theta: float,
    q_rating: float,
    attempts: int,
    q_attempts: int,
    correct: bool,
    confidence: Confidence,
) -> UpdateResult:
    """Apply one confidence-weighted Elo update to ability and question rating.

    ``attempts``/``q_attempts`` are the counts *before* this attempt (they set the
    decayed learning rates). Returns new ratings and the deltas; pure — no I/O.
    """
    outcome = "correct" if correct else "wrong"
    conf_mult = config.CONF_MULT[(outcome, confidence)]
    exp_score = expected(theta, q_rating)
    actual = 1.0 if correct else 0.0

    kt = k_theta(attempts)
    kq = k_q(q_attempts)

    raw_theta_delta = kt * conf_mult * (actual - exp_score)
    new_theta = _clamp(theta + raw_theta_delta, -THETA_ABS_MAX, THETA_ABS_MAX)
    theta_delta = new_theta - theta

    # Question rating moves opposite to the learner: it rises when the learner
    # fails (question proved hard), falls when the learner succeeds.
    q_delta = kq * conf_mult * (exp_score - actual)
    new_q = _clamp(q_rating + q_delta, -THETA_ABS_MAX, THETA_ABS_MAX)
    q_delta = new_q - q_rating

    return UpdateResult(
        theta=new_theta,
        q_rating=new_q,
        theta_delta=theta_delta,
        q_rating_delta=q_delta,
        expected=exp_score,
        k_theta=kt,
        k_q=kq,
    )
