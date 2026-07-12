"""FSRS-lite retention model (013 §3).

Retrievability follows an exponential forgetting curve keyed on a per-concept
*stability* (days). A correct attempt grows stability (more when the concept is
already well mastered and the answer was confident); a wrong attempt collapses
it toward ``STAB_MIN`` (harder when the learner was sure). Pure, I/O-free.
"""

from __future__ import annotations

import math

from weeker.core import config
from weeker.learn.elo import Confidence, _clamp


def retrievability(stability: float, days_since: float) -> float:
    """Probability of recall ``R = e^(-t/S)`` — ``1.0`` at ``t=0``, decaying in ``t``."""
    if stability <= 0:
        return 0.0
    t = max(0.0, days_since)
    return math.exp(-t / stability)


def days_until_due(stability: float) -> float:
    """Days until retrievability falls to ``DUE_R_THRESHOLD`` for this stability."""
    if stability <= 0:
        return 0.0
    return -stability * math.log(config.DUE_R_THRESHOLD)


def stability_update(
    stability: float,
    mastery: float,
    correct: bool,
    confidence: Confidence,
) -> float:
    """Return the post-attempt stability, floored at ``STAB_MIN``.

    Success multiplies stability by a growth factor between ``STAB_GROW_BASE`` and
    ``STAB_GROW_MASTERY`` (interpolated by mastery), tempered by confidence so a
    guess consolidates little. Failure shrinks by ``STAB_FAIL_MULT`` divided by the
    confidence multiplier — a *sure* wrong answer resets stability hardest.
    """
    outcome = "correct" if correct else "wrong"
    conf_mult = config.CONF_MULT[(outcome, confidence)]

    if correct:
        m = _clamp(mastery, 0.0, 1.0)
        grow = config.STAB_GROW_BASE + (config.STAB_GROW_MASTERY - config.STAB_GROW_BASE) * m
        grow_eff = 1.0 + (grow - 1.0) * conf_mult
        new = stability * grow_eff
    else:
        new = stability * config.STAB_FAIL_MULT / conf_mult

    return max(config.STAB_MIN, new)
