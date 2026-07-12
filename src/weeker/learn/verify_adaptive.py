"""Synthetic adaptive-engine verification (013 §8, T-35 — the M4 gate).

Generates a synthetic learner whose true per-concept ability is known, drives a
200-attempt run of confidence-weighted Elo updates against probing questions, and
checks that the *recovered* ability tracks the *true* ability (Pearson r above a
floor) with mastery staying inside ``[0, 1]`` and no numeric blow-ups. Pure and
self-contained — no DB required — so it is a reliable milestone gate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from weeker.learn import elo


@dataclass
class VerifyReport:
    """Outcome of the synthetic run."""

    ok: bool = True
    correlation: float = 0.0
    n_attempts: int = 0
    n_concepts: int = 0
    checks: list = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.checks.append(f"FAIL {msg}")

    def note(self, msg: str) -> None:
        self.checks.append(f"ok   {msg}")


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def verify_adaptive(
    *,
    seed: int = 20240201,
    n_concepts: int = 12,
    n_attempts: int = 200,
    min_correlation: float = 0.7,
) -> VerifyReport:
    """Run the synthetic replay and return a pass/fail report."""
    rng = random.Random(seed)
    report = VerifyReport(n_attempts=n_attempts, n_concepts=n_concepts)

    true_theta = [rng.uniform(-2.0, 2.0) for _ in range(n_concepts)]
    theta = [0.0] * n_concepts
    attempts_seen = [0] * n_concepts

    for _ in range(n_attempts):
        c = rng.randrange(n_concepts)
        # Probe near the learner's true ability so attempts are informative.
        q_rating = true_theta[c] + rng.choice([-1.0, 0.0, 1.0])
        p_true = elo.expected(true_theta[c], q_rating)
        correct = rng.random() < p_true
        # A calibrated learner: confidence tracks true success probability.
        confidence = "sure" if p_true >= 0.7 else "unsure" if p_true >= 0.45 else "guessing"
        res = elo.update(
            theta[c], q_rating, attempts_seen[c], attempts_seen[c], correct, confidence
        )
        theta[c] = res.theta
        attempts_seen[c] += 1
        m = elo.mastery(res.theta)
        if not (0.0 <= m <= 1.0):
            report.fail(f"mastery out of [0,1]: {m}")
        if not math.isfinite(res.theta) or abs(res.theta) > elo.THETA_ABS_MAX + 1e-9:
            report.fail(f"theta unbounded: {res.theta}")

    corr = _pearson(true_theta, theta)
    report.correlation = corr
    if corr >= min_correlation:
        report.note(f"recovered ability tracks truth (r={corr:.3f} ≥ {min_correlation})")
    else:
        report.fail(f"ability recovery too weak (r={corr:.3f} < {min_correlation})")

    if report.ok:
        report.note(f"{n_attempts} attempts, {n_concepts} concepts, bounds + [0,1] held")
    return report
