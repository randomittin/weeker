"""Score prediction with a bootstrap confidence band (013 §7, 017 §4, T-33).

Given a learner's per-question success probabilities, predict the exam score
under the negative-marking attempt policy of 017 §6 (attempt iff EV-positive, i.e.
``p ≥ neg/(1+neg)``). The point estimate is the analytic expected score; the CI
and ``P(pass)`` come from a bootstrap that perturbs ability by its standard error
(``PREDICT_SE_BASE`` shrinking with attempts) and draws outcomes. When too little
of the syllabus has been practised (coverage below ``PREDICT_MIN_COVERAGE``) the
prediction is flagged ``suppressed`` — an honest "not enough data yet".
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from atlas.core import config
from atlas.core.db import get_session
from atlas.core.models import Attempt, Mastery, Question
from atlas.learn import elo

# Exam rule from 017 §1 / the course manifest (mirrored into ``courses.exam_config``
# at runtime). Kept here as the pure-function default, not an adaptive tunable.
DEFAULT_NEG_FRACTION = 0.25
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5


@dataclass
class PredictItem:
    """One prospective exam question: its success prob and marks."""

    marks: float = 1.0
    p: float | None = None  # explicit P(correct); else derived from theta/q_rating
    theta: float = 0.0
    q_rating: float = 0.0
    attempts: int = 0
    covered: bool = True

    def prob(self) -> float:
        return self.p if self.p is not None else elo.expected(self.theta, self.q_rating)

    def se(self, se_base: float) -> float:
        return se_base / (1.0 + max(0, self.attempts)) ** 0.5


@dataclass(frozen=True)
class Prediction:
    """Predicted exam outcome with a bootstrap band and pass probability."""

    expected_marks: float
    expected_fraction: float
    ci_low: float
    ci_high: float
    p_pass: float
    coverage: float
    suppressed: bool
    total_marks: float
    passing_marks: float


def breakeven_p(neg_fraction: float) -> float:
    """Success probability at which attempting is EV-neutral (0.2 at neg=0.25)."""
    return neg_fraction / (1.0 + neg_fraction)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def predict_score(
    items: list[PredictItem],
    *,
    passing_marks: float,
    total_marks: float | None = None,
    neg_fraction: float = DEFAULT_NEG_FRACTION,
    bootstrap: int = config.PREDICT_BOOTSTRAP,
    se_base: float = config.PREDICT_SE_BASE,
    min_coverage: float = config.PREDICT_MIN_COVERAGE,
    rng: random.Random | None = None,
) -> Prediction:
    """Predict the marks-weighted score with a bootstrap CI and ``P(pass)``."""
    rng = rng or random.Random()
    total = total_marks if total_marks is not None else sum(i.marks for i in items)
    total = total or 1.0
    breakeven = breakeven_p(neg_fraction)

    # Analytic expected score under the attempt policy.
    expected = 0.0
    for it in items:
        p = it.prob()
        if p >= breakeven:
            expected += p * it.marks - (1.0 - p) * neg_fraction * it.marks

    # Bootstrap: perturb ability, re-decide attempts, draw outcomes.
    scores: list[float] = []
    passes = 0
    for _ in range(max(1, bootstrap)):
        s = 0.0
        for it in items:
            if it.p is not None:
                p = it.p
            else:
                theta = rng.gauss(it.theta, it.se(se_base))
                p = elo.expected(theta, it.q_rating)
            if p < breakeven:
                continue  # skip — no marks either way
            if rng.random() < p:
                s += it.marks
            else:
                s -= neg_fraction * it.marks
        scores.append(s)
        if s >= passing_marks:
            passes += 1
    scores.sort()

    coverage = (sum(1 for i in items if i.covered) / len(items)) if items else 0.0
    return Prediction(
        expected_marks=expected,
        expected_fraction=expected / total,
        ci_low=_percentile(scores, CI_LOW_PCT),
        ci_high=_percentile(scores, CI_HIGH_PCT),
        p_pass=passes / len(scores),
        coverage=coverage,
        suppressed=coverage < min_coverage,
        total_marks=total,
        passing_marks=passing_marks,
    )


def predict_from_db(
    db=None,
    *,
    user_id: str,
    course_id: object,
    passing_marks: float = 90.0,
    total_marks: float = 150.0,
    neg_fraction: float = DEFAULT_NEG_FRACTION,
    rng: random.Random | None = None,
) -> Prediction:
    """Build prediction items from the live bank + this learner's mastery.

    Each active standalone question becomes an item: success probability from the
    learner's concept ability against the question rating, ``covered`` when the
    concept has at least one attempt. Section-aware mock prediction (per 017 §4)
    is layered on in Wave 3; this is the whole-bank estimate.
    """
    from sqlalchemy import func, select

    own = db is None
    db = db or get_session()
    try:
        theta_by_concept = {
            m.concept_id: float(m.theta)
            for m in db.scalars(select(Mastery).where(Mastery.user_id == user_id)).all()
        }
        attempt_counts = dict(
            db.execute(
                select(Attempt.concept_id, func.count())
                .where(Attempt.user_id == user_id)
                .group_by(Attempt.concept_id)
            ).all()
        )
        questions = db.scalars(
            select(Question).where(
                Question.course_id == course_id,
                Question.status == "active",
                Question.case_group_id.is_(None),
            )
        ).all()
        items = [
            PredictItem(
                marks=float(q.marks),
                theta=theta_by_concept.get(q.concept_id, 0.0),
                q_rating=float(q.q_rating),
                attempts=int(attempt_counts.get(q.concept_id, 0)),
                covered=attempt_counts.get(q.concept_id, 0) > 0,
            )
            for q in questions
        ]
    finally:
        if own:
            db.close()

    return predict_score(
        items,
        passing_marks=passing_marks,
        total_marks=total_marks,
        neg_fraction=neg_fraction,
        rng=rng,
    )
