"""T-25 — question calibration: rating init, live Elo hook, retirement.

A question's ``q_rating`` starts from its authored difficulty and thereafter moves
on the shared Elo scale with every attempt (the attempt write path applies the
update; the hooks here read the same :class:`~atlas.learn.elo.UpdateResult`). Over
time the bank is audited:

* a rating that has drifted past :data:`RECALIBRATE_DIVERGENCE` logits from its
  authored init is flagged for recalibration;
* a distractor no learner picks after :data:`DEAD_DISTRACTOR_MIN_ATTEMPTS` pulls is
  a dead distractor;
* a question whose discrimination falls below :data:`DISCRIMINATION_FLOOR` at
  :data:`RETIRE_MIN_ATTEMPTS`+ attempts no longer separates strong from weak
  learners and is retired.

:func:`initial_q_rating` is consumed by the generation pipeline (T-24); the audit
functions drive the ``atlas generate recalibrate`` maintenance command.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core.models import Attempt, Question
from atlas.learn.elo import UpdateResult

# Authored difficulty (1 easy … 3 hard) → starting q_rating on the logit scale.
# Centered on 0 (an average question) so an average learner (theta≈0) sees ~50%.
DIFFICULTY_RATING = {1: -0.6, 2: 0.0, 3: 0.6}
# Live rating this far from the authored init → flag for human recalibration.
RECALIBRATE_DIVERGENCE = 1.5
# Minimum total option pulls before a never-picked distractor counts as dead.
DEAD_DISTRACTOR_MIN_ATTEMPTS = 20
# Point-biserial discrimination floor and the attempt count at which it applies.
DISCRIMINATION_FLOOR = 0.05
RETIRE_MIN_ATTEMPTS = 20


def initial_q_rating(difficulty: int) -> float:
    """Starting question rating for an authored difficulty tier (1/2/3)."""
    return DIFFICULTY_RATING.get(int(difficulty), 0.0)


def apply_rating_value(_current: float, result: UpdateResult) -> float:
    """Live hook: the new question rating is read off the shared Elo result."""
    return result.q_rating


def apply_rating(question: Question, result: UpdateResult) -> float:
    """Apply the shared Elo result's ``q_rating`` onto a question row."""
    question.q_rating = result.q_rating
    return result.q_rating


def needs_recalibration(question: Question) -> bool:
    """True when the live rating has drifted past the divergence threshold."""
    init = initial_q_rating(question.difficulty)
    return abs(float(question.q_rating) - init) > RECALIBRATE_DIVERGENCE


def dead_distractors(
    question: Question, min_attempts: int = DEAD_DISTRACTOR_MIN_ATTEMPTS
) -> list[str]:
    """Distractor keys never chosen after enough total pulls (dead distractors)."""
    pulls = question.pull_count or {}
    total = sum(int(v) for v in pulls.values())
    if total < min_attempts:
        return []
    correct = question.correct_key
    keys = [str(o.get("key")) for o in (question.options or [])]
    return [k for k in keys if k != correct and int(pulls.get(k, 0)) == 0]


def compute_discrimination(pairs: list[tuple[float, bool]]) -> float:
    """Point-biserial correlation between ability (theta) and correctness.

    ``pairs`` is ``(theta_before, correct)`` per attempt. Returns 0.0 when it is
    undefined (all-correct, all-wrong, or zero ability variance).
    """
    n = len(pairs)
    if n < 2:
        return 0.0
    thetas = [t for t, _ in pairs]
    mean = sum(thetas) / n
    var = sum((t - mean) ** 2 for t in thetas) / n
    if var == 0:
        return 0.0
    sd = var**0.5
    n_correct = sum(1 for _, c in pairs if c)
    p = n_correct / n
    if p in (0.0, 1.0):
        return 0.0
    m1 = sum(t for t, c in pairs if c) / n_correct
    m0 = sum(t for t, c in pairs if not c) / (n - n_correct)
    return ((m1 - m0) / sd) * (p * (1 - p)) ** 0.5


def should_retire(discrimination: float | None, attempts: int) -> bool:
    """Retire a question that stops discriminating at enough attempts."""
    if attempts < RETIRE_MIN_ATTEMPTS or discrimination is None:
        return False
    return discrimination < DISCRIMINATION_FLOOR


@dataclass
class CalibrationFlags:
    """Per-question audit outcome."""

    question_id: object
    attempts: int
    discrimination: float | None
    recalibrate: bool
    dead: list[str] = field(default_factory=list)
    retire: bool = False


def audit_question(session: Session, question_id: uuid.UUID) -> CalibrationFlags:
    """Recompute discrimination from the attempt log; flag + retire as warranted."""
    question = session.get(Question, question_id)
    if question is None:
        raise ValueError(f"question {question_id} not found")

    pairs = [
        (float(theta), bool(correct))
        for theta, correct in session.execute(
            select(Attempt.theta_before, Attempt.correct).where(
                Attempt.question_id == question_id
            )
        ).all()
    ]
    attempts = len(pairs)
    disc = compute_discrimination(pairs) if attempts else None
    if disc is not None:
        question.discrimination = disc

    retire = should_retire(disc, attempts)
    if retire:
        question.status = "retired"
    session.flush()

    return CalibrationFlags(
        question_id=question_id,
        attempts=attempts,
        discrimination=disc,
        recalibrate=needs_recalibration(question),
        dead=dead_distractors(question),
        retire=retire,
    )


def audit_bank(session: Session, course_id: uuid.UUID) -> dict:
    """Audit every active question of a course; return a tally + the flag list."""
    ids = list(
        session.scalars(
            select(Question.id).where(
                Question.course_id == course_id, Question.status == "active"
            )
        )
    )
    flags = [audit_question(session, qid) for qid in ids]
    return {
        "audited": len(flags),
        "recalibrate": sum(1 for f in flags if f.recalibrate),
        "retired": sum(1 for f in flags if f.retire),
        "dead_distractors": sum(1 for f in flags if f.dead),
        "flags": flags,
    }
