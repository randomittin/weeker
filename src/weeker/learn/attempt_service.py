"""Transactional attempt recording (013 §2, §6 misconception flow, T-31).

``record_attempt`` is the single write path for a graded answer. In one
transaction it: appends an :class:`Attempt` row carrying ``theta_before/after``,
upserts the learner's :class:`Mastery` (ability + stability + due date), updates
the question's live ``q_rating`` and per-option ``pull_count``, runs the
misconception flow (flag → tag → remediation flashcard on a confident wrong
answer; clear after two confident-correct answers), and emits an analytics event.

Pass ``db=<Session>`` to enlist in a caller-managed transaction (nothing is
committed — the caller owns commit/rollback); omit it to run inside a fresh
``session_scope`` that commits on success.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from weeker.core.db import session_scope
from weeker.core.models import (
    AnalyticsEvent,
    Attempt,
    Flashcard,
    Mastery,
    Misconception,
    MisconceptionFlag,
    Question,
)
from weeker.learn import elo, retention
from weeker.learn.elo import Confidence

# Confident-correct answers needed to clear an open misconception flag.
CLEAR_STREAK = 2


@dataclass(frozen=True)
class AttemptOutcome:
    """What one recorded attempt changed — returned to callers and the TUI."""

    attempt_id: object
    concept_id: object
    correct: bool
    confidence: str
    theta_before: float
    theta_after: float
    mastery_before: float
    mastery_after: float
    stability_before: float
    stability_after: float
    q_rating_before: float
    q_rating_after: float
    due_at: datetime
    misconception_fired: bool
    misconception_cleared: bool


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


def _option_text(question: Question, key: str | None) -> str:
    if key is None:
        return ""
    for opt in question.options or []:
        if isinstance(opt, dict) and opt.get("key") == key:
            return str(opt.get("text", ""))
    return ""


def _chosen_option(question: Question, key: str | None) -> dict:
    for opt in question.options or []:
        if isinstance(opt, dict) and opt.get("key") == key:
            return opt
    return {}


def _resolve_misconception(
    db: Session, question: Question, chosen_key: str | None
) -> object | None:
    """Map the chosen distractor to a :class:`Misconception`, creating it if the
    option carries a ``misconception`` label. Returns the misconception id or None."""
    label = _chosen_option(question, chosen_key).get("misconception")
    if not label:
        return None
    existing = db.scalars(
        select(Misconception).where(
            Misconception.concept_id == question.concept_id,
            Misconception.label == label,
        )
    ).first()
    if existing is not None:
        return existing.id
    misc = Misconception(concept_id=question.concept_id, label=str(label))
    db.add(misc)
    db.flush()
    return misc.id


def _upsert_flashcard(
    db: Session,
    user_id: str,
    concept_id: object,
    question: Question,
    misconception_id: object | None,
    now: datetime,
    stability: float,
) -> None:
    """Insert or refresh the remediation flashcard for this concept/misconception."""
    front = question.stem
    back = _option_text(question, question.correct_key)
    if question.explanation:
        back = f"{back} — {question.explanation}" if back else question.explanation
    card = db.scalars(
        select(Flashcard).where(
            Flashcard.user_id == user_id,
            Flashcard.concept_id == concept_id,
            Flashcard.misconception_id == misconception_id,
        )
    ).first()
    due = now + timedelta(days=retention.days_until_due(stability))
    if card is None:
        db.add(
            Flashcard(
                user_id=user_id,
                concept_id=concept_id,
                front=front,
                back=back,
                misconception_id=misconception_id,
                stability=stability,
                due_at=due,
            )
        )
    else:
        card.front = front
        card.back = back
        card.stability = stability
        card.due_at = due


def _record(
    db: Session,
    *,
    user_id: str,
    question_id: object,
    confidence: Confidence,
    chosen_key: str | None,
    correct: bool | None,
    session_id: object | None,
    now: datetime,
) -> AttemptOutcome:
    question = db.get(Question, question_id)
    if question is None:
        raise ValueError(f"question {question_id} not found")
    concept_id = question.concept_id

    if chosen_key is not None:
        correct = chosen_key == question.correct_key
    elif correct is None:
        raise ValueError("record_attempt requires either chosen_key or correct")

    mastery_row = db.scalars(
        select(Mastery).where(
            Mastery.user_id == user_id, Mastery.concept_id == concept_id
        )
    ).first()
    if mastery_row is None:
        mastery_row = Mastery(user_id=user_id, concept_id=concept_id, theta=0.0, stability=1.0)
        db.add(mastery_row)
        db.flush()

    attempts_prior = db.scalar(
        select(func.count())
        .select_from(Attempt)
        .where(Attempt.user_id == user_id, Attempt.concept_id == concept_id)
    ) or 0
    q_attempts_prior = db.scalar(
        select(func.count()).select_from(Attempt).where(Attempt.question_id == question_id)
    ) or 0

    theta_before = float(mastery_row.theta)
    q_rating_before = float(question.q_rating)
    stability_before = float(mastery_row.stability)

    res = elo.update(
        theta_before, q_rating_before, attempts_prior, q_attempts_prior, correct, confidence
    )
    mastery_before = elo.mastery(theta_before)
    mastery_after = elo.mastery(res.theta)
    stability_after = retention.stability_update(
        stability_before, mastery_after, correct, confidence
    )
    due_at = now + timedelta(days=retention.days_until_due(stability_after))

    attempt = Attempt(
        user_id=user_id,
        question_id=question_id,
        concept_id=concept_id,
        session_id=session_id,
        correct=correct,
        confidence=confidence,
        chosen_key=chosen_key,
        theta_before=theta_before,
        theta_after=res.theta,
        created_at=now,
    )
    db.add(attempt)

    mastery_row.theta = res.theta
    mastery_row.stability = stability_after
    mastery_row.last_seen = now
    mastery_row.due_at = due_at

    question.q_rating = res.q_rating
    if chosen_key is not None:
        pulls = dict(question.pull_count or {})
        pulls[chosen_key] = int(pulls.get(chosen_key, 0)) + 1
        question.pull_count = pulls

    fired, cleared = _run_misconception_flow(
        db,
        user_id=user_id,
        concept_id=concept_id,
        question=question,
        chosen_key=chosen_key,
        correct=correct,
        confidence=confidence,
        mastery_row=mastery_row,
        now=now,
        stability=stability_after,
    )

    db.add(
        AnalyticsEvent(
            user_id=user_id,
            kind="attempt",
            payload={
                "question_id": str(question_id),
                "concept_id": str(concept_id),
                "correct": correct,
                "confidence": confidence,
                "theta_before": theta_before,
                "theta_after": res.theta,
                "misconception_fired": fired,
                "misconception_cleared": cleared,
            },
        )
    )
    db.flush()

    return AttemptOutcome(
        attempt_id=attempt.id,
        concept_id=concept_id,
        correct=correct,
        confidence=confidence,
        theta_before=theta_before,
        theta_after=res.theta,
        mastery_before=mastery_before,
        mastery_after=mastery_after,
        stability_before=stability_before,
        stability_after=stability_after,
        q_rating_before=q_rating_before,
        q_rating_after=res.q_rating,
        due_at=due_at,
        misconception_fired=fired,
        misconception_cleared=cleared,
    )


def _run_misconception_flow(
    db: Session,
    *,
    user_id: str,
    concept_id: object,
    question: Question,
    chosen_key: str | None,
    correct: bool,
    confidence: str,
    mastery_row: Mastery,
    now: datetime,
    stability: float,
) -> tuple[bool, bool]:
    """Steps 1–3 (flag/tag/flashcard) and the two-confident-correct clear rule.

    Returns ``(fired, cleared)``.
    """
    open_flags = db.scalars(
        select(MisconceptionFlag).where(
            MisconceptionFlag.user_id == user_id,
            MisconceptionFlag.concept_id == concept_id,
            MisconceptionFlag.status == "open",
        )
    ).all()

    fired = False
    cleared = False

    if not correct and confidence == "sure":
        # Step 1 — flag (reuse an open flag rather than stacking duplicates).
        flag = open_flags[0] if open_flags else None
        if flag is None:
            flag = MisconceptionFlag(
                user_id=user_id, concept_id=concept_id, status="open"
            )
            db.add(flag)
            db.flush()
            open_flags = [flag]
        flag.confident_correct_streak = 0
        mastery_row.misconception_pending = True
        # Step 2 — tag with the specific misconception when the distractor names one.
        misc_id = _resolve_misconception(db, question, chosen_key)
        if misc_id is not None:
            flag.misconception_id = misc_id
        # Step 3 — remediation flashcard.
        _upsert_flashcard(db, user_id, concept_id, question, misc_id, now, stability)
        fired = True
    elif correct and confidence == "sure":
        for flag in open_flags:
            flag.confident_correct_streak = int(flag.confident_correct_streak) + 1
            if flag.confident_correct_streak >= CLEAR_STREAK:
                flag.status = "cleared"
                flag.cleared_at = now
                cleared = True
        if all(f.status != "open" for f in open_flags):
            mastery_row.misconception_pending = False
    elif not correct:
        # A non-confident lapse resets progress toward clearing.
        for flag in open_flags:
            flag.confident_correct_streak = 0

    return fired, cleared


def record_attempt(
    *,
    user_id: str,
    question_id: object,
    confidence: Confidence,
    chosen_key: str | None = None,
    correct: bool | None = None,
    session_id: object | None = None,
    now: datetime | None = None,
    db: Session | None = None,
) -> AttemptOutcome:
    """Record one graded attempt. See module docstring for the transaction rules."""
    when = _now(now)
    if db is not None:
        return _record(
            db,
            user_id=user_id,
            question_id=question_id,
            confidence=confidence,
            chosen_key=chosen_key,
            correct=correct,
            session_id=session_id,
            now=when,
        )
    with session_scope() as scoped:
        return _record(
            scoped,
            user_id=user_id,
            question_id=question_id,
            confidence=confidence,
            chosen_key=chosen_key,
            correct=correct,
            session_id=session_id,
            now=when,
        )
