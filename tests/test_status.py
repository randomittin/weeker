"""T-41 gate: golden-output status dashboard on a fixed fixture state.

The fixture is engineered to be fully deterministic: coverage is below the
prediction threshold (so the band is a stable "suppressed" line, not a bootstrap
float), attempts span three distinct days (so the realized-calibration table is
shown with exact fractions), and all timestamps are pinned.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from atlas.core.models import Attempt, Flashcard, Mastery, Question
from atlas.learn import status
from tests.test_learn_fixtures import (
    mem_session,
    seed_chapter,
    seed_concept,
    seed_course,
    seed_question,
)

NOW = datetime(2026, 7, 12, 12, 0, 0, tzinfo=UTC)
USER = "learner-1"

GOLDEN = """Status — NISM XA

Heatmap (mastery by chapter/concept):
  Ch1  [mean 0.43]
    - A1: mastery 0.55 (theta +0.50, 3 attempts, due +0.2d) !misconception
    - A2: mastery 0.31 (theta -0.50, 0 attempts, due never)
  Ch2  [mean 0.55]
    - B1: mastery 0.67 (theta +1.00, 0 attempts, due +0.2d)
    - B2: mastery 0.43 (theta +0.00, 0 attempts, due never)

Due: 2 concepts, 1 flashcards | Misconception flags: 1

Prediction: suppressed (coverage 25% < threshold — practise more)

Tonight's refill plan (weak-concept quotas):
  - A1: 10
  - A2: 10
  - B1: 10
  - B2: 10

Realized calibration:
  - sure: 0/1 = 0% -> attempt only if you can eliminate an option
  - unsure: 2/2 = 100% -> always attempt"""


def _q_id(db, concept):
    return db.scalars(select(Question).where(Question.concept_id == concept.id)).first().id


def _build_fixture(db):
    course = seed_course(db, title="NISM XA")
    ch1 = seed_chapter(db, course, 1, "Ch1")
    ch2 = seed_chapter(db, course, 2, "Ch2")
    a1 = seed_concept(db, course, "A1", chapter=ch1)
    a2 = seed_concept(db, course, "A2", chapter=ch1)
    b1 = seed_concept(db, course, "B1", chapter=ch2)
    b2 = seed_concept(db, course, "B2", chapter=ch2)
    for c in (a1, a2, b1, b2):
        seed_question(db, course, c, stem=f"{c.title}-q")

    db.add(
        Mastery(
            user_id=USER, concept_id=a1.id, theta=0.5, stability=1.0,
            last_seen=NOW, due_at=NOW - timedelta(days=1), misconception_pending=True,
        )
    )
    db.add(Mastery(user_id=USER, concept_id=a2.id, theta=-0.5, stability=1.0))
    db.add(
        Mastery(
            user_id=USER, concept_id=b1.id, theta=1.0, stability=1.0,
            last_seen=NOW, due_at=NOW - timedelta(days=1),
        )
    )
    db.flush()

    qid = _q_id(db, a1)
    for ts, conf, correct in [
        (NOW - timedelta(days=2), "unsure", True),
        (NOW - timedelta(days=1), "unsure", True),
        (NOW, "sure", False),
    ]:
        db.add(
            Attempt(
                user_id=USER, question_id=qid, concept_id=a1.id, correct=correct,
                confidence=conf, theta_before=0.0, theta_after=0.0, created_at=ts,
            )
        )
    db.add(
        Flashcard(
            user_id=USER, concept_id=a1.id, front="f", back="b",
            stability=1.0, due_at=NOW - timedelta(days=1),
        )
    )
    db.flush()
    return course


def test_status_golden_output():
    db = mem_session()
    course = _build_fixture(db)
    report = status.build_status(db, user_id=USER, course=course, now=NOW)
    assert status.render_status(report) == GOLDEN


def test_calibration_hidden_before_day_three():
    db = mem_session()
    course = seed_course(db, title="NISM XA")
    ch = seed_chapter(db, course, 1, "Ch1")
    c = seed_concept(db, course, "A1", chapter=ch)
    seed_question(db, course, c, stem="q")
    qid = _q_id(db, c)
    # Two attempts, both today → only one distinct day → not calibration-ready.
    for conf in ("sure", "unsure"):
        db.add(
            Attempt(
                user_id=USER, question_id=qid, concept_id=c.id, correct=True,
                confidence=conf, theta_before=0.0, theta_after=0.0, created_at=NOW,
            )
        )
    db.flush()
    report = status.build_status(db, user_id=USER, course=course, now=NOW)
    assert report.calibration_ready is False
    assert report.calibration == []
    assert "available from day 3" in status.render_status(report)
