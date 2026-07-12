"""Regression: scheduler must not choke on SQLite's tz-naive timestamps.

On SQLite ``DateTime(timezone=True)`` round-trips as a *naive* datetime, so an
aware ``now`` minus a persisted ``last_seen`` / ``max(created_at)`` raises
``TypeError: can't subtract offset-naive and offset-aware`` unless the loader
normalizes. ``load_question_states`` computes ``now - max(Attempt.created_at)``
and previously used the raw (naive) value — this drives that path with a real
prior attempt and asserts no crash.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tests.test_learn_fixtures import (
    new_user,
    seed_chapter,
    seed_concept,
    seed_course,
    seed_question,
)
from weeker.core.models import Attempt, Base
from weeker.learn import scheduler


def _mem():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def test_next_session_survives_naive_prior_attempt_timestamp():
    db = _mem()
    user = new_user()
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Funds")
    concept = seed_concept(db, course, "NAV", chapter=ch)
    q = seed_question(db, course, concept)

    # A prior attempt written WITHOUT an explicit tz — SQLite stores/returns naive.
    db.add(
        Attempt(
            user_id=user,
            question_id=q.id,
            concept_id=concept.id,
            correct=True,
            confidence="sure",
            chosen_key="A",
            theta_before=0.0,
            theta_after=0.1,
            created_at=datetime(2026, 1, 1, 12, 0, 0),  # naive, as SQLite yields
        )
    )
    db.flush()

    now = datetime.now(UTC)  # aware
    # Would raise TypeError (naive vs aware) before the _aware() normalization.
    qstates = scheduler.load_question_states(db, user, course.id, [concept.id], now)
    assert concept.id in qstates
    (state,) = qstates[concept.id]
    assert state.days_since_attempt is not None and state.days_since_attempt > 0

    # And the full composer runs end-to-end a day later without error.
    session2 = scheduler.next_session(
        db, user, course.id, now + timedelta(days=1), random.Random(0)
    )
    assert isinstance(session2, list)
