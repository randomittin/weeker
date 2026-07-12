"""T-35 tests — synthetic adaptive verification and mastery replay (M4 gate)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from atlas.core.models import Mastery
from atlas.learn.attempt_service import record_attempt
from atlas.learn.replay import replay
from atlas.learn.verify_adaptive import verify_adaptive
from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_concept,
    seed_course,
    seed_question,
)


def test_verify_adaptive_recovers_ability():
    report = verify_adaptive()
    assert report.ok, report.checks
    assert report.correlation > 0.7
    assert report.n_attempts == 200


def test_verify_adaptive_deterministic_for_seed():
    a = verify_adaptive(seed=123)
    b = verify_adaptive(seed=123)
    assert a.correlation == b.correlation


def test_replay_reproduces_stored_mastery():
    db = mem_session()
    course = seed_course(db)
    concepts = [seed_concept(db, course, f"C{i}") for i in range(3)]
    questions = [
        seed_question(db, course, c, correct_key="A", q_rating=0.0) for c in concepts
    ]
    user = new_user()

    base = datetime(2026, 1, 1, tzinfo=UTC)
    rng_keys = ["A", "B", "A", "C", "A", "D", "A", "A"]
    confs = ["sure", "unsure", "guessing"]
    step = 0
    for round_ in range(8):
        for qi, q in enumerate(questions):
            now = base + timedelta(seconds=step)
            step += 1
            record_attempt(
                user_id=user,
                question_id=q.id,
                chosen_key=rng_keys[(round_ + qi) % len(rng_keys)],
                confidence=confs[(round_ + qi) % 3],
                now=now,
                db=db,
            )
    db.commit()

    diff = replay(db=db, user_id=user)
    assert diff.ok, diff.mismatches
    assert diff.concepts_checked == 3
    assert diff.max_theta_diff < 1e-4
    assert diff.max_stability_diff < 1e-4


def test_replay_detects_tampered_mastery():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Tamper")
    q = seed_question(db, course, concept, correct_key="A")
    user = new_user()

    base = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(4):
        record_attempt(
            user_id=user, question_id=q.id, chosen_key="A", confidence="sure",
            now=base + timedelta(seconds=i), db=db,
        )
    db.commit()

    m = db.scalars(select(Mastery).where(Mastery.user_id == user)).first()
    m.theta = float(m.theta) + 1.0  # corrupt the stored ability
    db.commit()

    diff = replay(db=db, user_id=user)
    assert not diff.ok
    assert diff.max_theta_diff > 0.5


def test_replay_empty_db_is_ok():
    db = mem_session()
    diff = replay(db=db)
    assert diff.ok
    assert diff.concepts_checked == 0
