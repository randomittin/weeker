"""T-25 gate: q_rating init, live Elo hook, recalibrate / dead-distractor / retire.

Simulated attempt streams move q_rating toward the empirical difficulty; a rating
that drifts past 1.5 logits from its authored init flags for recalibration; a
distractor no one picks after enough attempts is dead; and a question that stops
discriminating (below floor at ≥ 20 attempts) is retired.
"""

from __future__ import annotations

from atlas.core.models import Attempt, Question
from atlas.generate import calibrate
from atlas.learn import elo
from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_concept,
    seed_course,
    seed_question,
)


def test_initial_q_rating_by_difficulty():
    assert calibrate.initial_q_rating(1) < calibrate.initial_q_rating(2)
    assert calibrate.initial_q_rating(2) < calibrate.initial_q_rating(3)
    assert calibrate.initial_q_rating(2) == 0.0


def test_live_hook_moves_rating_toward_empirical():
    # A fixed strong learner answering an "average" question correctly every time
    # is empirical evidence the question is easy → its rating must fall.
    theta = 2.0
    q_rating = calibrate.initial_q_rating(2)  # 0.0
    start = q_rating
    for i in range(30):
        res = elo.update(theta, q_rating, attempts=i, q_attempts=i, correct=True, confidence="sure")
        q_rating = calibrate.apply_rating_value(q_rating, res)
    assert q_rating < start  # rating dropped toward the empirical (easy) signal

    # The mirror: always-wrong drives the rating up.
    q_rating = 0.0
    for i in range(30):
        res = elo.update(0.0, q_rating, attempts=i, q_attempts=i, correct=False, confidence="sure")
        q_rating = calibrate.apply_rating_value(q_rating, res)
    assert q_rating > 0.0


def test_apply_rating_sets_question_field():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    q = seed_question(db, course, concept, q_rating=0.0)
    res = elo.update(0.0, 0.0, attempts=0, q_attempts=0, correct=False, confidence="sure")
    calibrate.apply_rating(q, res)
    assert float(q.q_rating) == res.q_rating


def test_needs_recalibration_on_divergence():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    ok = seed_question(db, course, concept, difficulty=2, q_rating=0.5)
    drifted = seed_question(db, course, concept, difficulty=2, q_rating=2.0)  # |2.0-0.0|>1.5
    assert not calibrate.needs_recalibration(ok)
    assert calibrate.needs_recalibration(drifted)


def test_dead_distractor_flagged():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    q = seed_question(db, course, concept)
    # 22 pulls, but option C never chosen → dead distractor.
    q.pull_count = {"A": 10, "B": 8, "D": 4}
    db.flush()
    dead = calibrate.dead_distractors(q)
    assert "C" in dead
    assert "A" not in dead  # A is the correct key, never a distractor

    # Too few pulls → not enough evidence to call anything dead.
    q.pull_count = {"A": 2, "B": 1}
    assert calibrate.dead_distractors(q) == []


def test_discrimination_and_retirement():
    # Correctness tracks ability → high discrimination, keep.
    good = [(2.0, True), (1.5, True), (-1.0, False), (-2.0, False)] * 6
    assert calibrate.compute_discrimination(good) > 0.05
    assert not calibrate.should_retire(calibrate.compute_discrimination(good), attempts=len(good))

    # Correctness independent of ability → ~zero discrimination, retire at ≥20 attempts.
    flat = [(1.0, True), (1.0, False), (-1.0, True), (-1.0, False)] * 6
    disc = calibrate.compute_discrimination(flat)
    assert abs(disc) < 0.05
    assert calibrate.should_retire(disc, attempts=len(flat))
    assert not calibrate.should_retire(disc, attempts=10)  # too few attempts yet


def test_audit_question_sets_discrimination_and_retires():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    q = seed_question(db, course, concept)
    user = new_user()
    # 20 attempts with correctness uncorrelated to ability → non-discriminating.
    for i in range(20):
        theta = 1.0 if i % 2 == 0 else -1.0
        correct = i % 4 in (0, 3)  # independent of theta
        db.add(
            Attempt(
                user_id=user,
                question_id=q.id,
                concept_id=concept.id,
                correct=correct,
                confidence="sure",
                theta_before=theta,
                theta_after=theta,
            )
        )
    db.flush()
    flags = calibrate.audit_question(db, q.id)
    assert flags.attempts == 20
    assert q.discrimination is not None
    assert flags.retire is True
    assert db.get(Question, q.id).status == "retired"

    # A fresh question with no attempts audits cleanly (nothing to retire).
    q2 = seed_question(db, course, concept)
    flags2 = calibrate.audit_question(db, q2.id)
    assert flags2.retire is False
    assert flags2.attempts == 0


def test_audit_bank_summarizes(tmp_path):
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    seed_question(db, course, concept, difficulty=2, q_rating=2.0)  # recalibrate candidate
    summary = calibrate.audit_bank(db, course.id)
    assert summary["audited"] == 1
    assert summary["recalibrate"] == 1
