"""T-31 tests for the transactional attempt service."""

from __future__ import annotations

from sqlalchemy import func, select

from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_concept,
    seed_course,
    seed_question,
)
from weeker.core.models import (
    AnalyticsEvent,
    Attempt,
    Flashcard,
    Mastery,
    MisconceptionFlag,
)
from weeker.learn.attempt_service import record_attempt


def _counts(db):
    return {
        "attempts": db.scalar(select(func.count()).select_from(Attempt)),
        "mastery": db.scalar(select(func.count()).select_from(Mastery)),
        "flags": db.scalar(select(func.count()).select_from(MisconceptionFlag)),
        "flashcards": db.scalar(select(func.count()).select_from(Flashcard)),
        "events": db.scalar(select(func.count()).select_from(AnalyticsEvent)),
    }


def test_one_wrong_sure_attempt_mutates_expected_rows():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Time value of money")
    q = seed_question(db, course, concept, correct_key="A", q_rating=0.0)
    user = new_user()

    before = _counts(db)
    out = record_attempt(
        user_id=user, question_id=q.id, chosen_key="B", confidence="sure", db=db
    )
    after = _counts(db)

    assert not out.correct
    # Exactly one new row in each of these tables; a fresh mastery row; a flag +
    # a remediation flashcard because it was a confident wrong answer.
    assert after["attempts"] - before["attempts"] == 1
    assert after["mastery"] - before["mastery"] == 1
    assert after["flags"] - before["flags"] == 1
    assert after["flashcards"] - before["flashcards"] == 1
    assert after["events"] - before["events"] == 1
    assert out.misconception_fired is True

    # Question mutated in place (no new question row) — pull_count + q_rating.
    db.refresh(q)
    assert q.pull_count == {"B": 1}
    # Wrong answer lowered ability and pushed the question rating up.
    assert out.theta_after < out.theta_before
    assert out.q_rating_after > out.q_rating_before

    m = db.scalars(select(Mastery).where(Mastery.user_id == user)).first()
    assert m.misconception_pending is True
    assert m.last_seen is not None
    assert m.due_at is not None


def test_misconception_fires_only_on_wrong_and_sure():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Compounding")
    q = seed_question(db, course, concept, correct_key="A")
    user = new_user()

    # wrong + unsure → no flag, no flashcard, pending stays False.
    out = record_attempt(
        user_id=user, question_id=q.id, chosen_key="C", confidence="unsure", db=db
    )
    assert out.misconception_fired is False
    assert db.scalar(select(func.count()).select_from(MisconceptionFlag)) == 0
    assert db.scalar(select(func.count()).select_from(Flashcard)) == 0
    m = db.scalars(select(Mastery).where(Mastery.user_id == user)).first()
    assert m.misconception_pending is False

    # correct + sure → still no flag.
    out = record_attempt(
        user_id=user, question_id=q.id, chosen_key="A", confidence="sure", db=db
    )
    assert out.misconception_fired is False
    assert db.scalar(select(func.count()).select_from(MisconceptionFlag)) == 0


def test_flag_clears_after_two_confident_correct():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Suitability")
    q = seed_question(db, course, concept, correct_key="A", q_rating=0.0)
    user = new_user()

    # Fire the misconception.
    record_attempt(user_id=user, question_id=q.id, chosen_key="B", confidence="sure", db=db)
    flag = db.scalars(select(MisconceptionFlag)).first()
    assert flag.status == "open"

    # First confident-correct: streak 1, still open.
    out = record_attempt(user_id=user, question_id=q.id, chosen_key="A", confidence="sure", db=db)
    db.refresh(flag)
    assert flag.status == "open"
    assert flag.confident_correct_streak == 1
    assert out.misconception_cleared is False

    # Second confident-correct: cleared, pending flips off.
    out = record_attempt(user_id=user, question_id=q.id, chosen_key="A", confidence="sure", db=db)
    db.refresh(flag)
    assert flag.status == "cleared"
    assert flag.cleared_at is not None
    assert out.misconception_cleared is True
    m = db.scalars(select(Mastery).where(Mastery.user_id == user)).first()
    assert m.misconception_pending is False


def test_confident_correct_streak_resets_on_lapse():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Taxation")
    q = seed_question(db, course, concept, correct_key="A")
    user = new_user()

    record_attempt(user_id=user, question_id=q.id, chosen_key="B", confidence="sure", db=db)
    record_attempt(user_id=user, question_id=q.id, chosen_key="A", confidence="sure", db=db)
    flag = db.scalars(select(MisconceptionFlag)).first()
    db.refresh(flag)
    assert flag.confident_correct_streak == 1

    # A wrong answer (even unsure) resets the streak.
    record_attempt(user_id=user, question_id=q.id, chosen_key="D", confidence="unsure", db=db)
    db.refresh(flag)
    assert flag.confident_correct_streak == 0
    assert flag.status == "open"


def test_distractor_misconception_is_tagged_and_reused():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Asset allocation")
    q = seed_question(
        db, course, concept, correct_key="A", distractor_misconception="confuses-B-with-A"
    )
    user = new_user()

    record_attempt(user_id=user, question_id=q.id, chosen_key="B", confidence="sure", db=db)
    flag = db.scalars(select(MisconceptionFlag)).first()
    assert flag.misconception_id is not None
    card = db.scalars(select(Flashcard)).first()
    assert card.misconception_id == flag.misconception_id
    # Only one misconception row created even if the same distractor recurs.
    from weeker.core.models import Misconception

    record_attempt(user_id=user, question_id=q.id, chosen_key="B", confidence="sure", db=db)
    assert db.scalar(select(func.count()).select_from(Misconception)) == 1
    # Flashcard is upserted, not duplicated.
    assert db.scalar(select(func.count()).select_from(Flashcard)) == 1


def test_chosen_key_and_explicit_correct_agree():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Risk")
    q = seed_question(db, course, concept, correct_key="A")
    user = new_user()

    out = record_attempt(user_id=user, question_id=q.id, correct=True, confidence="sure", db=db)
    assert out.correct is True
    assert out.theta_after > out.theta_before
