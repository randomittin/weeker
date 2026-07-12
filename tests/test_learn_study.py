"""T-34 Textual Pilot tests for the study and flashcard TUIs."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select

from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_concept,
    seed_course,
    seed_question,
)
from weeker.core.models import Attempt, Chunk, ConceptChunk, Flashcard
from weeker.learn.study_cli import FlashApp, StudyApp


async def test_full_five_question_session_writes_attempts():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Mutual funds")
    questions = [
        seed_question(db, course, concept, stem=f"Q{i}", correct_key="A", q_rating=0.0)
        for i in range(5)
    ]
    user = new_user()

    app = StudyApp(questions, user_id=user, course_id=course.id, db=db, now=datetime.now(UTC))
    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.press("a")   # choose option A
            await pilot.press("3")   # confidence: sure → grades + shows explanation
            await pilot.press("n")   # advance
        assert app.phase == "done"

    # Exactly five attempts persisted, all correct (A is the key).
    n = db.scalar(select(func.count()).select_from(Attempt).where(Attempt.user_id == user))
    assert n == 5
    assert all(a.correct for a in db.scalars(select(Attempt).where(Attempt.user_id == user)))
    assert len(app.outcomes) == 5


async def test_answer_then_confidence_flow_and_wrong_answer():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Derivatives")
    q = seed_question(db, course, concept, correct_key="A")
    user = new_user()

    app = StudyApp([q], user_id=user, course_id=course.id, db=db)
    async with app.run_test() as pilot:
        await pilot.press("b")             # wrong option
        assert app.pending_key == "B"
        assert app.phase == "answer"       # not committed until confidence given
        await pilot.press("2")             # unsure
        assert app.phase == "explain"
        _, out = app.outcomes[-1]
        assert out.correct is False
        assert out.confidence == "unsure"


async def test_peek_source_toggles_and_renders_chunk():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Valuation")
    q = seed_question(db, course, concept, correct_key="A")
    chunk = Chunk(
        course_id=course.id,
        source_id=course.id,  # standalone fixture; FK to sources not enforced on SQLite here
        content="Net asset value is assets minus liabilities per unit.",
        content_hash="h",
        page_start=12,
        page_end=13,
    )
    db.add(chunk)
    db.flush()
    db.add(ConceptChunk(concept_id=concept.id, chunk_id=chunk.id, cos=0.9))
    db.flush()

    app = StudyApp([q], user_id=new_user(), course_id=course.id, db=db)
    async with app.run_test() as pilot:
        assert app.show_source is False
        await pilot.press("p")
        assert app.show_source is True
        text = app.source_provider(q)
        assert "p.12-13" in text and "Net asset value" in text


async def test_flashcard_drain_reschedules_cards():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "Ratios")
    user = new_user()
    cards = [
        Flashcard(
            user_id=user, concept_id=concept.id, front=f"front{i}", back=f"back{i}", stability=1.0
        )
        for i in range(2)
    ]
    for c in cards:
        db.add(c)
    db.flush()

    app = FlashApp(cards, db=db, now=datetime.now(UTC))
    async with app.run_test() as pilot:
        for _ in range(2):
            await pilot.press("space")  # flip
            await pilot.press("3")      # "good" → correct/sure
        assert app.phase == "done"
        assert app.graded == 2

    for c in db.scalars(select(Flashcard).where(Flashcard.user_id == user)):
        assert c.due_at is not None
        assert float(c.stability) > 1.0  # a good grade grew stability
