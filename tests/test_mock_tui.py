"""Textual Pilot smoke tests: the mock and diagnostic UIs drive the real engines."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select

from atlas.core.models import Attempt, MockExam, SessionRun
from atlas.learn import diagnose, mock
from atlas.learn.mock_cli import DiagnosticApp, MockApp
from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_chapter,
    seed_concept,
    seed_course,
    seed_question,
)


async def test_mock_app_answers_submits_and_reviews():
    db = mem_session()
    course = seed_course(db)
    items = []
    for i in range(3):
        c = seed_concept(db, course, f"C{i}")
        q = seed_question(db, course, c, stem=f"Q{i}", correct_key="A")
        items.append(mock.item_from_question(q))
    user = new_user()
    row = mock.start_mock(
        db, user_id=user, course=course, items=items, predicted_before={}, duration_minutes=180,
        now=datetime.now(UTC),
    )
    app = MockApp(
        items, mock_row=row, db=db, passing_marks=2.0, neg_fraction=0.25,
        duration_minutes=180, now=datetime.now(UTC),
    )
    async with app.run_test() as pilot:
        # Answer all three correctly (A, sure); the last `n` submits.
        for _ in range(3):
            await pilot.press("a")
            await pilot.press("3")
            await pilot.press("n")
        assert app.phase == "done"
        assert app.result is not None
        assert app.result.raw_score == 3.0
        assert app.result.answered == 3
        await pilot.press("v")
        assert app.phase == "review"

    n = db.scalar(select(func.count()).select_from(Attempt).where(Attempt.user_id == user))
    assert n == 3  # one attempt recorded per answered question
    assert float(db.get(MockExam, row.id).raw_score) == 3.0


async def test_diagnostic_app_collects_and_seeds():
    db = mem_session()
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Ch1")
    questions = []
    for i in range(3):
        c = seed_concept(db, course, f"C{i}", chapter=ch, difficulty=(i % 3) + 1)
        questions.append(seed_question(db, course, c, stem=f"Q{i}", correct_key="A", difficulty=(i % 3) + 1))
    user = new_user()

    app = DiagnosticApp(questions, user_id=user, course=course, db=db, now=datetime.now(UTC))
    async with app.run_test() as pilot:
        for _ in range(3):
            await pilot.press("a")
            await pilot.press("3")
            await pilot.press("n")  # last press seeds
        assert app.phase == "done"
        assert app.result is not None
        assert len(app.result.tested) == 3

    # A diagnostic session was recorded and mastery seeded.
    assert diagnose.diagnostic_exists(db, user, course.id) is True
    n_sessions = db.scalar(
        select(func.count()).select_from(SessionRun).where(SessionRun.kind == "diagnostic")
    )
    assert n_sessions == 1
