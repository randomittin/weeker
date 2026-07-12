"""T-40 gate: timed mock scoring, negative marking, pause/resume, review order.

The headline gate: a scripted 135-answer run produces a ``mock_exams`` row with
the correct raw score and negative-marking deduction against a known answer set.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_concept,
    seed_course,
    seed_question,
)
from weeker.core.models import MockExam
from weeker.learn import mock


def _mk_items(db, course, n, correct_key="A"):
    # One concept per question so per-concept stability growth stays bounded
    # (a real mock spreads across many concepts, never 100 in a row on one).
    items = []
    for i in range(n):
        concept = seed_concept(db, course, f"Concept {i}")
        q = seed_question(db, course, concept, stem=f"Q{i}", correct_key=correct_key, marks=1)
        items.append(mock.item_from_question(q))
    return items


def test_score_item_right_wrong_skip():
    it = mock.MockItem(question_id="q", concept_id="c", marks=2.0, correct_key="A")
    assert mock.score_item(it, mock.MockAnswer("A", "sure")) == 2.0
    assert mock.score_item(it, mock.MockAnswer("B", "sure")) == -0.5  # 0.25 x 2 marks
    assert mock.score_item(it, mock.MockAnswer(None, None)) == 0.0
    assert mock.score_item(it, None) == 0.0


def test_scripted_135_answer_run_scores_and_persists():
    db = mem_session()
    course = seed_course(db)
    items = _mk_items(db, course, 135, correct_key="A")
    user = new_user()

    # Known answer set: 100 correct, 20 confident-wrong, 15 skipped.
    answers: dict = {}
    for i, it in enumerate(items):
        if i < 100:
            answers[it.question_id] = mock.MockAnswer("A", "sure")
        elif i < 120:
            answers[it.question_id] = mock.MockAnswer("B", "unsure")
        # else: skipped (no entry)

    # Hand-computed: 100*1 - 20*(0.25*1) + 0 = 100 - 5 = 95.0 of 135.
    g = mock.grade(items, answers)
    assert g.total_marks == 135.0
    assert g.raw_score == 95.0

    row = mock.start_mock(
        db,
        user_id=user,
        course=course,
        items=items,
        predicted_before={},
        duration_minutes=180,
        now=datetime.now(UTC),
    )
    result = mock.submit_mock(
        db,
        mock=row,
        items=items,
        answers=answers,
        passing_marks=90.0,
        now=datetime.now(UTC),
    )
    assert result.raw_score == 95.0
    assert result.total_marks == 135.0
    assert result.passed is True
    assert result.answered == 120
    assert result.skipped == 15

    persisted = db.get(MockExam, row.id)
    assert float(persisted.raw_score) == 95.0
    assert float(persisted.total_marks) == 135.0
    assert persisted.passed is True
    assert persisted.submitted_at is not None
    assert persisted.config["submitted"] is True


def test_no_feedback_until_submit_via_predicted_snapshot():
    db = mem_session()
    course = seed_course(db)
    items = _mk_items(db, course, 3)
    row = mock.start_mock(
        db,
        user_id=new_user(),
        course=course,
        items=items,
        predicted_before={"expected_marks": 1.5},
        duration_minutes=180,
    )
    # The prediction snapshot is stored before any answer is graded.
    assert row.predicted_before == {"expected_marks": 1.5}
    assert row.raw_score is None  # nothing scored until submit


def test_pause_resume_round_trips_answers_and_position():
    db = mem_session()
    course = seed_course(db)
    items = _mk_items(db, course, 10)
    user = new_user()
    row = mock.start_mock(
        db, user_id=user, course=course, items=items, predicted_before={}, duration_minutes=180
    )

    answers = {
        items[0].question_id: mock.MockAnswer("A", "sure"),
        items[1].question_id: mock.MockAnswer("B", "guessing"),
    }
    mock.save_progress(db, row, answers=answers, index=2, elapsed_seconds=42.0)

    resumed = mock.resume_mock(db, row.id)
    assert resumed.index == 2
    assert resumed.elapsed_seconds == 42.0
    assert len(resumed.items) == 10
    assert resumed.items[0].question_id == items[0].question_id
    assert resumed.items[0].correct_key == "A"
    assert resumed.answers[items[0].question_id].chosen_key == "A"
    assert resumed.answers[items[1].question_id].confidence == "guessing"

    # Resume and finish: answer the rest correctly, submit.
    for it in resumed.items:
        resumed.answers.setdefault(it.question_id, mock.MockAnswer("A", "sure"))
    result = mock.submit_mock(
        db,
        mock=resumed.mock,
        items=resumed.items,
        answers=resumed.answers,
        passing_marks=5.0,
    )
    # 9 correct (A) + 1 wrong (B, 0.25 deduction) = 9 - 0.25 = 8.75 of 10.
    assert result.raw_score == 8.75


def test_review_orders_confident_wrong_first_caselets_grouped():
    # Standalone items s0..s2 and one caselet group (two questions).
    s0 = mock.MockItem("s0", "c", 1, "A")
    s1 = mock.MockItem("s1", "c", 1, "A")
    s2 = mock.MockItem("s2", "c", 1, "A")
    cg = "case-1"
    q1 = mock.MockItem("q1", "c", 1, "A", section=mock.CASELET, case_group_id=cg, case_position=1)
    q2 = mock.MockItem("q2", "c", 1, "A", section=mock.CASELET, case_group_id=cg, case_position=2)
    items = [s0, q1, s1, q2, s2]

    answers = {
        "s0": mock.MockAnswer("A", "sure"),  # correct
        "s1": mock.MockAnswer("B", "unsure"),  # plain wrong
        "s2": mock.MockAnswer("B", "sure"),  # confident-wrong
        "q1": mock.MockAnswer("B", "sure"),  # confident-wrong (in caselet)
        "q2": mock.MockAnswer("A", "sure"),  # correct (in caselet)
    }
    order = mock.review_order(items, answers)
    ids = [it.question_id for it in order]

    # Caselet group floats to rank 0 (its q1 is confident-wrong) and stays contiguous,
    # in case_position order; s2 (confident-wrong) next; s1 (wrong); s0 (correct) last.
    assert ids[0:2] == ["q1", "q2"]  # caselet grouped, position order
    assert ids[2] == "s2"  # confident-wrong standalone
    assert ids[3] == "s1"  # plain wrong
    assert ids[4] == "s0"  # correct last
