"""T-42 gate: full-pattern mock composer, marks-weighted scoring with 2-mark
negatives, per-section prediction, and the calibration + skip-policy report.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_chapter,
    seed_concept,
    seed_course,
    seed_question,
)
from weeker.core.models import Attempt, BlueprintWeight, CaseGroup, MockExam, Question
from weeker.learn import full, mock


def _seed_caselet(db, course, concepts, *, marks_each, position_key="A"):
    cg = CaseGroup(
        course_id=course.id,
        scenario="An investor case " * 20,
        concept_ids=[str(c.id) for c in concepts],
        status="active",
        generator_model="test",
    )
    db.add(cg)
    db.flush()
    qs = []
    for pos in range(1, 6):
        concept = concepts[(pos - 1) % len(concepts)]
        q = Question(
            course_id=course.id,
            concept_id=concept.id,
            case_group_id=cg.id,
            stem=f"Caselet {cg.id} q{pos}",
            options=[
                {"key": "A", "text": "right"},
                {"key": "B", "text": "wrong"},
                {"key": "C", "text": "wrong"},
                {"key": "D", "text": "wrong"},
            ],
            correct_key=position_key,
            explanation="because",
            difficulty=2,
            marks=marks_each,
            case_position=pos,
        )
        db.add(q)
        qs.append(q)
    db.flush()
    return cg, qs


def test_grade_includes_two_mark_negatives():
    # 3 one-mark standalone + a 2-mark caselet question set — hand-computed score.
    one = [mock.MockItem(f"s{i}", "c", 1, "A") for i in range(3)]
    two = [mock.MockItem(f"t{i}", "c", 2, "A") for i in range(2)]
    items = one + two
    answers = {
        "s0": mock.MockAnswer("A", "sure"),  # +1
        "s1": mock.MockAnswer("A", "unsure"),  # +1
        "s2": mock.MockAnswer("B", "sure"),  # -0.25
        "t0": mock.MockAnswer("A", "sure"),  # +2
        "t1": mock.MockAnswer("B", "guessing"),  # -0.5  (0.25 x 2 marks)
    }
    g = mock.grade(items, answers)
    # 1 + 1 - 0.25 + 2 - 0.5 = 3.25 of total 3*1 + 2*2 = 7.
    assert g.total_marks == 7.0
    assert g.raw_score == 3.25


def test_full_pattern_e2e_score_with_two_mark_caselet():
    db = mem_session()
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Ch1")
    concepts = [seed_concept(db, course, f"C{i}", chapter=ch) for i in range(3)]
    standalone = [
        mock.item_from_question(seed_question(db, course, concepts[i % 3], stem=f"S{i}"))
        for i in range(3)
    ]
    _cg, cqs = _seed_caselet(db, course, concepts, marks_each=2)
    caselet_items = [mock.item_from_question(q, mock.CASELET) for q in cqs]
    items = standalone + caselet_items
    user = new_user()

    answers = {}
    for it in standalone:
        answers[it.question_id] = mock.MockAnswer("A", "sure")  # 3 x +1
    # Caselet: 4 correct (+2 each = 8), 1 confident-wrong (-0.5).
    for i, it in enumerate(caselet_items):
        answers[it.question_id] = mock.MockAnswer("A" if i < 4 else "B", "sure")

    row = mock.start_mock(
        db, user_id=user, course=course, items=items, predicted_before={}, duration_minutes=180
    )
    result = mock.submit_mock(
        db, mock=row, items=items, answers=answers, passing_marks=5.0
    )
    # 3 + (4*2 - 0.5) = 3 + 7.5 = 10.5 of total 3*1 + 5*2 = 13.
    assert result.total_marks == 13.0
    assert result.raw_score == 10.5
    assert float(db.get(MockExam, row.id).raw_score) == 10.5


def test_compose_caselets_picks_weak_coverage_and_excludes_recent():
    db = mem_session()
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Ch1")
    # Weak concepts (low theta) vs strong (high theta).
    weak = [seed_concept(db, course, f"W{i}", chapter=ch) for i in range(3)]
    strong = [seed_concept(db, course, f"S{i}", chapter=ch) for i in range(3)]
    user = new_user()
    from tests.test_learn_fixtures import seed_mastery

    for c in strong:
        seed_mastery(db, user, c, theta=4.0)  # high mastery → low weak coverage
    for c in weak:
        seed_mastery(db, user, c, theta=-4.0)  # low mastery → high weak coverage

    cg_weak, _ = _seed_caselet(db, course, weak, marks_each=1)
    cg_strong, _ = _seed_caselet(db, course, strong, marks_each=1)
    now = datetime.now(UTC)

    items = full.compose_caselet_items(db, user, course, now, one_mark=1, two_mark=0)
    # Highest weak coverage wins → the weak-concept caselet.
    assert {it.case_group_id for it in items} == {cg_weak.id}

    # Mark the weak caselet as recently attempted → it is excluded, strong picked.
    from sqlalchemy import select

    q_weak = db.scalars(
        select(Question).where(Question.case_group_id == cg_weak.id)
    ).first()
    db.add(
        Attempt(
            user_id=user,
            question_id=q_weak.id,
            concept_id=weak[0].id,
            correct=True,
            confidence="sure",
            theta_before=0.0,
            theta_after=0.0,
            created_at=now - timedelta(days=1),
        )
    )
    db.flush()
    items2 = full.compose_caselet_items(db, user, course, now, one_mark=1, two_mark=0)
    assert {it.case_group_id for it in items2} == {cg_strong.id}


def test_compose_full_items_shape_and_sections():
    db = mem_session()
    course = seed_course(db)
    ch1 = seed_chapter(db, course, 1, "Ch1")
    ch2 = seed_chapter(db, course, 2, "Ch2")
    db.add(BlueprintWeight(course_id=course.id, chapter="Ch1", weight=0.5))
    db.add(BlueprintWeight(course_id=course.id, chapter="Ch2", weight=0.5))
    c1 = [seed_concept(db, course, f"A{i}", chapter=ch1) for i in range(3)]
    c2 = [seed_concept(db, course, f"B{i}", chapter=ch2) for i in range(3)]
    for c in c1 + c2:
        for j in range(2):
            seed_question(db, course, c, stem=f"{c.title}-q{j}")
    _seed_caselet(db, course, c1, marks_each=1)
    _seed_caselet(db, course, c2, marks_each=1)
    _seed_caselet(db, course, c1, marks_each=2)
    user = new_user()
    now = datetime.now(UTC)

    items = full.compose_full_items(
        db, user, course, now, random.Random(1), standalone=6, one_mark=2, two_mark=1
    )
    standalone = [it for it in items if it.section == mock.STANDALONE]
    caselet = [it for it in items if it.section == mock.CASELET]
    assert len(standalone) == 6
    assert len(caselet) == 3 * 5  # (2 one-mark + 1 two-mark) caselets x 5 questions

    sections, overall = full.predict_sections(
        db, user, items, passing_marks=90.0, rng=random.Random(2)
    )
    names = {s.section for s in sections}
    assert full.SECTION_STANDALONE in names
    assert full.SECTION_CASELET_1 in names
    assert full.SECTION_CASELET_2 in names
    assert 0.0 <= overall.p_pass <= 1.0


def test_skip_policy_and_calibration():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "C")
    user = new_user()
    q = seed_question(db, course, concept)
    # 3 'unsure' attempts, 2 correct → 66% > break-even 0.2 → "always attempt".
    for i in range(3):
        db.add(
            Attempt(
                user_id=user,
                question_id=q.id,
                concept_id=concept.id,
                correct=(i < 2),
                confidence="unsure",
                theta_before=0.0,
                theta_after=0.0,
                created_at=datetime.now(UTC),
            )
        )
    db.flush()

    lines = full.skip_policy_lines(0.25)
    assert any("Break-even p = 0.20" in ln for ln in lines)

    rows = full.realized_calibration(db, user)
    unsure = next(r for r in rows if r.confidence == "unsure")
    assert unsure.n == 3 and unsure.correct == 2
    assert unsure.recommendation == "always attempt"
