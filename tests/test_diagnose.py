"""T-36 gate: diagnostic selection (coverage-greedy ≥30 concepts), chapter-mate
seeding shrinks toward 0, and idempotent reseeding (refuses without --force).
"""

from __future__ import annotations

import random

import pytest
from sqlalchemy import select

from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_chapter,
    seed_concept,
    seed_course,
    seed_question,
)
from weeker.core.models import Mastery
from weeker.learn import diagnose


def _seed_12_chapters(db, course):
    """12 chapters × 3 concepts, one active question per concept at difficulty 1/2/3."""
    for ch_i in range(12):
        ch = seed_chapter(db, course, ch_i + 1, f"Chapter {ch_i + 1}")
        for tier in (1, 2, 3):
            c = seed_concept(db, course, f"c{ch_i}-{tier}", chapter=ch, difficulty=tier)
            seed_question(db, course, c, stem=f"q{ch_i}-{tier}", difficulty=tier)


def test_coverage_greedy_picks_at_least_30_distinct_concepts():
    db = mem_session()
    course = seed_course(db)
    _seed_12_chapters(db, course)
    selected = diagnose.select_diagnostic_questions(db, course.id, random.Random(0))
    assert len(selected) == 36  # 3 per chapter × 12
    distinct_concepts = {q.concept_id for q in selected}
    assert len(distinct_concepts) >= 30


def test_chapter_mate_seeding_shrinks_toward_zero():
    db = mem_session()
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Ch1")
    tested = seed_concept(db, course, "tested", chapter=ch)
    untested = seed_concept(db, course, "untested", chapter=ch)
    q = seed_question(db, course, tested, correct_key="A", q_rating=0.0)
    user = new_user()

    # A confident-correct answer drives the tested theta positive.
    graded = [diagnose.Graded(question=q, chosen_key="A", confidence="sure")]
    result = diagnose.seed_diagnostic(
        db, user_id=user, course=course, graded=graded, build_status=False
    )

    tested_theta = result.tested[tested.id]
    assert tested_theta > 0.0
    seed = result.seeded_untested[untested.id]
    # Untested chapter-mate = 0.6 × mean(tested) and strictly shrunk toward 0.
    assert seed == pytest.approx(diagnose.DIAG_SHRINK * tested_theta)
    assert abs(seed) < abs(tested_theta)

    # Persisted to mastery.
    m = db.scalars(
        select(Mastery).where(Mastery.concept_id == untested.id)
    ).first()
    assert m is not None


def test_reseeding_is_idempotent_without_force():
    db = mem_session()
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Ch1")
    c = seed_concept(db, course, "c", chapter=ch)
    q = seed_question(db, course, c, correct_key="A")
    user = new_user()
    graded = [diagnose.Graded(question=q, chosen_key="A", confidence="sure")]

    diagnose.seed_diagnostic(db, user_id=user, course=course, graded=graded, build_status=False)
    assert diagnose.diagnostic_exists(db, user, course.id) is True

    # Second run without force refuses.
    with pytest.raises(diagnose.DiagnosticAlreadyRun):
        diagnose.seed_diagnostic(
            db, user_id=user, course=course, graded=graded, build_status=False
        )

    # With force it reseeds.
    result = diagnose.seed_diagnostic(
        db, user_id=user, course=course, graded=graded, force=True, build_status=False
    )
    assert result.n_questions == 1


def test_confident_wrong_sets_misconception_pending():
    db = mem_session()
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Ch1")
    c = seed_concept(db, course, "c", chapter=ch)
    q = seed_question(db, course, c, correct_key="A")
    user = new_user()
    graded = [diagnose.Graded(question=q, chosen_key="B", confidence="sure")]  # confident-wrong

    result = diagnose.seed_diagnostic(
        db, user_id=user, course=course, graded=graded, build_status=False
    )
    assert c.id in result.misconceptions
    m = db.scalars(select(Mastery).where(Mastery.concept_id == c.id)).first()
    assert m.misconception_pending is True
