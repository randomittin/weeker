"""T-24 gate: disputed-question review TUI (Textual Pilot) — keep / fix-key / discard."""

from __future__ import annotations

from sqlalchemy import select

from tests.test_learn_fixtures import mem_session, seed_concept, seed_course
from weeker.core.models import Question
from weeker.generate.review_tui import DisputeApp


def _disputed(db, course, concept, correct_key="A", votes=None):
    q = Question(
        course_id=course.id,
        concept_id=concept.id,
        stem="What is the answer?",
        options=[
            {"key": "A", "text": "alpha"},
            {"key": "B", "text": "beta", "misconception": "b"},
            {"key": "C", "text": "gamma", "misconception": "c"},
            {"key": "D", "text": "delta", "misconception": "d"},
        ],
        correct_key=correct_key,
        explanation="alpha is keyed correct",
        difficulty=2,
        status="disputed",
        gate_log={
            "G3": {
                "passed": False,
                "disputed": True,
                "votes": votes or ["B"],
                "solver_rationales": ["I believe B is correct"],
            }
        },
    )
    db.add(q)
    db.flush()
    return q


async def test_keep_fix_discard_resolutions():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    q_keep = _disputed(db, course, concept)
    q_fix = _disputed(db, course, concept)
    q_discard = _disputed(db, course, concept)

    app = DisputeApp([q_keep, q_fix, q_discard], db=db)
    async with app.run_test() as pilot:
        await pilot.press("k")   # keep first
        await pilot.press("2")   # fix key to B on second
        await pilot.press("x")   # discard third
        assert app.phase == "done"

    assert db.get(Question, q_keep.id).status == "active"
    fixed = db.get(Question, q_fix.id)
    assert fixed.status == "active" and fixed.correct_key == "B"
    assert fixed.gate_log["dispute"]["resolution"] == "fix-key"
    assert db.get(Question, q_discard.id).status == "retired"


async def test_resumable_leaves_unresolved_disputed():
    db = mem_session()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    q1 = _disputed(db, course, concept)
    q2 = _disputed(db, course, concept)

    app = DisputeApp([q1, q2], db=db)
    async with app.run_test() as pilot:
        await pilot.press("k")  # resolve only the first, then quit
        await pilot.press("q")

    # first resolved active, second still disputed → next launch re-queues it
    assert db.get(Question, q1.id).status == "active"
    still = db.scalars(select(Question).where(Question.status == "disputed")).all()
    assert q2.id in {q.id for q in still}
