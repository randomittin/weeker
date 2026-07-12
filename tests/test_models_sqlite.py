"""Models import and create cleanly on an in-memory SQLite engine."""

from __future__ import annotations

import uuid

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from weeker.core.models import Base, Concept, Course, Question

EXPECTED_TABLES = {
    "courses",
    "sources",
    "pages",
    "chapters",
    "sections",
    "chunks",
    "concepts",
    "concept_chunks",
    "objectives",
    "blueprint_weights",
    "case_groups",
    "questions",
    "attempts",
    "mastery",
    "flashcards",
    "misconceptions",
    "misconception_flags",
    "sessions",
    "mock_exams",
    "analytics_events",
    "review_queue",
}


def _mem_engine():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine


def test_all_tables_created():
    engine = _mem_engine()
    tables = set(inspect(engine).get_table_names())
    missing = EXPECTED_TABLES - tables
    assert not missing, f"missing tables: {missing}"
    assert len(EXPECTED_TABLES) >= 20


def test_one_correct_per_question_index_exists():
    engine = _mem_engine()
    idx_names = {ix["name"] for ix in inspect(engine).get_indexes("questions")}
    assert "one_correct_per_question" in idx_names


def test_insert_roundtrip():
    engine = _mem_engine()
    with Session(engine) as s:
        course = Course(title="NISM XA")
        s.add(course)
        s.flush()
        concept = Concept(course_id=course.id, title="Time value of money")
        s.add(concept)
        s.flush()
        q = Question(
            course_id=course.id,
            concept_id=concept.id,
            stem="What is compounding?",
            options=[{"key": "A", "text": "x"}, {"key": "B", "text": "y"}],
            correct_key="A",
        )
        s.add(q)
        s.commit()
        got = s.get(Question, q.id)
        assert got.correct_key == "A"
        assert got.marks == 1
        assert isinstance(got.id, uuid.UUID)


def test_concept_defaults():
    engine = _mem_engine()
    with Session(engine) as s:
        course = Course(title="c")
        s.add(course)
        s.flush()
        concept = Concept(course_id=course.id, title="t")
        s.add(concept)
        s.commit()
        assert concept.review_status == "pending"
        assert concept.prerequisites == []
        assert concept.keywords == []
