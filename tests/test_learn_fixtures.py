"""Shared builders for the learn-wave tests (no test cases live here).

Named ``test_learn_*`` to satisfy the wave's file-naming rule; pytest collects it
but finds nothing to run. Provides an in-memory SQLite session plus seed helpers
so each learn test builds an isolated corpus without touching the real DB.
"""

from __future__ import annotations

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atlas.core.models import Base, Chapter, Concept, Course, Mastery, Question


def mem_session() -> Session:
    """Fresh in-memory SQLite session with the full schema created."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def seed_course(db: Session, title: str = "NISM XA") -> Course:
    course = Course(title=title)
    db.add(course)
    db.flush()
    return course


def seed_chapter(db: Session, course: Course, ordinal: int, title: str) -> Chapter:
    ch = Chapter(
        course_id=course.id,
        ordinal=ordinal,
        title=title,
        page_start=ordinal * 10,
        page_end=ordinal * 10 + 9,
    )
    db.add(ch)
    db.flush()
    return ch


def seed_concept(
    db: Session,
    course: Course,
    title: str,
    *,
    chapter: Chapter | None = None,
    difficulty: int = 2,
) -> Concept:
    c = Concept(
        course_id=course.id,
        chapter_id=chapter.id if chapter else None,
        title=title,
        difficulty=difficulty,
    )
    db.add(c)
    db.flush()
    return c


def seed_question(
    db: Session,
    course: Course,
    concept: Concept,
    *,
    stem: str = "What is X?",
    correct_key: str = "A",
    q_rating: float = 0.0,
    difficulty: int = 2,
    marks: float = 1,
    distractor_misconception: str | None = None,
    variant_group: str | None = None,
) -> Question:
    options = [
        {"key": "A", "text": "the right answer"},
        {"key": "B", "text": "a wrong answer"},
        {"key": "C", "text": "another wrong answer"},
        {"key": "D", "text": "yet another"},
    ]
    if distractor_misconception:
        options[1]["misconception"] = distractor_misconception
    gate_log = {"variant_group": variant_group} if variant_group else {}
    q = Question(
        course_id=course.id,
        concept_id=concept.id,
        stem=stem,
        options=options,
        correct_key=correct_key,
        explanation="because reasons",
        difficulty=difficulty,
        q_rating=q_rating,
        marks=marks,
        gate_log=gate_log,
    )
    db.add(q)
    db.flush()
    return q


def seed_mastery(
    db: Session, user_id: str, concept: Concept, *, theta: float = 0.0, stability: float = 1.0
) -> Mastery:
    m = Mastery(user_id=user_id, concept_id=concept.id, theta=theta, stability=stability)
    db.add(m)
    db.flush()
    return m


def new_user() -> str:
    return f"user-{uuid.uuid4().hex[:8]}"
