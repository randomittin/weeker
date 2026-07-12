"""SQLAlchemy 2.0 declarative models — the full Weeker schema.

Derived from the interfaces referenced across 016 (task graph) and 017 (caselets
patch). Concepts are the unit of mastery; every question, flashcard, attempt and
prediction keys on a concept. Embeddings use the cross-dialect :class:`Vector`
type so the schema imports and tests run on SQLite while Postgres gets real
``vector(dim)`` columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from weeker.core.config import EMBED_DIM
from weeker.core.types import IntArray, JSONType, TextArray, UuidArray, Vector


class Base(DeclarativeBase):
    """Declarative base for all Weeker models."""


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


# ── corpus ─────────────────────────────────────────────────────────────────────
class Course(Base):
    __tablename__ = "courses"

    id: Mapped[uuid.UUID] = _pk()
    title: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str | None] = mapped_column(Text, unique=True)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=EMBED_DIM)
    exam_config: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created()

    sources: Mapped[list[Source]] = relationship(back_populates="course")


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("course_id", "sha256", name="uq_source_sha"),)

    id: Mapped[uuid.UUID] = _pk()
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    page_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _created()

    course: Mapped[Course] = relationship(back_populates="sources")


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("source_id", "page_number", name="uq_page_number"),)

    id: Mapped[uuid.UUID] = _pk()
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ocr_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[uuid.UUID] = _pk()
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[uuid.UUID] = _pk()
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _pk()
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    chapter_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chapters.id", ondelete="SET NULL")
    )
    section_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sections.id", ondelete="SET NULL")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list | None] = mapped_column(Vector(EMBED_DIM))
    created_at: Mapped[datetime] = _created()


# ── concepts ─────────────────────────────────────────────────────────────────
class Concept(Base):
    __tablename__ = "concepts"

    id: Mapped[uuid.UUID] = _pk()
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    chapter_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chapters.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    prerequisites: Mapped[list] = mapped_column(TextArray, nullable=False, default=list)
    keywords: Mapped[list] = mapped_column(TextArray, nullable=False, default=list)
    misconceptions: Mapped[list] = mapped_column(TextArray, nullable=False, default=list)
    source_pages: Mapped[list] = mapped_column(IntArray, nullable=False, default=list)
    embedding: Mapped[list | None] = mapped_column(Vector(EMBED_DIM))
    review_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        CheckConstraint(
            "review_status in ('pending','accepted','edited','merged','dropped')",
            name="ck_concept_review_status",
        ),
        CheckConstraint("difficulty in (1,2,3)", name="ck_concept_difficulty"),
    )


class ConceptChunk(Base):
    __tablename__ = "concept_chunks"
    __table_args__ = (
        UniqueConstraint("concept_id", "chunk_id", name="uq_concept_chunk"),
    )

    id: Mapped[uuid.UUID] = _pk()
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    cos: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)


class Objective(Base):
    __tablename__ = "objectives"

    id: Mapped[uuid.UUID] = _pk()
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    bloom: Mapped[str] = mapped_column(String(16), nullable=False)
    assessment_style: Mapped[str] = mapped_column(String(24), nullable=False)


class BlueprintWeight(Base):
    __tablename__ = "blueprint_weights"
    __table_args__ = (
        UniqueConstraint("course_id", "chapter", name="uq_blueprint_chapter"),
    )

    id: Mapped[uuid.UUID] = _pk()
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    chapter: Mapped[str] = mapped_column(Text, nullable=False)
    weight: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)


# ── questions & caselets ───────────────────────────────────────────────────────
class CaseGroup(Base):
    """Caselet scenario grouping five concept probes (017 §2)."""

    __tablename__ = "case_groups"
    __table_args__ = (
        CheckConstraint(
            "status in ('active','disputed','retired')", name="ck_case_group_status"
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    scenario: Mapped[str] = mapped_column(Text, nullable=False)
    concept_ids: Mapped[list] = mapped_column(UuidArray, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    gate_log: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    generator_model: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created()


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        CheckConstraint("correct_key in ('A','B','C','D')", name="ck_question_correct_key"),
        # Partial unique index enforcing at most one keyed (correct) row per
        # question id — the "one correct answer per question" invariant. Created
        # on both dialects; verify.schema asserts its presence.
        Index(
            "one_correct_per_question",
            "id",
            unique=True,
            postgresql_where=text("correct_key is not null"),
            sqlite_where=text("correct_key is not null"),
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    case_group_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("case_groups.id", ondelete="SET NULL")
    )
    stem: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[list] = mapped_column(JSONType, nullable=False)
    correct_key: Mapped[str] = mapped_column(String(1), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    q_rating: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, default=0.0)
    discrimination: Mapped[float | None] = mapped_column(Numeric(6, 4))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    gate_log: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    generator_model: Mapped[str | None] = mapped_column(Text)
    marks: Mapped[float] = mapped_column(Numeric(3, 1), nullable=False, default=1)
    case_position: Mapped[int | None] = mapped_column(Integer)
    pull_count: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created()


# ── learning state ──────────────────────────────────────────────────────────────
class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL")
    )
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    chosen_key: Mapped[str | None] = mapped_column(String(1))
    theta_before: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    theta_after: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    created_at: Mapped[datetime] = _created()

    __table_args__ = (
        CheckConstraint(
            "confidence in ('sure','unsure','guessing')", name="ck_attempt_confidence"
        ),
    )


class Mastery(Base):
    __tablename__ = "mastery"
    __table_args__ = (
        UniqueConstraint("user_id", "concept_id", name="uq_mastery_user_concept"),
    )

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    theta: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False, default=0.0)
    stability: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False, default=1.0)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    misconception_pending: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Flashcard(Base):
    __tablename__ = "flashcards"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    front: Mapped[str] = mapped_column(Text, nullable=False)
    back: Mapped[str] = mapped_column(Text, nullable=False)
    misconception_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("misconceptions.id", ondelete="SET NULL")
    )
    stability: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False, default=1.0)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()


class Misconception(Base):
    __tablename__ = "misconceptions"

    id: Mapped[uuid.UUID] = _pk()
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")


class MisconceptionFlag(Base):
    __tablename__ = "misconception_flags"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    misconception_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("misconceptions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    confident_correct_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _created()
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ── sessions, mocks, telemetry ──────────────────────────────────────────────────
class SessionRun(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(
            "kind in ('warmup','core','mixed','review','diagnostic','caselet','mock','flash')",
            name="ck_session_kind",
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    config: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    started_at: Mapped[datetime] = _created()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MockExam(Base):
    __tablename__ = "mock_exams"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL")
    )
    raw_score: Mapped[float | None] = mapped_column(Numeric(6, 2))
    total_marks: Mapped[float | None] = mapped_column(Numeric(6, 2))
    passed: Mapped[bool | None] = mapped_column(Boolean)
    predicted_before: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    config: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    started_at: Mapped[datetime] = _created()
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created()


class ReviewQueue(Base):
    __tablename__ = "review_queue"
    __table_args__ = (
        CheckConstraint(
            "kind in ('concept','question','caselet')", name="ck_review_queue_kind"
        ),
        CheckConstraint(
            "status in ('pending','resolved','skipped')", name="ck_review_queue_status"
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    ref_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[datetime] = _created()
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
