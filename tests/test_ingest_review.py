"""T-16 gate: scripted Textual Pilot pass over 3 fixture concepts.

Drives the accept/edit/drop keystrokes and asserts the resulting
``review_status`` transitions persist to the database. Uses a file-backed
SQLite engine so the app's own session (opened from the same engine) sees the
seeded rows and the assertions see the app's commits.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from weeker.core.config import EMBED_DIM
from weeker.core.models import (
    Base,
    Chapter,
    Chunk,
    Concept,
    ConceptChunk,
    Course,
    Source,
)
from weeker.ingest.review import ReviewApp


def _engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'review.db'}")
    Base.metadata.create_all(engine)
    return engine


def _seed(engine):
    with Session(engine) as s:
        course = Course(title="NISM")
        s.add(course)
        s.flush()
        source = Source(course_id=course.id, filename="wb.pdf", sha256="a" * 64)
        s.add(source)
        s.flush()
        ch = Chapter(course_id=course.id, ordinal=1, title="Funds", page_start=1, page_end=9)
        s.add(ch)
        s.flush()
        # Three pending concepts, ordered by title: Alpha, Beta, Gamma.
        for title, page in (("Alpha", 1), ("Beta", 4), ("Gamma", 7)):
            c = Concept(
                course_id=course.id,
                chapter_id=ch.id,
                title=title,
                description=f"{title} description",
                difficulty=2,
                keywords=[title.lower()],
                source_pages=[page],
                review_status="pending",
            )
            s.add(c)
            s.flush()
        # A supporting chunk + link for Alpha so 'peek' has real source to render.
        chunk = Chunk(
            course_id=course.id,
            source_id=source.id,
            chapter_id=ch.id,
            content="Alpha grounds in this passage about pooled fund mechanics.",
            content_hash="c" * 64,
            token_count=9,
            page_start=1,
            page_end=1,
        )
        chunk.embedding = np.zeros(EMBED_DIM, dtype=np.float32)
        s.add(chunk)
        s.flush()
        alpha = s.query(Concept).filter(Concept.title == "Alpha").one()
        s.add(ConceptChunk(concept_id=alpha.id, chunk_id=chunk.id, cos=0.90))
        s.commit()
        return course.id


@pytest.mark.asyncio
async def test_review_pilot_accept_edit_drop(tmp_path):
    engine = _engine(tmp_path)
    course_id = _seed(engine)
    app = ReviewApp(engine, course_id=course_id)

    async with app.run_test() as pilot:
        # Alpha -> accept
        await pilot.press("a")
        await pilot.pause()
        # Beta -> edit; the edit modal opens, Enter on the title input saves.
        await pilot.press("e")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        # Gamma -> drop
        await pilot.press("d")
        await pilot.pause()

    with Session(engine) as s:
        by_title = {c.title: c for c in s.query(Concept).all()}
        assert by_title["Alpha"].review_status == "accepted"
        assert by_title["Beta"].review_status == "edited"
        assert by_title["Gamma"].review_status == "dropped"
        for title in ("Alpha", "Beta", "Gamma"):
            assert by_title[title].reviewed_at is not None


@pytest.mark.asyncio
async def test_review_pilot_resumable(tmp_path):
    """A relaunch only re-queries rows still pending (accept one, reopen)."""
    engine = _engine(tmp_path)
    course_id = _seed(engine)

    app = ReviewApp(engine, course_id=course_id)
    async with app.run_test() as pilot:
        await pilot.press("a")  # Alpha accepted
        await pilot.pause()

    # Relaunch: Alpha is done, so only Beta + Gamma remain pending.
    app2 = ReviewApp(engine, course_id=course_id)
    async with app2.run_test() as pilot:
        await pilot.pause()
        assert [c.title for c in app2.pending] == ["Beta", "Gamma"]
        await pilot.press("d")  # Beta dropped
        await pilot.pause()

    with Session(engine) as s:
        by_title = {c.title: c.review_status for c in s.query(Concept).all()}
        assert by_title == {"Alpha": "accepted", "Beta": "dropped", "Gamma": "pending"}


@pytest.mark.asyncio
async def test_review_pilot_peek_source(tmp_path):
    """'p' renders the top linked source chunk for the current concept."""
    engine = _engine(tmp_path)
    course_id = _seed(engine)
    app = ReviewApp(engine, course_id=course_id)
    async with app.run_test() as pilot:
        await pilot.press("p")  # peek Alpha's source
        await pilot.pause()
        from textual.widgets import Static

        src = app.query_one("#source", Static).render()
        assert "pooled fund mechanics" in str(src)
