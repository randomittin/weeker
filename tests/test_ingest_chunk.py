"""T-13 gate: section-bounded chunking, token histogram, coverage, no cross-chapter."""

from __future__ import annotations

import random

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atlas.core.config import CHUNK_MAX, CHUNK_MIN
from atlas.core.models import Base, Chapter, Chunk, Course, Page, Section, Source
from atlas.ingest import chunk as C
from atlas.verify import ingest_checks

_SENTENCE = (
    "A mutual fund pools money from many investors to buy a diversified portfolio "
    "of securities managed by a professional asset management company. "
)


def _para(n_sentences: int) -> str:
    return "".join(_SENTENCE for _ in range(n_sentences))


def test_chunk_segment_atomic_table_not_split():
    text = _para(20) + "\nFund | NAV | Return\nA | 10.5 | 8%\nB | 12.0 | 9%\n" + _para(20)
    units = C.units_from_pages([(5, text)])
    table_units = [u for u in units if "|" in u.text and "\n" in u.text]
    assert table_units, "table should be one atomic multi-line unit"


def test_chunk_histogram_and_coverage():
    # One section, plenty of text → several in-range chunks.
    pages = [(p, _para(60)) for p in range(1, 9)]
    chunks = C.chunk_segment(
        C.units_from_pages(pages), chapter_id=None, section_id=None
    )
    assert len(chunks) >= 2
    for ch in chunks:
        assert CHUNK_MIN <= ch.token_count <= CHUNK_MAX, ch.token_count
        assert ch.page_start >= 1 and ch.page_end <= 8
        assert len(ch.content_hash) == 64


def test_no_chunk_crosses_chapter_boundary_randomized():
    rng = random.Random(1234)
    for _ in range(25):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        with Session(engine) as s:
            course = Course(title="c")
            s.add(course)
            s.flush()
            source = Source(course_id=course.id, filename="f", sha256="x" * 64)
            s.add(source)
            s.flush()

            page = 1
            chapter_ranges = {}
            n_chap = rng.randint(2, 5)
            for ci in range(n_chap):
                n_pages = rng.randint(2, 6)
                start = page
                end = page + n_pages - 1
                ch = Chapter(
                    course_id=course.id, ordinal=ci + 1, title=f"Ch{ci}",
                    page_start=start, page_end=end,
                )
                s.add(ch)
                s.flush()
                chapter_ranges[ch.id] = (start, end)
                # random sections tiling the chapter
                p = start
                secs = []
                while p <= end:
                    seg_end = min(end, p + rng.randint(0, 2))
                    secs.append((p, seg_end))
                    p = seg_end + 1
                for si, (a, b) in enumerate(secs):
                    s.add(Section(chapter_id=ch.id, ordinal=si + 1, title=f"S{si}",
                                  page_start=a, page_end=b))
                for pg in range(start, end + 1):
                    s.add(Page(source_id=source.id, page_number=pg,
                               text=_para(rng.randint(20, 40)), char_count=1))
                page = end + 1
            s.flush()

            chunks = C.chunk_course(s, course)
            s.commit()
            assert chunks
            for ch in s.query(Chunk).all():
                assert ch.chapter_id is not None
                lo, hi = chapter_ranges[ch.chapter_id]
                assert lo <= ch.page_start <= ch.page_end <= hi, (
                    ch.page_start, ch.page_end, lo, hi
                )


def test_chunk_course_persists_and_verifies():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course = Course(title="c")
        s.add(course)
        s.flush()
        source = Source(course_id=course.id, filename="f", sha256="y" * 64)
        s.add(source)
        s.flush()
        ch = Chapter(course_id=course.id, ordinal=1, title="Ch1", page_start=1, page_end=6)
        s.add(ch)
        s.flush()
        s.add(Section(chapter_id=ch.id, ordinal=1, title="S1", page_start=1, page_end=6))
        for pg in range(1, 7):
            s.add(Page(source_id=source.id, page_number=pg, text=_para(60), char_count=1))
        s.flush()
        C.chunk_course(s, course)
        s.commit()
        assert ingest_checks.check_chunks(s, course.id) == []
        # re-run replaces, no duplication
        C.chunk_course(s, course)
        s.commit()
        assert ingest_checks.check_chunks(s, course.id) == []
