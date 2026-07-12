"""T-12 gate: strategy chain outline→font→LLM; structure gate as code."""

from __future__ import annotations

import json

import fitz
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from weeker.core.models import Base, Chapter, Course, Section
from weeker.ingest import structure as S


def test_check_structure_flags_overlap_and_gap():
    assert S.check_structure([(1, 10), (11, 20)], 20) == []
    assert any("overlap" in f for f in S.check_structure([(1, 12), (10, 20)], 20))
    assert any("coverage" in f for f in S.check_structure([(1, 5)], 100))


def _blank(doc, n):
    for _ in range(n):
        doc.new_page()


def test_outline_strategy(tmp_path):
    doc = fitz.open()
    _blank(doc, 20)
    doc.set_toc(
        [
            [1, "Chapter One: Basics", 1],
            [2, "1.1 Intro", 2],
            [2, "1.2 Terms", 5],
            [1, "Chapter Two: Regulation", 11],
            [2, "2.1 SEBI", 12],
        ]
    )
    p = tmp_path / "toc.pdf"
    doc.save(str(p))
    doc.close()
    res = S.build_structure(p)
    assert res.strategy == "outline"
    assert [c.title for c in res.chapters] == ["Chapter One: Basics", "Chapter Two: Regulation"]
    assert res.chapters[0].page_end == 10
    assert res.chapters[1].page_end == 20
    assert len(res.chapters[0].sections) == 2
    assert S.check_structure([(c.page_start, c.page_end) for c in res.chapters], 20) == []


def test_font_heuristic_strategy(tmp_path):
    doc = fitz.open()
    titles = {2: "Introduction to Mutual Funds", 8: "Regulatory Framework"}
    for i in range(14):
        page = doc.new_page()
        if (i + 1) in titles:
            page.insert_text((72, 90), titles[i + 1], fontsize=24)  # chapter heading
        page.insert_text((72, 140), f"Body text for page {i + 1}. " * 6, fontsize=11)
    p = tmp_path / "font.pdf"
    doc.save(str(p))
    doc.close()
    res = S.build_structure(p)
    assert res.strategy == "font"
    assert len(res.chapters) == 2
    assert res.chapters[0].title == "Introduction to Mutual Funds"
    assert S.check_structure([(c.page_start, c.page_end) for c in res.chapters], 14) == []


def test_llm_strategy_when_no_outline_or_font(tmp_path):
    # Uniform font, no outline → font finds no headings → LLM strategy used.
    doc = fitz.open()
    for i in range(10):
        page = doc.new_page()
        page.insert_text((72, 100), f"Uniform body text page {i + 1}. " * 5, fontsize=11)
    p = tmp_path / "flat.pdf"
    doc.save(str(p))
    doc.close()

    payload = {
        "chapters": [
            {"ordinal": 1, "title": "Part A", "page_start": 1, "page_end": 5,
             "sections": [{"ordinal": 1, "title": "A.1", "page_start": 1, "page_end": 5}]},
            {"ordinal": 2, "title": "Part B", "page_start": 6, "page_end": 10, "sections": []},
        ]
    }

    class FakeTransport:
        def __init__(self):
            self.calls = 0

        def request(self, model, system, user):
            self.calls += 1
            return json.dumps(payload)

    ft = FakeTransport()
    res = S.build_structure(p, transport=ft)
    assert res.strategy == "llm"
    assert ft.calls == 1
    assert [c.title for c in res.chapters] == ["Part A", "Part B"]


def test_persist_structure(tmp_path):
    doc = fitz.open()
    _blank(doc, 20)
    doc.set_toc([[1, "Ch1", 1], [1, "Ch2", 11], [2, "2.1", 12]])
    p = tmp_path / "toc.pdf"
    doc.save(str(p))
    doc.close()
    res = S.build_structure(p)

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course = Course(title="c")
        s.add(course)
        s.flush()
        chapters, sections = S.persist_structure(s, course, res)
        s.commit()
        assert s.query(Chapter).count() == 2
        assert s.query(Section).count() == 1
        # re-persist replaces, does not duplicate
        S.persist_structure(s, course, res)
        s.commit()
        assert s.query(Chapter).count() == 2
