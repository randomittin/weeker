"""T-11 gate: PyMuPDF parse — de-hyphenation, header/footer strip, OCR fallback, coverage."""

from __future__ import annotations

import fitz
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atlas.core.models import Base, Course, Source
from atlas.ingest import parse as P
from atlas.verify import ingest_checks


def test_dehyphenate_joins_line_broken_words():
    src = "The regula-\ntory frame-\nwork applies. Well-\nknown facts."
    out = P.dehyphenate(src)
    assert "regulatory" in out
    assert "framework" in out
    # A hyphen followed by uppercase is a real compound, not a line break —
    # but our source has none; ensure no spurious joins of separate words.
    assert "regula-tory" not in out


def test_detect_header_footer_by_page_share():
    # 5 pages: identical top line, page-numbered bottom line, unique bodies.
    pages_lines = [
        ["NISM WORKBOOK", f"body {i}", f"Page {i + 1}"] for i in range(5)
    ]
    headers, footers = P.detect_header_footer(pages_lines, share=0.3)
    combined = headers | footers
    assert P._normalize("NISM WORKBOOK") in combined
    assert P._normalize("Page 1") in combined  # digits normalised → shared form
    assert P._normalize("body 1") not in combined


def _make_pdf(path, n=6):
    doc = fitz.open()
    for i in range(n):
        page = doc.new_page()
        page.insert_text((72, 60), "NISM WORKBOOK")  # repeated header
        body = (
            f"Chapter body paragraph number {i}. Mutual funds pool investor "
            f"capital. The regula-\ntory frame-\nwork is set by the regulator "
            f"and enforced through inspection and audit procedures each year."
        )
        page.insert_text((72, 120), body)
        page.insert_text((72, 760), f"Page {i + 1}")  # running footer
    doc.save(str(path))
    doc.close()


def test_parse_source_strips_chrome_and_dehyphenates(tmp_path):
    pdf = tmp_path / "wb.pdf"
    _make_pdf(pdf, n=6)
    res = P.parse_source(pdf)
    assert len(res.pages) == 6
    text0 = res.pages[0].text
    assert "NISM WORKBOOK" not in text0      # header stripped
    assert "Page 1" not in text0             # footer stripped
    assert "regulatory" in text0             # de-hyphenated
    assert "framework" in text0
    assert res.coverage >= 0.98
    assert res.empty_pages == []


def test_ocr_fallback_triggers_below_min_chars(tmp_path):
    # A page with almost no extractable text must invoke the OCR function.
    doc = fitz.open()
    doc.new_page()  # blank page
    doc.new_page().insert_text((72, 72), "x")  # < OCR_MIN_CHARS
    p = tmp_path / "scan.pdf"
    doc.save(str(p))
    doc.close()

    calls = {"n": 0}

    def fake_ocr(page):
        calls["n"] += 1
        return "Recovered scanned text: net asset value is computed daily here."

    res = P.parse_source(p, ocr_fn=fake_ocr)
    assert calls["n"] == 2                    # both thin pages OCR'd
    assert all(pg.ocr_used for pg in res.pages)
    assert res.empty_pages == []
    assert res.coverage >= 0.98


def test_serialize_table_pipe_rows():
    rows = [["Fund", "NAV"], ["A", "10.5"], ["B", None]]
    out = P.serialize_table(rows)
    assert "Fund | NAV" in out
    assert "B |" in out


def test_persist_and_verify_parse(tmp_path):
    pdf = tmp_path / "wb.pdf"
    _make_pdf(pdf, n=4)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course = Course(title="c")
        s.add(course)
        s.flush()
        source = Source(course_id=course.id, filename="wb.pdf", sha256="a" * 64)
        s.add(source)
        s.flush()
        res = P.parse_source(pdf)
        P.persist_pages(s, source, res)
        s.commit()
        failures = ingest_checks.check_parse(s, source.id)
        assert failures == [], failures
