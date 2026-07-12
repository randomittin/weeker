"""T-10 gate: manifest loads/validates; missing file, bad blueprint sum, dup sha rejected."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atlas.core.models import Base, Course, Source
from atlas.ingest import manifest as M

VALID = """
course:
  title: NISM Series X-A
  slug: nism-xa
sources:
  - filename: workbook.pdf
    path: workbook.pdf
    page_offset: 12
exam:
  duration_minutes: 180
  passing_marks: 90
  total_marks: 150
  negative_marking_fraction: 0.25
  sections:
    - kind: standalone
      questions: 90
      marks_each: 1
    - kind: caselet
      count: 6
      questions_each: 5
      marks_each: 1
    - kind: caselet
      count: 3
      questions_each: 5
      marks_each: 2
blueprint:
  overrides:
    Chapter 1: 0.4
    Chapter 2: 0.6
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _mem():
    e = create_engine("sqlite://")
    Base.metadata.create_all(e)
    return e


def test_valid_manifest_loads(tmp_path):
    p = _write(tmp_path, "manifest.yaml", VALID)
    man = M.load_manifest(p)
    assert man.course.title == "NISM Series X-A"
    assert man.sources[0].page_offset == 12
    assert man.exam.total_marks == 150
    # computed marks across sections match declared total.
    assert man.exam.computed_marks() == 150
    assert abs(sum(man.blueprint.overrides.values()) - 1.0) < 1e-9


def test_missing_manifest_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.load_manifest(tmp_path / "nope.yaml")


def test_bad_blueprint_sum_rejected(tmp_path):
    bad = VALID.replace("Chapter 2: 0.6", "Chapter 2: 0.3")
    p = _write(tmp_path, "m.yaml", bad)
    with pytest.raises(ValueError, match="blueprint"):
        M.load_manifest(p)


def test_exam_total_mismatch_rejected(tmp_path):
    bad = VALID.replace("total_marks: 150", "total_marks: 999")
    p = _write(tmp_path, "m.yaml", bad)
    with pytest.raises(ValueError, match="total_marks"):
        M.load_manifest(p)


def test_upsert_course_and_sources(tmp_path):
    _write(tmp_path, "workbook.pdf", "hello pdf bytes")
    p = _write(tmp_path, "manifest.yaml", VALID)
    man = M.load_manifest(p)
    e = _mem()
    with Session(e) as s:
        course, sources = M.upsert(s, man, corpus_root=tmp_path)
        s.commit()
        assert course.slug == "nism-xa"
        assert len(sources) == 1
        want = hashlib.sha256(b"hello pdf bytes").hexdigest()
        assert sources[0].sha256 == want
        assert sources[0].page_offset == 12
    # idempotent: re-run upserts same rows, no duplicates.
    with Session(e) as s:
        course2, sources2 = M.upsert(s, man, corpus_root=tmp_path)
        s.commit()
        assert course2.id == course.id
        assert s.query(Course).count() == 1
        assert s.query(Source).count() == 1


def test_missing_source_file_rejected(tmp_path):
    p = _write(tmp_path, "manifest.yaml", VALID)  # no workbook.pdf on disk
    man = M.load_manifest(p)
    e = _mem()
    with Session(e) as s:
        with pytest.raises(FileNotFoundError):
            M.upsert(s, man, corpus_root=tmp_path)


def test_duplicate_sha_rejected(tmp_path):
    _write(tmp_path, "a.pdf", "identical")
    _write(tmp_path, "b.pdf", "identical")
    dup = VALID.replace(
        "  - filename: workbook.pdf\n    path: workbook.pdf\n    page_offset: 12",
        "  - filename: a.pdf\n    path: a.pdf\n  - filename: b.pdf\n    path: b.pdf",
    )
    p = _write(tmp_path, "manifest.yaml", dup)
    man = M.load_manifest(p)
    e = _mem()
    with Session(e) as s:
        with pytest.raises(ValueError, match="duplicate"):
            M.upsert(s, man, corpus_root=tmp_path)
