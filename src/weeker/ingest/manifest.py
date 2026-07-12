"""S0 — manifest.yaml models, loader, and course/source upsert (016 §T-10, 017 §1).

The manifest is the single declared input to ingestion: the course identity, the
source PDFs (hashed by content for idempotent upsert), the exam structure (017 §1
verbatim), and optional per-chapter blueprint weight overrides. Validation is
strict: a missing file, a blueprint that does not sum to 1, or a declared
``total_marks`` that disagrees with the section breakdown all raise before any
row is written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from weeker.core.config import EMBED_DIM
from weeker.core.models import Course, Source

_BLUEPRINT_TOL = 1e-6


class CourseSpec(BaseModel):
    title: str
    slug: str | None = None


class SourceSpec(BaseModel):
    filename: str
    path: str
    page_offset: int = 0

    @field_validator("page_offset")
    @classmethod
    def _nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("page_offset must be >= 0")
        return v


class ExamSection(BaseModel):
    kind: Literal["standalone", "caselet"]
    questions: int | None = None       # standalone: total questions
    marks_each: float = 1
    count: int | None = None           # caselet: number of caselets
    questions_each: int | None = None  # caselet: questions per caselet

    def marks(self) -> float:
        if self.kind == "standalone":
            if self.questions is None:
                raise ValueError("standalone section needs 'questions'")
            return self.questions * self.marks_each
        if self.count is None or self.questions_each is None:
            raise ValueError("caselet section needs 'count' and 'questions_each'")
        return self.count * self.questions_each * self.marks_each


class ExamBlock(BaseModel):
    duration_minutes: int
    passing_marks: float
    total_marks: float
    negative_marking_fraction: float = 0.25
    sections: list[ExamSection]

    def computed_marks(self) -> float:
        return sum(s.marks() for s in self.sections)

    @model_validator(mode="after")
    def _check(self) -> ExamBlock:
        if self.passing_marks > self.total_marks:
            raise ValueError("passing_marks exceeds total_marks")
        computed = self.computed_marks()
        if abs(computed - self.total_marks) > _BLUEPRINT_TOL:
            raise ValueError(
                f"exam total_marks {self.total_marks} != sum of sections {computed}"
            )
        return self


class BlueprintBlock(BaseModel):
    overrides: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sum_to_one(self) -> BlueprintBlock:
        if self.overrides:
            total = sum(self.overrides.values())
            if abs(total - 1.0) > _BLUEPRINT_TOL:
                raise ValueError(f"blueprint overrides sum to {total}, expected 1.0")
        return self


class Manifest(BaseModel):
    course: CourseSpec
    sources: list[SourceSpec]
    exam: ExamBlock
    blueprint: BlueprintBlock = Field(default_factory=BlueprintBlock)

    @field_validator("sources")
    @classmethod
    def _nonempty(cls, v: list[SourceSpec]) -> list[SourceSpec]:
        if not v:
            raise ValueError("manifest must declare at least one source")
        return v


def load_manifest(path: str | Path) -> Manifest:
    """Parse and validate ``manifest.yaml``. Raises on missing file / bad shape."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Manifest.model_validate(data)
    except ValidationError as e:  # surface a clean message for the gates
        raise ValueError(f"invalid manifest: {e}") from e


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def upsert(
    session: Session, manifest: Manifest, corpus_root: str | Path
) -> tuple[Course, list[Source]]:
    """Upsert the course and its sources by content sha256; idempotent.

    Files are read from ``corpus_root`` (joined with each source's ``path``).
    A missing file raises ``FileNotFoundError``; two sources hashing to the same
    content raise ``ValueError`` (a duplicate upload).
    """
    root = Path(corpus_root)
    spec = manifest.course
    course = None
    if spec.slug is not None:
        course = session.scalar(select(Course).where(Course.slug == spec.slug))
    if course is None:
        course = Course(title=spec.title, slug=spec.slug, embedding_dim=EMBED_DIM)
        course.exam_config = manifest.exam.model_dump()
        session.add(course)
        session.flush()
    else:
        course.title = spec.title
        course.exam_config = manifest.exam.model_dump()
        session.flush()

    seen: dict[str, str] = {}
    sources: list[Source] = []
    for spec_src in manifest.sources:
        fpath = root / spec_src.path
        if not fpath.exists():
            raise FileNotFoundError(f"source file not found: {fpath}")
        sha = _sha256(fpath)
        if sha in seen:
            raise ValueError(
                f"duplicate source content: {spec_src.filename} == {seen[sha]} (sha {sha[:12]})"
            )
        seen[sha] = spec_src.filename

        row = session.scalar(
            select(Source).where(Source.course_id == course.id, Source.sha256 == sha)
        )
        if row is None:
            row = Source(
                course_id=course.id,
                filename=spec_src.filename,
                sha256=sha,
                page_offset=spec_src.page_offset,
            )
            session.add(row)
            session.flush()
        else:
            row.filename = spec_src.filename
            row.page_offset = spec_src.page_offset
            session.flush()
        sources.append(row)

    return course, sources
