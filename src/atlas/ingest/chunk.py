"""S3 — section-bounded chunking (016 §T-13).

Text is packed into ~``CHUNK_TARGET_TOKENS`` chunks *within a single section*
(and therefore within a single chapter — the property the gate enforces).
Tables and worked examples are atomic units that never split across a chunk
boundary. Consecutive chunks overlap by ``CHUNK_OVERLAP`` of the target to keep
retrieval context continuous. Every chunk carries a ``content_hash`` and the
page range it was drawn from.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core.config import (
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_OVERLAP,
    CHUNK_TARGET_TOKENS,
)
from atlas.core.models import Chapter, Chunk, Course, Page, Section, Source
from atlas.ingest.textutil import count_tokens

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_EXAMPLE_RE = re.compile(r"^\s*(example|illustration|case)\b", re.IGNORECASE)


@dataclass
class Unit:
    """An indivisible piece of source text on a single page."""

    page_number: int
    text: str
    atomic: bool = False


@dataclass
class ChunkData:
    content: str
    content_hash: str
    token_count: int
    page_start: int
    page_end: int
    chapter_id: uuid.UUID | None
    section_id: uuid.UUID | None


def _is_table_line(line: str) -> bool:
    return " | " in line


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _prose_units(prose: list[str], page_number: int) -> list[Unit]:
    joined = " ".join(x.strip() for x in prose if x.strip())
    out = []
    for sent in _SENT_SPLIT.split(joined):
        sent = sent.strip()
        if sent:
            out.append(Unit(page_number, sent))
    return out


def units_from_pages(pages: list[tuple[int, str]]) -> list[Unit]:
    """Split page text into ordered units, keeping tables/examples atomic."""
    units: list[Unit] = []
    for page_number, text in pages:
        lines = (text or "").splitlines()
        i = 0
        prose: list[str] = []
        while i < len(lines):
            line = lines[i]
            if _is_table_line(line):
                units.extend(_prose_units(prose, page_number))
                prose = []
                block = []
                while i < len(lines) and _is_table_line(lines[i]):
                    block.append(lines[i])
                    i += 1
                units.append(Unit(page_number, "\n".join(block), atomic=True))
                continue
            if _EXAMPLE_RE.match(line):
                units.extend(_prose_units(prose, page_number))
                prose = []
                block = [line]
                i += 1
                while i < len(lines) and lines[i].strip() and not _is_table_line(lines[i]):
                    block.append(lines[i])
                    i += 1
                units.append(Unit(page_number, "\n".join(block), atomic=True))
                continue
            prose.append(line)
            i += 1
        units.extend(_prose_units(prose, page_number))
    return units


def _finalize(
    group: list[Unit], chapter_id: uuid.UUID | None, section_id: uuid.UUID | None
) -> ChunkData:
    content = "\n".join(u.text for u in group).strip()
    pages = [u.page_number for u in group]
    return ChunkData(
        content=content,
        content_hash=_hash(content),
        token_count=count_tokens(content),
        page_start=min(pages),
        page_end=max(pages),
        chapter_id=chapter_id,
        section_id=section_id,
    )


def chunk_segment(
    units: list[Unit],
    chapter_id: uuid.UUID | None,
    section_id: uuid.UUID | None,
    target: int = CHUNK_TARGET_TOKENS,
    min_tokens: int = CHUNK_MIN,
    max_tokens: int = CHUNK_MAX,
    overlap: float = CHUNK_OVERLAP,
) -> list[ChunkData]:
    """Pack ``units`` into overlapping chunks bounded by the token budget."""
    if not units:
        return []
    overlap_budget = int(target * overlap)
    chunks: list[ChunkData] = []
    cur: list[Unit] = []
    cur_tok = 0
    seed_count = 0  # leading units carried over as overlap (not new content)

    def close() -> None:
        nonlocal cur, cur_tok, seed_count
        chunks.append(_finalize(cur, chapter_id, section_id))
        tail: list[Unit] = []
        tail_tok = 0
        for u in reversed(cur):
            ut = count_tokens(u.text)
            if tail_tok + ut > overlap_budget:
                break
            tail.insert(0, u)
            tail_tok += ut
        cur = tail
        cur_tok = tail_tok
        seed_count = len(cur)

    for unit in units:
        ut = count_tokens(unit.text)
        if cur and cur_tok + ut > max_tokens:
            close()
        cur.append(unit)
        cur_tok += ut
        if cur_tok >= target:
            close()

    # Trailing remainder: emit only if it carries units beyond the overlap seed.
    new_units = cur[seed_count:]
    if new_units:
        leftover = _finalize(cur, chapter_id, section_id)
        if leftover.token_count < min_tokens and chunks:
            prev = chunks[-1]
            content = (prev.content + "\n" + "\n".join(u.text for u in new_units)).strip()
            pages = [prev.page_start, prev.page_end, *(u.page_number for u in new_units)]
            merged = ChunkData(
                content=content,
                content_hash=_hash(content),
                token_count=count_tokens(content),
                page_start=min(pages),
                page_end=max(pages),
                chapter_id=chapter_id,
                section_id=section_id,
            )
            if merged.token_count <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(leftover)
        else:
            chunks.append(leftover)
    return chunks


def _segments(session: Session, course: Course) -> list[tuple]:
    """Yield (chapter_id, section_id, page_start, page_end) segments for a course."""
    segments: list[tuple] = []
    chapters = list(
        session.scalars(
            select(Chapter).where(Chapter.course_id == course.id).order_by(Chapter.page_start)
        )
    )
    for ch in chapters:
        sections = list(
            session.scalars(
                select(Section).where(Section.chapter_id == ch.id).order_by(Section.page_start)
            )
        )
        if sections:
            for sec in sections:
                segments.append((ch.id, sec.id, sec.page_start, sec.page_end))
        else:
            segments.append((ch.id, None, ch.page_start, ch.page_end))
    return segments


def _page_texts(session: Session, course: Course, lo: int, hi: int) -> list[tuple[int, str]]:
    rows = session.execute(
        select(Page.page_number, Page.text)
        .join(Source, Page.source_id == Source.id)
        .where(Source.course_id == course.id, Page.page_number >= lo, Page.page_number <= hi)
        .order_by(Page.page_number)
    ).all()
    return [(pn, txt or "") for pn, txt in rows]


def chunk_course(session: Session, course: Course) -> list[Chunk]:
    """Chunk every section of ``course`` and persist (replacing prior chunks)."""
    session.query(Chunk).filter(Chunk.course_id == course.id).delete()
    session.flush()
    source_id = session.scalar(select(Source.id).where(Source.course_id == course.id))

    rows: list[Chunk] = []
    for chapter_id, section_id, lo, hi in _segments(session, course):
        pages = _page_texts(session, course, lo, hi)
        units = units_from_pages(pages)
        for cd in chunk_segment(units, chapter_id, section_id):
            row = Chunk(
                course_id=course.id,
                source_id=source_id,
                chapter_id=cd.chapter_id,
                section_id=cd.section_id,
                content=cd.content,
                content_hash=cd.content_hash,
                token_count=cd.token_count,
                page_start=cd.page_start,
                page_end=cd.page_end,
            )
            session.add(row)
            rows.append(row)
    session.flush()
    return rows
