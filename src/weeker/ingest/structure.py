"""S2 — chapter/section structure via a strategy chain (016 §T-12).

Order of attempts, first one that satisfies :func:`check_structure` wins:

1. **text headings** — explicit in-body ``Chapter N: Title`` / ``Unit N. Title``
   markers on their own line (the strongest, keyless signal; robust to
   degenerate bookmarks);
2. **outline** — the PDF's own bookmarks (``doc.get_toc``);
3. **font heuristic** — lines whose span size stands out from the body size,
   clustered into two heading levels;
4. **LLM** — ``prompts/structure.txt`` over the first pages + font candidates.

The LLM stage is skipped when no key is configured and no explicit ``transport``
is supplied: the best heuristic structure is returned instead so keyless
ingestion never crashes. The structure gate (011 §S2) is pure code: chapters
must not overlap, must cover ≥ 95% of pages, and depth is capped at 2 by the
data model itself.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from sqlalchemy.orm import Session

from weeker.core import config
from weeker.core.llm import Transport, complete
from weeker.core.models import Chapter, Course, Section
from weeker.ingest.textutil import load_prompt, render_prompt

STRUCTURE_COVERAGE_MIN = 0.95
_FONT_HEADING_RATIO = 1.15  # a heading span is ≥ this × the modal body size
# An explicit chapter/unit heading: "Chapter 3: Mutual Funds", "Unit 12. Ethics".
_CHAPTER_HEADING = re.compile(r"^(chapter|unit)\s+(\d+)\s*[:.\-]\s*(.+)$", re.IGNORECASE)


class StructureError(RuntimeError):
    """Raised when no structure can be recovered and the LLM stage is unavailable."""

_STRUCTURE_SCHEMA = {
    "type": "object",
    "required": ["chapters"],
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ordinal", "title", "page_start", "page_end"],
                "properties": {
                    "ordinal": {"type": "integer"},
                    "title": {"type": "string"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                    "sections": {"type": "array"},
                },
            },
        }
    },
}


@dataclass
class SectionNode:
    ordinal: int
    title: str
    page_start: int
    page_end: int


@dataclass
class ChapterNode:
    ordinal: int
    title: str
    page_start: int
    page_end: int
    sections: list[SectionNode] = field(default_factory=list)


@dataclass
class StructureResult:
    chapters: list[ChapterNode]
    strategy: str
    page_offset: int = 0


def check_structure(spans: list[tuple[int, int]], page_count: int) -> list[str]:
    """Validate chapter page spans: ordered, non-overlapping, ≥95% coverage."""
    failures: list[str] = []
    if not spans:
        return ["no chapters"]
    ordered = sorted(spans, key=lambda s: s[0])
    for start, end in ordered:
        if start > end:
            failures.append(f"chapter span inverted: {start}>{end}")
        if start < 1:
            failures.append(f"chapter start {start} < 1")
    for (s1, e1), (s2, e2) in zip(ordered, ordered[1:], strict=False):
        if s2 <= e1:
            failures.append(f"overlapping chapters: [{s1},{e1}] & [{s2},{e2}]")
    if page_count > 0:
        covered: set[int] = set()
        for start, end in ordered:
            covered |= set(range(max(1, start), min(page_count, end) + 1))
        coverage = len(covered) / page_count
        if coverage < STRUCTURE_COVERAGE_MIN:
            failures.append(f"structure coverage {coverage:.3f} < {STRUCTURE_COVERAGE_MIN}")
    return failures


def _spans(chapters: list[ChapterNode]) -> list[tuple[int, int]]:
    return [(c.page_start, c.page_end) for c in chapters]


def _passes(chapters: list[ChapterNode], page_count: int) -> bool:
    return chapters != [] and check_structure(_spans(chapters), page_count) == []


def from_text_headings(doc: fitz.Document) -> StructureResult | None:
    """Build chapters from explicit ``Chapter N: Title`` heading lines.

    Scans each page for a line matching :data:`_CHAPTER_HEADING` that stands out
    from body text (font larger than the modal body size) and is not a
    table-of-contents dotted-leader entry. Keeps the first occurrence per chapter
    number (the title page, after the TOC). ``None`` if fewer than two are found.
    """
    sizes: Counter[int] = Counter()
    lines: list[tuple[str, int, float]] = []
    for pno, page in enumerate(doc, start=1):
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(sp.get("text", "") for sp in spans).strip()
                if not text:
                    continue
                size = max(float(sp.get("size", 0)) for sp in spans)
                sizes[round(size)] += 1
                lines.append((text, pno, size))
    if not sizes:
        return None
    body = float(sizes.most_common(1)[0][0])
    found: dict[int, tuple[int, str]] = {}
    for text, pno, size in lines:
        if text.count(".") >= 4:  # skip TOC dotted-leader entries
            continue
        if size <= body:  # a heading stands out from body text
            continue
        m = _CHAPTER_HEADING.match(text)
        if not m:
            continue
        num = int(m.group(2))
        if num not in found:
            title = f"{m.group(1).title()} {num}: {m.group(3).strip()}"
            found[num] = (pno, title)
    if len(found) < 2:
        return None
    ordered = sorted(found.items(), key=lambda kv: kv[1][0])  # by page
    page_count = doc.page_count
    chapters: list[ChapterNode] = []
    for i, (_num, (page, title)) in enumerate(ordered):
        end = (ordered[i + 1][1][0] - 1) if i + 1 < len(ordered) else page_count
        chapters.append(ChapterNode(i + 1, title, max(1, page), max(page, end)))
    chapters[0].page_start = 1  # absorb front matter (cover, TOC) into chapter 1
    return StructureResult(chapters, "text")


def from_outline(doc: fitz.Document) -> StructureResult | None:
    """Build structure from PDF bookmarks; None if there are no level-1 entries."""
    toc = doc.get_toc(simple=True)
    if not toc:
        return None
    page_count = doc.page_count
    chapters: list[ChapterNode] = []
    # First pass: level-1 entries become chapters.
    lvl1 = [(lvl, title, page) for lvl, title, page in toc if lvl == 1]
    if not lvl1:
        return None
    for i, (_lvl, title, page) in enumerate(lvl1):
        start = max(1, page)
        end = (lvl1[i + 1][2] - 1) if i + 1 < len(lvl1) else page_count
        chapters.append(ChapterNode(i + 1, title.strip(), start, max(start, end)))
    # Second pass: level-2 entries become sections of their enclosing chapter.
    for lvl, title, page in toc:
        if lvl != 2:
            continue
        for ch in chapters:
            if ch.page_start <= page <= ch.page_end:
                ch.sections.append(
                    SectionNode(len(ch.sections) + 1, title.strip(), max(1, page), ch.page_end)
                )
                break
    for ch in chapters:
        for j, sec in enumerate(ch.sections):
            sec.page_end = (
                ch.sections[j + 1].page_start - 1 if j + 1 < len(ch.sections) else ch.page_end
            )
            sec.page_end = max(sec.page_start, sec.page_end)
    return StructureResult(chapters, "outline")


def _font_headings(doc: fitz.Document) -> tuple[list[tuple[str, int, float]], float]:
    """Return (heading candidates as (text,page,size)) and the modal body size."""
    sizes: Counter[int] = Counter()
    lines: list[tuple[str, int, float]] = []
    for pno, page in enumerate(doc, start=1):
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(sp.get("text", "") for sp in spans).strip()
                if not text:
                    continue
                size = max(float(sp.get("size", 0)) for sp in spans)
                sizes[round(size)] += 1
                lines.append((text, pno, size))
    if not sizes:
        return [], 0.0
    body = float(sizes.most_common(1)[0][0])
    candidates = [(t, p, s) for (t, p, s) in lines if s >= body * _FONT_HEADING_RATIO]
    return candidates, body


def from_font(doc: fitz.Document) -> StructureResult | None:
    """Cluster stand-out font sizes into two heading levels; None if none found."""
    candidates, body = _font_headings(doc)
    if not candidates:
        return None
    heading_sizes = sorted({round(s) for _t, _p, s in candidates}, reverse=True)
    chapter_size = heading_sizes[0]
    page_count = doc.page_count
    chapter_heads = [(t, p, s) for (t, p, s) in candidates if round(s) == chapter_size]
    if not chapter_heads:
        return None
    chapter_heads.sort(key=lambda x: x[1])
    chapters: list[ChapterNode] = []
    for i, (title, page, _s) in enumerate(chapter_heads):
        start = page
        end = (chapter_heads[i + 1][1] - 1) if i + 1 < len(chapter_heads) else page_count
        chapters.append(ChapterNode(i + 1, title, max(1, start), max(start, end)))
    # Sub-headings (any smaller heading size) → sections.
    for title, page, s in candidates:
        if round(s) == chapter_size:
            continue
        for ch in chapters:
            if ch.page_start <= page <= ch.page_end:
                ch.sections.append(SectionNode(len(ch.sections) + 1, title, page, ch.page_end))
                break
    if chapters:
        chapters[0].page_start = 1  # absorb any front matter into chapter 1
    return StructureResult(chapters, "font")


def from_llm(
    doc: fitz.Document,
    source_filename: str,
    transport: Transport | None = None,
    model: str = "structure",
) -> StructureResult:
    """Reconstruct structure with the LLM using ``prompts/structure.txt``."""
    system, user_t = load_prompt("structure")
    first = "\n\n".join(
        f"[page {i + 1}]\n{doc[i].get_text()}" for i in range(min(15, doc.page_count))
    )
    candidates, _body = _font_headings(doc)
    cand_text = "\n".join(f"- {t!r} (page {p})" for t, p, _s in candidates[:80]) or "(none)"
    user = render_prompt(
        user_t,
        source_filename=source_filename,
        page_count=doc.page_count,
        first_15_pages_text=first,
        heading_candidates=cand_text,
    )
    data = complete(model, system, user, json_schema=_STRUCTURE_SCHEMA, transport=transport)
    assert isinstance(data, dict)
    page_offset = int(data.get("page_offset", 0) or 0)
    chapters: list[ChapterNode] = []
    for i, ch in enumerate(data["chapters"]):
        node = ChapterNode(
            ordinal=int(ch.get("ordinal", i + 1)),
            title=str(ch["title"]).strip(),
            page_start=int(ch["page_start"]),
            page_end=int(ch["page_end"]),
        )
        for j, sec in enumerate(ch.get("sections", []) or []):
            node.sections.append(
                SectionNode(
                    ordinal=int(sec.get("ordinal", j + 1)),
                    title=str(sec["title"]).strip(),
                    page_start=int(sec["page_start"]),
                    page_end=int(sec["page_end"]),
                )
            )
        chapters.append(node)
    return StructureResult(chapters, "llm", page_offset)


def build_structure(
    path: str | Path,
    source_filename: str | None = None,
    transport: Transport | None = None,
) -> StructureResult:
    """Run the strategy chain; return the first structure that passes the gate."""
    path = Path(path)
    doc = fitz.open(str(path))
    try:
        page_count = doc.page_count
        best: StructureResult | None = None
        for strat in (from_text_headings, from_outline, from_font):
            result = strat(doc)
            if result and result.chapters:
                if _passes(result.chapters, page_count):
                    return result
                best = best or result
        # No heuristic passed the gate. Use the LLM stage when it is available;
        # otherwise fall back to the best heuristic so keyless ingestion runs.
        if transport is None and not config.llm_key_present():
            if best is not None:
                return best
            raise StructureError(
                "no structure recovered and no LLM configured — set WEEKER_LLM_API_KEY "
                "or supply a PDF with bookmarks or explicit chapter headings"
            )
        return from_llm(doc, source_filename or path.name, transport)
    finally:
        doc.close()


def persist_structure(
    session: Session, course: Course, result: StructureResult
) -> tuple[list[Chapter], list[Section]]:
    """Persist chapters/sections for ``course`` (replacing any prior structure)."""
    prior = list(session.query(Chapter).filter(Chapter.course_id == course.id))
    for ch in prior:
        session.query(Section).filter(Section.chapter_id == ch.id).delete()
    session.query(Chapter).filter(Chapter.course_id == course.id).delete()
    session.flush()

    chapters: list[Chapter] = []
    sections: list[Section] = []
    for node in result.chapters:
        row = Chapter(
            course_id=course.id,
            ordinal=node.ordinal,
            title=node.title,
            page_start=node.page_start,
            page_end=node.page_end,
        )
        session.add(row)
        session.flush()
        chapters.append(row)
        for snode in node.sections:
            srow = Section(
                chapter_id=row.id,
                ordinal=snode.ordinal,
                title=snode.title,
                page_start=snode.page_start,
                page_end=snode.page_end,
            )
            session.add(srow)
            sections.append(srow)
    session.flush()
    return chapters, sections
