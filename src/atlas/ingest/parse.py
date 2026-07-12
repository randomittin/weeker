"""S1 — PyMuPDF page extraction (016 §T-11).

Turns a source PDF into clean per-page text: de-hyphenates line-broken words,
strips repeated running headers/footers (any short line recurring on at least
``HEADER_FOOTER_PAGE_SHARE`` of pages, with digits normalised so page numbers
collapse), serialises tables to pipe rows, and OCRs any page whose extracted
text falls below ``OCR_MIN_CHARS``. Coverage and silent-empty pages are reported
for the parse gate.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from sqlalchemy.orm import Session

from atlas.core.config import HEADER_FOOTER_PAGE_SHARE, OCR_DPI, OCR_MIN_CHARS
from atlas.core.models import Page, Source

# Band of lines at the top/bottom of a page eligible to be running chrome.
# One line each end: running headers/footers and page numbers are single lines,
# and a wider band risks eating short body lines that repeat by coincidence.
_CHROME_BAND = 1
_HYPHEN_RE = re.compile(r"([A-Za-z])-\n([a-z])")
_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


@dataclass
class ParsedPage:
    page_number: int
    text: str
    char_count: int
    ocr_used: bool = False


@dataclass
class ParseResult:
    source_filename: str
    pages: list[ParsedPage]
    headers: set[str] = field(default_factory=set)
    footers: set[str] = field(default_factory=set)

    @property
    def empty_pages(self) -> list[int]:
        # Silent-empty: no text AND OCR was not used to try to recover it.
        return [p.page_number for p in self.pages if not p.text.strip() and not p.ocr_used]

    @property
    def coverage(self) -> float:
        if not self.pages:
            return 0.0
        good = sum(1 for p in self.pages if len(p.text) >= OCR_MIN_CHARS)
        return good / len(self.pages)


def dehyphenate(text: str) -> str:
    """Join words split by a hyphen at a line break (``regula-\\ntory`` → ``regulatory``)."""
    return _HYPHEN_RE.sub(r"\1\2", text)


def _normalize(line: str) -> str:
    """Canonical form for chrome detection: lowercased, digits → ``#``, ws collapsed."""
    return _WS_RE.sub(" ", _DIGITS_RE.sub("#", line.strip().lower())).strip()


def detect_header_footer(
    pages_lines: list[list[str]], share: float = HEADER_FOOTER_PAGE_SHARE
) -> tuple[set[str], set[str]]:
    """Return the normalised header and footer lines recurring across pages."""
    n = len(pages_lines)
    if n == 0:
        return set(), set()
    top: Counter[str] = Counter()
    bottom: Counter[str] = Counter()
    for lines in pages_lines:
        norm = [_normalize(ln) for ln in lines if ln.strip()]
        for ln in dict.fromkeys(norm[:_CHROME_BAND]):
            top[ln] += 1
        for ln in dict.fromkeys(norm[-_CHROME_BAND:]):
            bottom[ln] += 1
    threshold = max(2, int(share * n) if share * n >= 1 else 2)
    headers = {ln for ln, c in top.items() if ln and c >= threshold}
    footers = {ln for ln, c in bottom.items() if ln and c >= threshold}
    return headers, footers


def serialize_table(rows: list[list[str | None]]) -> str:
    """Serialise an extracted table into ``cell | cell`` lines (empty cells blank)."""
    out = []
    for row in rows:
        out.append(" | ".join((c or "").strip() for c in row))
    return "\n".join(out)


def _default_ocr(page: fitz.Page) -> str:
    """Real OCR via PyMuPDF's Tesseract bridge (requires Tesseract installed)."""
    try:
        tp = page.get_textpage_ocr(flags=0, dpi=OCR_DPI, full=True)
        return page.get_text(textpage=tp)
    except Exception as exc:  # Tesseract absent / OCR failure — surface clearly
        raise RuntimeError(
            f"OCR required for page {page.number} but unavailable: {exc}"
        ) from exc


def _page_lines(page: fitz.Page) -> list[str]:
    """Ordered text lines of a page (blocks → lines → spans), top-to-bottom."""
    data = page.get_text("dict")
    lines: list[str] = []
    for block in sorted(data.get("blocks", []), key=lambda b: b.get("bbox", [0, 0])[1]):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = "".join(sp.get("text", "") for sp in line.get("spans", []))
            if spans.strip():
                lines.append(spans)
    return lines


def _append_tables(page: fitz.Page, body: str) -> str:
    try:
        finder = page.find_tables()
    except Exception:
        return body
    tables = getattr(finder, "tables", [])
    for tbl in tables:
        try:
            rows = tbl.extract()
        except Exception:
            continue
        if rows:
            body = f"{body}\n{serialize_table(rows)}"
    return body


def parse_source(
    path: str | Path,
    page_offset: int = 0,
    ocr_fn: Callable[[fitz.Page], str] | None = None,
    header_footer_share: float = HEADER_FOOTER_PAGE_SHARE,
) -> ParseResult:
    """Parse ``path`` into cleaned :class:`ParsedPage` records with coverage stats."""
    path = Path(path)
    ocr_fn = ocr_fn or _default_ocr
    doc = fitz.open(str(path))
    try:
        raw_lines = [_page_lines(page) for page in doc]
        headers, footers = detect_header_footer(raw_lines, header_footer_share)

        pages: list[ParsedPage] = []
        for idx, page in enumerate(doc):
            lines = raw_lines[idx]
            kept: list[str] = []
            last = len(lines) - 1
            for i, ln in enumerate(lines):
                norm = _normalize(ln)
                in_top = i < _CHROME_BAND
                in_bottom = i > last - _CHROME_BAND
                if in_top and norm in headers:
                    continue
                if in_bottom and norm in footers:
                    continue
                kept.append(ln)
            body = dehyphenate("\n".join(kept))
            body = _append_tables(page, body).strip()

            ocr_used = False
            if len(body) < OCR_MIN_CHARS:
                body = (ocr_fn(page) or "").strip()
                ocr_used = True
            pages.append(
                ParsedPage(
                    page_number=idx + 1 + page_offset,
                    text=body,
                    char_count=len(body),
                    ocr_used=ocr_used,
                )
            )
        return ParseResult(path.name, pages, headers, footers)
    finally:
        doc.close()


def persist_pages(session: Session, source: Source, result: ParseResult) -> int:
    """Write parsed pages for ``source`` (replacing any prior rows). Returns count."""
    session.query(Page).filter(Page.source_id == source.id).delete()
    session.flush()
    for p in result.pages:
        session.add(
            Page(
                source_id=source.id,
                page_number=p.page_number,
                text=p.text,
                char_count=p.char_count,
                ocr_used=p.ocr_used,
            )
        )
    source.page_count = len(result.pages)
    session.flush()
    return len(result.pages)
