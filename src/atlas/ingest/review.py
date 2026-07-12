"""S5 human pass — concept review TUI (016 §T-16).

A Textual app that walks the ``pending`` concepts one at a time and records a
decision per keystroke: (a)ccept, (e)dit, (m)erge, (s)plit, (d)rop, (p)eek at the
supporting source chunk. Every decision writes ``review_status`` + ``reviewed_at``
and commits, so the pass is fully resumable — relaunching simply re-queries the
rows still marked ``pending``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Static

from atlas.core.models import Chunk, Concept, ConceptChunk


def _now() -> datetime:
    return datetime.now(UTC)


class EditScreen(ModalScreen):
    """Modal to edit a concept's title and description; Enter saves, Esc cancels."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, description: str):
        super().__init__()
        self._title = title
        self._description = description

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Edit concept — Enter to save, Esc to cancel", id="edit-help"),
            Input(value=self._title, id="edit-title"),
            Input(value=self._description, id="edit-desc"),
            id="edit-box",
        )

    def on_mount(self) -> None:
        self.query_one("#edit-title", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(
            (
                self.query_one("#edit-title", Input).value,
                self.query_one("#edit-desc", Input).value,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class ReviewApp(App):
    """Concept review pass over one course's pending concepts."""

    CSS = "#body { padding: 1 2; } #source { padding: 1 2; color: $text-muted; }"
    BINDINGS = [
        Binding("a", "accept", "Accept"),
        Binding("e", "edit", "Edit"),
        Binding("m", "merge", "Merge"),
        Binding("s", "split", "Split"),
        Binding("d", "drop", "Drop"),
        Binding("p", "peek", "Peek source"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, engine: Engine, course_id=None):
        super().__init__()
        self.session: Session = Session(engine)
        self.course_id = course_id
        self.pending: list[Concept] = []
        self.idx = 0
        self._last_kept: Concept | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="progress")
        yield Static("", id="body")
        yield Static("", id="source")
        yield Footer()

    def on_mount(self) -> None:
        self._reload()
        self._render()

    def on_unmount(self) -> None:
        self.session.close()

    def _reload(self) -> None:
        stmt = select(Concept).where(Concept.review_status == "pending")
        if self.course_id is not None:
            stmt = stmt.where(Concept.course_id == self.course_id)
        self.pending = list(self.session.scalars(stmt.order_by(Concept.title)))
        self.idx = 0

    def _current(self) -> Concept | None:
        return self.pending[self.idx] if self.idx < len(self.pending) else None

    def _render(self) -> None:
        c = self._current()
        prog = self.query_one("#progress", Static)
        body = self.query_one("#body", Static)
        self.query_one("#source", Static).update("")
        if c is None:
            prog.update("All concepts reviewed.")
            body.update("Press q to quit.")
            return
        prog.update(f"Concept {self.idx + 1} / {len(self.pending)}  ·  pending")
        body.update(
            f"[b]{c.title}[/b]  (difficulty {c.difficulty})\n\n"
            f"{c.description}\n\n"
            f"keywords: {', '.join(c.keywords) or '—'}\n"
            f"prerequisites: {', '.join(c.prerequisites) or '—'}\n"
            f"misconceptions: {', '.join(c.misconceptions) or '—'}\n"
            f"source pages: {', '.join(map(str, c.source_pages)) or '—'}"
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _resolve(self, concept: Concept, status: str) -> None:
        concept.review_status = status
        concept.reviewed_at = _now()
        self.session.commit()
        if status in ("accepted", "edited"):
            self._last_kept = concept
        self.idx += 1
        self._render()

    # ── actions ────────────────────────────────────────────────────────────
    def action_accept(self) -> None:
        c = self._current()
        if c:
            self._resolve(c, "accepted")

    def action_drop(self) -> None:
        c = self._current()
        if c:
            self._resolve(c, "dropped")

    def action_edit(self) -> None:
        c = self._current()
        if not c:
            return

        def _apply(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            c.title = result[0].strip() or c.title
            c.description = result[1].strip()
            self._resolve(c, "edited")

        self.push_screen(EditScreen(c.title, c.description), _apply)

    def action_merge(self) -> None:
        c = self._current()
        if not c:
            return
        if self._last_kept is not None:
            tgt = self._last_kept
            tgt.keywords = list(dict.fromkeys([*tgt.keywords, *c.keywords]))
            tgt.misconceptions = list(dict.fromkeys([*tgt.misconceptions, *c.misconceptions]))
            tgt.source_pages = sorted(set([*tgt.source_pages, *c.source_pages]))
        self._resolve(c, "merged")

    def action_split(self) -> None:
        c = self._current()
        if not c:
            return
        clone = Concept(
            course_id=c.course_id,
            chapter_id=c.chapter_id,
            title=f"{c.title} (part 2)",
            description=c.description,
            difficulty=c.difficulty,
            prerequisites=list(c.prerequisites),
            keywords=list(c.keywords),
            misconceptions=list(c.misconceptions),
            source_pages=list(c.source_pages),
            review_status="pending",
        )
        if c.embedding is not None:
            clone.embedding = c.embedding
        self.session.add(clone)
        self.session.flush()
        self.pending.append(clone)  # review the new half later this session
        self._resolve(c, "edited")

    def action_peek(self) -> None:
        c = self._current()
        if not c:
            return
        link = self.session.scalars(
            select(ConceptChunk)
            .where(ConceptChunk.concept_id == c.id)
            .order_by(ConceptChunk.cos.desc())
        ).first()
        src = self.query_one("#source", Static)
        if link is None:
            src.update("(no linked source chunk)")
            return
        chunk = self.session.get(Chunk, link.chunk_id)
        pages = f"pp. {chunk.page_start}-{chunk.page_end}" if chunk else ""
        text = (chunk.content[:1200] if chunk else "").strip()
        src.update(f"[source {pages}]\n{text}")
