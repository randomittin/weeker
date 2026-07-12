"""Study & flashcard TUI and the ``weeker study`` command group (015 §3, T-34/T-35).

Two Textual apps drive the learner loop:

* :class:`StudyApp` — one question per screen; ``a``–``d`` picks an answer, then
  ``1``/``2``/``3`` records confidence (guessing / unsure / sure) and grades the
  attempt through :func:`weeker.learn.attempt_service.record_attempt`; the
  explanation block shows after answering; ``p`` peeks the page-ranged source
  chunk; an end-of-session delta report closes the run.
* :class:`FlashApp` — flashcard drain: ``space`` flips, ``1``/``2``/``3`` grades
  and reschedules the card via the retention model.

``register`` attaches ``weeker study run`` (a live adaptive session), ``weeker
study verify`` (the synthetic M4 gate), ``weeker study replay`` (mastery
consistency), and ``weeker study flash``. Because ``cli/main.py`` marks the study
group ``no_args_is_help`` (and must not be edited), bare ``weeker study`` prints
the command list; ``weeker study run`` starts a session.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import typer
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from weeker.core.db import get_session, session_scope
from weeker.core.models import (
    Chunk,
    Concept,
    ConceptChunk,
    Course,
    Flashcard,
    Question,
)
from weeker.learn import retention, scheduler
from weeker.learn.attempt_service import record_attempt
from weeker.learn.replay import replay
from weeker.learn.verify_adaptive import verify_adaptive

CONFIDENCE_BY_KEY = {"1": "guessing", "2": "unsure", "3": "sure"}


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


def default_source_provider(db):
    """Return ``question -> page-ranged source text`` for the peek panel."""

    def provider(question: Question) -> str:
        rows = (
            db.query(Chunk)
            .join(ConceptChunk, ConceptChunk.chunk_id == Chunk.id)
            .filter(ConceptChunk.concept_id == question.concept_id)
            .order_by(ConceptChunk.cos.desc())
            .limit(1)
            .all()
        )
        if not rows:
            concept = db.get(Concept, question.concept_id)
            pages = getattr(concept, "source_pages", None) or []
            return f"(no mapped chunk; concept source pages: {pages})"
        chunk = rows[0]
        head = f"p.{chunk.page_start}-{chunk.page_end}"
        body = chunk.content[:800]
        return f"[source {head}]\n{body}"

    return provider


class StudyApp(App):
    """One-question-per-screen adaptive study session."""

    CSS = """
    Screen { layout: vertical; }
    #progress { color: $accent; height: 1; }
    #stem { padding: 1 0; text-style: bold; }
    #options { padding: 0 2; }
    #prompt { color: $warning; height: auto; }
    #feedback { padding: 1 0; }
    #source { color: $text-muted; padding: 1 0; }
    """

    def __init__(
        self,
        questions,
        *,
        user_id: str,
        course_id: object,
        session_id: object | None = None,
        db=None,
        source_provider=None,
        now: datetime | None = None,
    ):
        super().__init__()
        self.questions = list(questions)
        self.user_id = user_id
        self.course_id = course_id
        self.session_id = session_id
        self.now = _now(now)
        self._own_db = db is None
        self._db = db or get_session()
        self.source_provider = source_provider or default_source_provider(self._db)
        self.index = 0
        self.phase = "answer" if self.questions else "done"
        self.pending_key: str | None = None
        self.show_source = False
        self.outcomes: list = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="progress")
            yield Static(id="stem")
            yield Static(id="options")
            yield Static(id="prompt")
            yield Static(id="feedback")
            yield Static(id="source")

    def on_mount(self) -> None:
        self._render()

    @property
    def current(self) -> Question | None:
        if 0 <= self.index < len(self.questions):
            return self.questions[self.index]
        return None

    def _set(self, wid: str, text: str) -> None:
        self.query_one(f"#{wid}", Static).update(text)

    def _render(self) -> None:
        if self.phase == "done":
            self._set("progress", "")
            self._set("stem", "")
            self._set("options", "")
            self._set("prompt", "")
            self._set("feedback", self._delta_report_text())
            self._set("source", "")
            return

        q = self.current
        self._set("progress", f"Question {self.index + 1}/{len(self.questions)}")
        self._set("stem", q.stem)
        opt_lines = []
        for opt in q.options or []:
            key = opt.get("key", "?")
            marker = ">" if self.pending_key == key else " "
            opt_lines.append(f"{marker} {key}) {opt.get('text', '')}")
        self._set("options", "\n".join(opt_lines))

        if self.phase == "answer":
            if self.pending_key is None:
                self._set("prompt", "Press a-d to answer.")
            else:
                self._set(
                    "prompt",
                    f"Answer {self.pending_key}. Confidence: 1 guessing / 2 unsure / 3 sure.",
                )
            self._set("feedback", "")
        elif self.phase == "explain":
            q_, out = self.outcomes[-1]
            verdict = "Correct!" if out.correct else f"Incorrect — correct answer: {q.correct_key}"
            self._set("prompt", "Press n / space for the next question.")
            fb = [verdict, "", q.explanation or ""]
            fb.append(
                f"Δθ {out.theta_after - out.theta_before:+.3f}  "
                f"mastery {out.mastery_before:.2f}→{out.mastery_after:.2f}"
            )
            if out.misconception_fired:
                fb.append("(misconception flagged — a flashcard was added)")
            if out.misconception_cleared:
                fb.append("(misconception cleared)")
            self._set("feedback", "\n".join(fb))

        self._set("source", self.source_provider(q) if self.show_source else "")

    def _delta_report_text(self) -> str:
        n = len(self.outcomes)
        if n == 0:
            return "No questions this session. Press q to exit."
        correct = sum(1 for _, o in self.outcomes if o.correct)
        lines = [f"Session complete — {correct}/{n} correct", ""]
        agg: dict = {}
        for _, o in self.outcomes:
            d = agg.setdefault(o.concept_id, [0.0, 0])
            d[0] += o.theta_after - o.theta_before
            d[1] += 1
        for cid, (dth, cnt) in agg.items():
            lines.append(f"concept {cid}: Δθ {dth:+.3f} over {cnt} attempt(s)")
        lines.append("")
        lines.append("Press q to exit.")
        return "\n".join(lines)

    def _commit(self, confidence: str) -> None:
        q = self.current
        out = record_attempt(
            user_id=self.user_id,
            question_id=q.id,
            chosen_key=self.pending_key,
            confidence=confidence,
            session_id=self.session_id,
            now=self.now,
            db=self._db,
        )
        self.outcomes.append((q, out))
        self.phase = "explain"
        self._render()

    def _advance(self) -> None:
        self.index += 1
        self.pending_key = None
        self.show_source = False
        if self.index >= len(self.questions):
            self.phase = "done"
            if self._own_db:
                self._db.commit()
        else:
            self.phase = "answer"
        self._render()

    def on_key(self, event) -> None:
        key = event.key
        if self.phase == "done":
            if key in ("q", "escape", "enter", "space"):
                self.exit()
            return
        if key == "p":
            self.show_source = not self.show_source
            self._render()
            return
        if self.phase == "answer":
            if key in ("a", "b", "c", "d"):
                self.pending_key = key.upper()
                self._render()
            elif key in CONFIDENCE_BY_KEY and self.pending_key is not None:
                self._commit(CONFIDENCE_BY_KEY[key])
        elif self.phase == "explain" and key in ("n", "space", "enter"):
            self._advance()

    def on_unmount(self) -> None:
        if self._own_db:
            try:
                self._db.commit()
            finally:
                self._db.close()


class FlashApp(App):
    """Flashcard drain: flip and grade, rescheduling via the retention model."""

    CSS = """
    Screen { layout: vertical; }
    #progress { color: $accent; height: 1; }
    #front { padding: 1 0; text-style: bold; }
    #back { padding: 1 0; }
    #prompt { color: $warning; }
    """

    GRADE = {
        "1": (False, "sure"),
        "2": (True, "unsure"),
        "3": (True, "sure"),
    }

    def __init__(self, cards, *, db=None, now: datetime | None = None):
        super().__init__()
        self.cards = list(cards)
        self.now = _now(now)
        self._own_db = db is None
        self._db = db or get_session()
        self.index = 0
        self.flipped = False
        self.graded = 0
        self.phase = "front" if self.cards else "done"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="progress")
            yield Static(id="front")
            yield Static(id="back")
            yield Static(id="prompt")

    def on_mount(self) -> None:
        self._render()

    def _set(self, wid: str, text: str) -> None:
        self.query_one(f"#{wid}", Static).update(text)

    @property
    def current(self) -> Flashcard | None:
        if 0 <= self.index < len(self.cards):
            return self.cards[self.index]
        return None

    def _render(self) -> None:
        if self.phase == "done":
            self._set("progress", "")
            self._set("front", f"Flashcards complete — {self.graded} graded.")
            self._set("back", "")
            self._set("prompt", "Press q to exit.")
            return
        c = self.current
        self._set("progress", f"Card {self.index + 1}/{len(self.cards)}")
        self._set("front", c.front)
        if self.flipped:
            self._set("back", c.back)
            self._set("prompt", "Grade: 1 again / 2 hard / 3 good.")
        else:
            self._set("back", "")
            self._set("prompt", "Press space to flip.")

    def _grade(self, key: str) -> None:
        c = self.current
        correct, confidence = self.GRADE[key]
        proxy_mastery = 1.0 if correct else 0.0
        new_stability = retention.stability_update(
            float(c.stability), proxy_mastery, correct, confidence
        )
        c.stability = new_stability
        c.due_at = self.now + timedelta(days=retention.days_until_due(new_stability))
        self._db.add(c)
        self._db.flush()
        self.graded += 1
        self.index += 1
        self.flipped = False
        if self.index >= len(self.cards):
            self.phase = "done"
            if self._own_db:
                self._db.commit()
        self._render()

    def on_key(self, event) -> None:
        key = event.key
        if self.phase == "done":
            if key in ("q", "escape", "enter"):
                self.exit()
            return
        if not self.flipped and key == "space":
            self.flipped = True
            self._render()
        elif self.flipped and key in self.GRADE:
            self._grade(key)

    def on_unmount(self) -> None:
        if self._own_db:
            try:
                self._db.commit()
            finally:
                self._db.close()


# ── CLI ──────────────────────────────────────────────────────────────────────
def _resolve_course(db, course: str | None) -> Course | None:
    from sqlalchemy import select

    stmt = select(Course)
    if course:
        stmt = stmt.where((Course.slug == course) | (Course.title == course))
    return db.scalars(stmt).first()


def _run_study(user: str, course: str | None, count: int | None) -> None:
    with session_scope() as db:
        c = _resolve_course(db, course)
        if c is None:
            typer.echo("No course found — ingest a course first.")
            raise typer.Exit(1)
        rng = random.Random()
        questions = scheduler.next_session(db, user, c.id, datetime.now(UTC), rng)
        if count is not None:
            questions = questions[:count]
        if not questions:
            typer.echo("No active questions available for this course yet.")
            raise typer.Exit(1)
        StudyApp(questions, user_id=user, course_id=c.id, db=db).run()


def _run_flash(user: str, course: str | None) -> None:
    from sqlalchemy import select

    with session_scope() as db:
        stmt = select(Flashcard).where(Flashcard.user_id == user)
        cards = db.scalars(stmt.order_by(Flashcard.due_at.is_(None).desc(), Flashcard.due_at)).all()
        if not cards:
            typer.echo("No flashcards due.")
            raise typer.Exit(0)
        FlashApp(cards, db=db).run()


def register(group_app: typer.Typer) -> None:
    """Attach the study/flash commands to the ``weeker study`` group."""

    @group_app.command("run")
    def run(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
        count: int = typer.Option(None, "--count", "-n", help="Cap questions this session."),
    ) -> None:
        """Start an adaptive study session."""
        _run_study(user, course, count)

    @group_app.command("flash")
    def flash(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Drain due flashcards."""
        _run_flash(user, course)

    @group_app.command("verify")
    def verify() -> None:
        """Synthetic 200-attempt adaptive verification (M4 gate)."""
        report = verify_adaptive()
        for line in report.checks:
            typer.echo(line)
        typer.echo(f"correlation={report.correlation:.3f} ok={report.ok}")
        raise typer.Exit(0 if report.ok else 1)

    @group_app.command("replay")
    def replay_cmd(
        user: str = typer.Option(None, "--user", "-u"),
    ) -> None:
        """Replay the attempt log and diff recomputed mastery."""
        diff = replay(user_id=user)
        typer.echo(
            f"concepts={diff.concepts_checked} "
            f"max_theta_diff={diff.max_theta_diff:.2e} "
            f"max_stability_diff={diff.max_stability_diff:.2e} ok={diff.ok}"
        )
        for m in diff.mismatches:
            typer.echo(f"  mismatch: {m}")
        raise typer.Exit(0 if diff.ok else 1)
