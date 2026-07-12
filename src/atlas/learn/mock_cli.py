"""The ``atlas mock`` command group + its Textual exam UIs (016 §T-36/T-40/T-42).

``register`` attaches the real mock commands to the group created in
``cli/main.py`` (which imports this module and calls ``register`` — no edit to
that file):

* ``atlas mock run`` — a timed standalone mock (T-40): one question per screen,
  ``a``–``d`` answer, ``1``/``2``/``3`` confidence, **no feedback until submit**,
  ``escape`` to pause (state persisted to resume), post-mock review ordered
  confident-wrong first;
* ``atlas mock resume`` — resume the latest paused mock exactly where it stopped;
* ``atlas mock full`` — the full-pattern mock (T-42): 90 standalone + 6 one-mark
  + 3 two-mark caselets, with the per-section prediction + calibration + skip
  policy strategy report printed on the summary;
* ``atlas mock diagnose`` — the Day-0 diagnostic (T-36). Spec names this
  ``atlas diagnose``; ``cli/main.py`` is frozen with no ``diagnose`` group, so it
  is wired here as ``atlas mock diagnose`` (documented deviation, same pattern as
  ``atlas study flash``).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import typer
from sqlalchemy import select
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from atlas.core.db import session_scope
from atlas.core.models import Course, Question
from atlas.learn import diagnose, full, mock
from atlas.learn.status import render_status

CONFIDENCE_BY_KEY = {"1": "guessing", "2": "unsure", "3": "sure"}


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


def _resolve_course(db, course: str | None) -> Course | None:
    stmt = select(Course)
    if course:
        stmt = stmt.where((Course.slug == course) | (Course.title == course))
    return db.scalars(stmt.order_by(Course.created_at.desc())).first()


def _questions_map(db, items: list[mock.MockItem]) -> dict:
    ids = [it.question_id for it in items]
    return {q.id: q for q in db.scalars(select(Question).where(Question.id.in_(ids))).all()}


# ── timed mock UI (no feedback until submit) ────────────────────────────────────
class MockApp(App):
    """One-question-per-screen timed exam; grades only on submit."""

    CSS = """
    Screen { layout: vertical; }
    #clock { color: $accent; height: 1; }
    #progress { color: $accent; height: 1; }
    #stem { padding: 1 0; text-style: bold; }
    #options { padding: 0 2; }
    #prompt { color: $warning; height: auto; }
    #summary { padding: 1 0; }
    """

    def __init__(
        self,
        items,
        *,
        mock_row,
        db,
        passing_marks: float,
        neg_fraction: float,
        duration_minutes: float,
        now: datetime | None = None,
        start_index: int = 0,
        elapsed_seconds: float = 0.0,
        answers: dict | None = None,
        record_attempts: bool = True,
    ):
        super().__init__()
        self.items = list(items)
        self.mock_row = mock_row
        self._db = db
        self.passing_marks = passing_marks
        self.neg_fraction = neg_fraction
        self.duration_seconds = duration_minutes * 60.0
        self.now = _now(now)
        self.deadline = self.now + timedelta(seconds=self.duration_seconds - elapsed_seconds)
        self.answers = dict(answers or {})
        self.index = start_index
        self.record_attempts = record_attempts
        self.questions = _questions_map(db, self.items)
        self.pending_key: str | None = None
        self.phase = "answer" if self.items else "done"
        self.result: mock.MockResult | None = None
        self.review_index = 0
        self.paused = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="clock")
            yield Static(id="progress")
            yield Static(id="stem")
            yield Static(id="options")
            yield Static(id="prompt")
            yield Static(id="summary")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)
        self._render()

    def _set(self, wid: str, text: str) -> None:
        self.query_one(f"#{wid}", Static).update(text)

    def _remaining(self) -> float:
        return max(0.0, (self.deadline - datetime.now(UTC)).total_seconds())

    def _elapsed(self) -> float:
        return max(0.0, self.duration_seconds - self._remaining())

    def _tick(self) -> None:
        if self.phase != "answer":
            return
        if self._remaining() <= 0:
            self._submit()
        else:
            m, s = divmod(int(self._remaining()), 60)
            self._set("clock", f"Time left: {m:02d}:{s:02d}")

    @property
    def current(self) -> mock.MockItem | None:
        if 0 <= self.index < len(self.items):
            return self.items[self.index]
        return None

    def _render(self) -> None:
        if self.phase == "done":
            self._render_summary()
            return
        if self.phase == "review":
            self._render_review()
            return
        it = self.current
        q = self.questions.get(it.question_id)
        m, s = divmod(int(self._remaining()), 60)
        self._set("clock", f"Time left: {m:02d}:{s:02d}")
        answered = sum(1 for a in self.answers.values() if a.chosen_key is not None)
        self._set(
            "progress",
            f"Q {self.index + 1}/{len(self.items)}  ({answered} answered)  [marks {it.marks:g}]",
        )
        self._set("stem", q.stem if q else "(question unavailable)")
        cur = self.answers.get(it.question_id)
        chosen = cur.chosen_key if cur else self.pending_key
        lines = []
        for opt in (q.options if q else []) or []:
            key = opt.get("key", "?")
            marker = ">" if chosen == key else " "
            lines.append(f"{marker} {key}) {opt.get('text', '')}")
        self._set("options", "\n".join(lines))
        conf = cur.confidence if cur else None
        self._set(
            "prompt",
            "a-d answer / 1-2-3 confidence / n next / left back / s submit / esc pause"
            f"   [you: {chosen or '-'} / {conf or '-'}]",
        )
        self._set("summary", "")

    def _render_summary(self) -> None:
        r = self.result
        self._set("clock", "")
        self._set("progress", "Mock submitted")
        self._set("stem", "")
        self._set("options", "")
        self._set("prompt", "Press v to review, q to exit.")
        if r is None:
            self._set("summary", "")
            return
        verdict = "PASS" if r.passed else "FAIL"
        self._set(
            "summary",
            f"Score {r.raw_score:.2f}/{r.total_marks:.0f}  ({verdict} at {r.passing_marks:.0f})\n"
            f"Answered {r.answered}, skipped {r.skipped}.",
        )

    def _render_review(self) -> None:
        r = self.result
        review = r.review if r else []
        if not review:
            self.phase = "done"
            self._render()
            return
        self.review_index = max(0, min(self.review_index, len(review) - 1))
        it = review[self.review_index]
        q = self.questions.get(it.question_id)
        a = self.answers.get(it.question_id)
        chosen = a.chosen_key if a else None
        conf = a.confidence if a else None
        correct = chosen == it.correct_key if chosen is not None else None
        tag = "SKIPPED" if chosen is None else ("CORRECT" if correct else "WRONG")
        if chosen is not None and not correct and conf == "sure":
            tag = "CONFIDENT-WRONG"
        self._set("clock", "")
        self._set("progress", f"Review {self.review_index + 1}/{len(review)}  [{tag}]")
        self._set("stem", q.stem if q else "")
        lines = []
        for opt in (q.options if q else []) or []:
            key = opt.get("key", "?")
            mark = " (key)" if key == it.correct_key else ""
            if key == chosen:
                mark += " <- you"
            lines.append(f"  {key}) {opt.get('text', '')}{mark}")
        self._set("options", "\n".join(lines))
        self._set("summary", (q.explanation if q else "") or "")
        self._set("prompt", "n next / left back / q exit")

    def _commit_answer(self, confidence: str) -> None:
        it = self.current
        if it is None or self.pending_key is None:
            return
        self.answers[it.question_id] = mock.MockAnswer(
            chosen_key=self.pending_key, confidence=confidence
        )
        self._render()

    def _advance(self, step: int) -> None:
        self.index = max(0, min(self.index + step, len(self.items) - 1))
        cur = self.answers.get(self.current.question_id) if self.current else None
        self.pending_key = cur.chosen_key if cur else None
        self._render()

    def _submit(self) -> None:
        self.result = mock.submit_mock(
            self._db,
            mock=self.mock_row,
            items=self.items,
            answers=self.answers,
            passing_marks=self.passing_marks,
            neg_fraction=self.neg_fraction,
            record_attempts=self.record_attempts,
            now=datetime.now(UTC),
        )
        self.phase = "done"
        self._render()

    def _pause(self) -> None:
        mock.save_progress(
            self._db,
            self.mock_row,
            answers=self.answers,
            index=self.index,
            elapsed_seconds=self._elapsed(),
        )
        self.paused = True
        self.exit()

    def on_key(self, event) -> None:
        key = event.key
        if self.phase == "answer":
            if key in ("a", "b", "c", "d"):
                self.pending_key = key.upper()
                self._render()
            elif key in CONFIDENCE_BY_KEY:
                self._commit_answer(CONFIDENCE_BY_KEY[key])
            elif key in ("n", "space"):
                if self.index >= len(self.items) - 1:
                    self._submit()
                else:
                    self._advance(1)
            elif key == "left":
                self._advance(-1)
            elif key == "s":
                self._submit()
            elif key == "escape":
                self._pause()
        elif self.phase == "done":
            if key == "v":
                self.phase = "review"
                self.review_index = 0
                self._render()
            elif key in ("q", "enter"):
                self.exit()
        elif self.phase == "review":
            if key in ("n", "space"):
                self.review_index += 1
                if self.review_index >= len(self.result.review):
                    self.phase = "done"
                self._render()
            elif key == "left":
                self.review_index -= 1
                self._render()
            elif key in ("q", "escape"):
                self.phase = "done"
                self._render()


# ── silent diagnostic UI (measurement, review after) ────────────────────────────
class DiagnosticApp(App):
    """Collects the 36 diagnostic answers with NO feedback, then reviews and seeds."""

    CSS = """
    Screen { layout: vertical; }
    #progress { color: $accent; height: 1; }
    #stem { padding: 1 0; text-style: bold; }
    #options { padding: 0 2; }
    #prompt { color: $warning; }
    #summary { padding: 1 0; }
    """

    def __init__(self, questions, *, user_id, course, db, now: datetime | None = None, force: bool = False):
        super().__init__()
        self.questions = list(questions)
        self.user_id = user_id
        self.course = course
        self._db = db
        self.now = _now(now)
        self.force = force
        self.answers: dict = {}
        self.index = 0
        self.pending_key: str | None = None
        self.phase = "answer" if self.questions else "done"
        self.result: diagnose.DiagnosticResult | None = None
        self.review_index = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="progress")
            yield Static(id="stem")
            yield Static(id="options")
            yield Static(id="prompt")
            yield Static(id="summary")

    def on_mount(self) -> None:
        self._render()

    def _set(self, wid: str, text: str) -> None:
        self.query_one(f"#{wid}", Static).update(text)

    @property
    def current(self) -> Question | None:
        if 0 <= self.index < len(self.questions):
            return self.questions[self.index]
        return None

    def _render(self) -> None:
        if self.phase == "done":
            self._render_summary()
            return
        if self.phase == "review":
            self._render_review()
            return
        q = self.current
        self._set("progress", f"Diagnostic {self.index + 1}/{len(self.questions)} (no feedback)")
        self._set("stem", q.stem)
        chosen = self.answers.get(q.id, (None, None))[0] or self.pending_key
        lines = []
        for opt in q.options or []:
            key = opt.get("key", "?")
            marker = ">" if chosen == key else " "
            lines.append(f"{marker} {key}) {opt.get('text', '')}")
        self._set("options", "\n".join(lines))
        self._set("prompt", "a-d answer / 1-2-3 confidence / n next")
        self._set("summary", "")

    def _render_summary(self) -> None:
        self._set("progress", "Diagnostic complete — seeded")
        self._set("stem", "")
        self._set("options", "")
        r = self.result
        if r is not None:
            self._set(
                "summary",
                f"Tested {len(r.tested)} concepts, seeded {len(r.seeded_untested)} "
                f"chapter-mates, {len(r.misconceptions)} misconception flag(s).",
            )
        self._set("prompt", "Press v to review, q to exit.")

    def _render_review(self) -> None:
        if self.review_index >= len(self.questions):
            self.phase = "done"
            self._render()
            return
        q = self.questions[self.review_index]
        chosen, _conf = self.answers.get(q.id, (None, None))
        self._set("progress", f"Review {self.review_index + 1}/{len(self.questions)}")
        self._set("stem", q.stem)
        lines = []
        for opt in q.options or []:
            key = opt.get("key", "?")
            mark = " (key)" if key == q.correct_key else ""
            if key == chosen:
                mark += " <- you"
            lines.append(f"  {key}) {opt.get('text', '')}{mark}")
        self._set("options", "\n".join(lines))
        self._set("summary", q.explanation or "")
        self._set("prompt", "n next / q exit")

    def _graded(self) -> list:
        return [
            diagnose.Graded(question=q, chosen_key=ck, confidence=cf or "guessing")
            for q in self.questions
            for (ck, cf) in [self.answers.get(q.id, (None, "guessing"))]
        ]

    def _seed(self) -> None:
        self.result = diagnose.seed_diagnostic(
            self._db,
            user_id=self.user_id,
            course=self.course,
            graded=self._graded(),
            now=self.now,
            force=self.force,
        )
        self.phase = "done"
        self._render()

    def on_key(self, event) -> None:
        key = event.key
        if self.phase == "answer":
            if key in ("a", "b", "c", "d"):
                self.pending_key = key.upper()
                self._render()
            elif key in CONFIDENCE_BY_KEY and self.pending_key is not None:
                q = self.current
                self.answers[q.id] = (self.pending_key, CONFIDENCE_BY_KEY[key])
                self._render()
            elif key in ("n", "space"):
                if self.index >= len(self.questions) - 1:
                    self._seed()
                else:
                    self.index += 1
                    self.pending_key = self.answers.get(self.current.id, (None, None))[0]
                    self._render()
        elif self.phase == "done":
            if key == "v":
                self.phase = "review"
                self.review_index = 0
                self._render()
            elif key in ("q", "enter"):
                self.exit()
        elif self.phase == "review":
            if key in ("n", "space"):
                self.review_index += 1
                self._render()
            elif key in ("q", "escape"):
                self.phase = "done"
                self._render()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _echo_mock_outcome(app: MockApp) -> None:
    if app.paused:
        typer.echo("Mock paused — resume with `atlas mock resume`.")
        return
    r = app.result
    if r is None:
        typer.echo("Mock exited without submission.")
        return
    verdict = "PASS" if r.passed else "FAIL"
    typer.echo(
        f"Mock score {r.raw_score:.2f}/{r.total_marks:.0f} "
        f"({verdict} at {r.passing_marks:.0f}); answered {r.answered}, skipped {r.skipped}."
    )


def _echo_strategy(report: full.StrategyReport) -> None:
    typer.echo("")
    typer.echo("Per-section prediction:")
    for s in report.sections:
        typer.echo(
            f"  {s.section}: {s.expected_marks:.1f}/{s.total_marks:.0f} "
            f"(95% CI {s.ci_low:.1f}-{s.ci_high:.1f})"
        )
    o = report.overall
    typer.echo(f"  overall: {o.expected_marks:.1f}/{o.total_marks:.0f}, P(pass) = {o.p_pass:.0%}")
    typer.echo("")
    typer.echo("Realized calibration:")
    if report.calibration:
        for c in report.calibration:
            typer.echo(f"  {c.confidence}: {c.rate:.0%} ({c.correct}/{c.n}) -> {c.recommendation}")
    else:
        typer.echo("  (not enough attempts yet)")
    typer.echo("")
    typer.echo("Attempt/skip policy:")
    for line in report.skip_policy:
        typer.echo(f"  {line}")


def register(group_app: typer.Typer) -> None:
    """Attach the mock/diagnostic commands to the ``atlas mock`` group."""

    @group_app.command("run")
    def run(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
        total: int = typer.Option(90, "--total", "-n", help="Standalone questions."),
    ) -> None:
        """Run a timed standalone mock (T-40)."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found — ingest a course first.")
                raise typer.Exit(1)
            params = mock.exam_params(c)
            now = datetime.now(UTC)
            items = mock.compose_standalone_items(db, user, c, now, random.Random(), total=total)
            if not items:
                typer.echo("No active questions available for a mock yet.")
                raise typer.Exit(1)
            predicted = mock.snapshot_prediction(
                db,
                user,
                c,
                passing_marks=params["passing_marks"],
                total_marks=params["total_marks"],
                neg_fraction=params["neg_fraction"],
            )
            row = mock.start_mock(
                db,
                user_id=user,
                course=c,
                items=items,
                predicted_before=predicted,
                duration_minutes=params["duration_minutes"],
                kind="standalone",
                now=now,
            )
            app = MockApp(
                items,
                mock_row=row,
                db=db,
                passing_marks=params["passing_marks"],
                neg_fraction=params["neg_fraction"],
                duration_minutes=params["duration_minutes"],
                now=now,
            )
            app.run()
            _echo_mock_outcome(app)

    @group_app.command("resume")
    def resume(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Resume the latest paused mock."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found.")
                raise typer.Exit(1)
            open_row = mock.latest_open_mock(db, user, c.id)
            if open_row is None:
                typer.echo("No paused mock to resume.")
                raise typer.Exit(0)
            resumed = mock.resume_mock(db, open_row.id)
            params = mock.exam_params(c)
            app = MockApp(
                resumed.items,
                mock_row=resumed.mock,
                db=db,
                passing_marks=params["passing_marks"],
                neg_fraction=params["neg_fraction"],
                duration_minutes=params["duration_minutes"],
                start_index=resumed.index,
                elapsed_seconds=resumed.elapsed_seconds,
                answers=resumed.answers,
            )
            app.run()
            _echo_mock_outcome(app)

    @group_app.command("full")
    def full_cmd(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Run the full-pattern mock (90 standalone + 9 caselets) with the strategy report (T-42)."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found.")
                raise typer.Exit(1)
            params = mock.exam_params(c)
            now = datetime.now(UTC)
            items = full.compose_full_items(db, user, c, now, random.Random())
            if not items:
                typer.echo("No active questions available for a full mock yet.")
                raise typer.Exit(1)
            report = full.strategy_report(
                db,
                user,
                items,
                passing_marks=params["passing_marks"],
                neg_fraction=params["neg_fraction"],
            )
            predicted = {
                "expected_marks": report.overall.expected_marks,
                "p_pass": report.overall.p_pass,
                "sections": [
                    {"section": s.section, "expected_marks": s.expected_marks}
                    for s in report.sections
                ],
            }
            row = mock.start_mock(
                db,
                user_id=user,
                course=c,
                items=items,
                predicted_before=predicted,
                duration_minutes=params["duration_minutes"],
                kind="full",
                now=now,
            )
            app = MockApp(
                items,
                mock_row=row,
                db=db,
                passing_marks=params["passing_marks"],
                neg_fraction=params["neg_fraction"],
                duration_minutes=params["duration_minutes"],
                now=now,
            )
            app.run()
            _echo_mock_outcome(app)
            _echo_strategy(report)

    @group_app.command("diagnose")
    def diagnose_cmd(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
        force: bool = typer.Option(False, "--force", help="Reseed even if a diagnostic exists."),
    ) -> None:
        """Run the Day-0 diagnostic and seed initial mastery (T-36; spec: `atlas diagnose`)."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found — ingest a course first.")
                raise typer.Exit(1)
            if diagnose.diagnostic_exists(db, user, c.id) and not force:
                typer.echo("Diagnostic already run. Pass --force to reseed.")
                raise typer.Exit(1)
            questions = diagnose.select_diagnostic_questions(db, c.id, random.Random())
            if not questions:
                typer.echo("No active questions available for the diagnostic yet.")
                raise typer.Exit(1)
            app = DiagnosticApp(questions, user_id=user, course=c, db=db, force=force)
            app.run()
            if app.result is not None and app.result.status is not None:
                typer.echo(render_status(app.result.status))
