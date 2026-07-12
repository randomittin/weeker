"""The ``weeker status`` command group (016 §T-41/T-50).

``register`` attaches the real commands to the group created in ``cli/main.py``
(which imports this module and calls ``register`` — no edit to that file):

* ``weeker status show`` — the learner dashboard (T-41): heatmap, due counts,
  flags, prediction band, tonight's refill plan and (from day 3) the realized
  calibration table;
* ``weeker status refill`` — the nightly loop (T-50). Spec names this
  ``weeker refill``; ``cli/main.py`` is frozen with no ``refill`` group, so it is
  wired here as ``weeker status refill`` (documented deviation, same pattern as
  ``weeker mock diagnose``). It runs generate-refill → bank verify → status print
  in one command.
"""

from __future__ import annotations

from datetime import UTC, datetime

import typer
from sqlalchemy import select

from weeker.core.db import session_scope
from weeker.core.models import Course
from weeker.learn import refill, status


def _resolve_course(db, course: str | None) -> Course | None:
    stmt = select(Course)
    if course:
        stmt = stmt.where((Course.slug == course) | (Course.title == course))
    return db.scalars(stmt.order_by(Course.created_at.desc())).first()


def register(group_app: typer.Typer) -> None:
    """Attach the status/refill commands to the ``weeker status`` group."""

    @group_app.command("show")
    def show(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Print the learner status dashboard (T-41)."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found — ingest a course first.")
                raise typer.Exit(1)
            report = status.build_status(
                db, user_id=user, course=c, now=datetime.now(UTC)
            )
            typer.echo(status.render_status(report))
        raise typer.Exit(0)

    @group_app.command("refill")
    def refill_cmd(
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
        count: int = typer.Option(40, "--count", "-n", help="Questions to author."),
        consensus: int = typer.Option(1, "--consensus", help="Blind-solve consensus count."),
        caselets: bool = typer.Option(False, "--caselets", help="Also apply the caselet bank bar."),
    ) -> None:
        """Nightly loop: generate --refill + bank verify + status print (T-50; spec: `weeker refill`)."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found — ingest a course first.")
                raise typer.Exit(1)
            result = refill.run_refill(
                db, c, user, count=count, consensus=consensus, caselets=caselets
            )
            g = result.generation
            typer.echo(
                f"refill: generated={g.generated} accepted={g.accepted} "
                f"disputed={g.disputed} rejected={g.rejected}"
            )
            b = result.bank
            typer.echo(
                f"bank: active={b.total_active} blind_agreement={b.blind_agreement:.3f} "
                f"near_dup={b.near_dup_rate:.3f} passed={b.passed}"
            )
            typer.echo("")
            typer.echo(status.render_status(result.status))
        raise typer.Exit(0)
