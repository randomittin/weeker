"""The ``weeker generate`` command group (016 §T-20..T-26).

``register`` attaches the real generation commands to the group created in
``cli/main.py`` (which imports ``weeker.generate.cli`` and calls ``register`` — no
edit to that file needed):

* ``weeker generate run`` — overgenerate → gate → persist a batch of questions;
* ``weeker generate caselets`` — the caselet engine (T-26);
* ``weeker generate review`` — the disputed-question review TUI;
* ``weeker generate verify-bank`` — the M3 bank gate (``--caselets`` adds the
  caselet bar).

Every command wires the real pipeline/verify functions; there is no stub path.
"""

from __future__ import annotations

import typer
from sqlalchemy import select

from weeker.core.db import get_session, session_scope
from weeker.core.models import Course, Question
from weeker.generate.calibrate import audit_bank
from weeker.generate.caselets import run_caselets
from weeker.generate.pipeline import run_generation
from weeker.generate.review_tui import DisputeApp
from weeker.seed.load_authored import load_authored
from weeker.verify.bank_checks import verify_bank


def _resolve_course(db, course: str | None) -> Course | None:
    stmt = select(Course)
    if course:
        stmt = stmt.where((Course.slug == course) | (Course.title == course))
    return db.scalars(stmt.order_by(Course.created_at.desc())).first()


def register(group_app: typer.Typer) -> None:
    """Attach the generation commands to the ``weeker generate`` group."""

    @group_app.command("run")
    def run(
        count: int = typer.Option(40, "--count", "-n", help="Questions to author."),
        refill: bool = typer.Option(False, "--refill", help="Weak-concept refill mode."),
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
        consensus: int = typer.Option(1, "--consensus", help="Blind-solve consensus count."),
    ) -> None:
        """Generate, gate and persist questions for a course."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found — ingest a course first.")
                raise typer.Exit(1)
            report = run_generation(
                db, c, user, count=count, refill=refill, consensus=consensus
            )
        typer.echo(
            f"generated={report.generated} accepted={report.accepted} "
            f"disputed={report.disputed} rejected={report.rejected} by_gate={report.by_gate}"
        )
        raise typer.Exit(0)

    @group_app.command("caselets")
    def caselets_cmd(
        count: int = typer.Option(9, "--count", "-n", help="Caselets to author."),
        two_mark: int = typer.Option(3, "--two-mark", help="How many are 2-mark caselets."),
        refill: bool = typer.Option(False, "--refill", help="Weak-concept refill mode."),
        user: str = typer.Option("default", "--user", "-u"),
        course: str = typer.Option(None, "--course", "-c"),
        consensus: int = typer.Option(1, "--consensus", help="Blind-solve consensus count."),
    ) -> None:
        """Generate, gate (incl. G6) and persist caselets for a course."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found — ingest a course first.")
                raise typer.Exit(1)
            report = run_caselets(
                db,
                c,
                user,
                count=count,
                two_mark_count=two_mark,
                refill=refill,
                consensus=consensus,
            )
        typer.echo(
            f"caselets generated={report.generated} accepted={report.accepted} "
            f"disputed={report.disputed} rejected={report.rejected} "
            f"g6_regenerated={report.g6_regenerated}"
        )
        raise typer.Exit(0)

    @group_app.command("load-authored")
    def load_authored_cmd(
        directory: str = typer.Option(
            "courses/nism-xa/authored", "--dir", "-d", help="Authored chapter JSON directory."
        ),
        course: str = typer.Option(None, "--course", "-c"),
        review_status: str = typer.Option(
            "accepted", "--review-status", help="Concept review_status to set."
        ),
    ) -> None:
        """Load hand-authored concept/question/caselet JSON (keyless, code-gate enforced)."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found — ingest a course first.")
                raise typer.Exit(1)
            report = load_authored(directory, db=db, review_status=review_status, course=course)

        header = (
            f"{'ch':>3}  {'title':<28} {'conc':>4} {'obj':>4} {'q':>4} "
            f"{'case':>4} {'rejected(by gate)':>24}"
        )
        typer.echo(header)
        typer.echo("-" * len(header))
        for chp in sorted(report.chapters, key=lambda x: x.ordinal):
            rej = ", ".join(f"{g}:{n}" for g, n in sorted(chp.rejected_by_gate.items())) or "-"
            typer.echo(
                f"{chp.ordinal:>3}  {chp.title[:28]:<28} "
                f"{chp.concepts_inserted:>4} {chp.objectives_inserted:>4} "
                f"{chp.questions_inserted:>4} {chp.caselets_inserted:>4} {rej:>24}"
            )
        typer.echo(
            f"TOTAL concepts={report.concepts_inserted} objectives={report.objectives_inserted} "
            f"questions={report.questions_inserted} caselets={report.caselets_inserted} "
            f"rejected={report.questions_rejected} by_gate={report.rejected_by_gate}"
        )
        for fname, err in report.file_errors.items():
            typer.echo(f"FILE ERROR {fname}: {err}")
        raise typer.Exit(1 if report.file_errors else 0)

    @group_app.command("review")
    def review(
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Launch the disputed-question review TUI."""
        db = get_session()
        c = _resolve_course(db, course)
        if c is None:
            typer.echo("No course found.")
            db.close()
            raise typer.Exit(1)
        disputed = list(
            db.scalars(
                select(Question).where(
                    Question.course_id == c.id, Question.status == "disputed"
                )
            )
        )
        if not disputed:
            typer.echo("No disputed questions to review.")
            db.close()
            raise typer.Exit(0)
        DisputeApp(disputed, db=db).run()

    @group_app.command("recalibrate")
    def recalibrate(
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Audit the active bank: recompute discrimination, flag drift, retire dead items."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found.")
                raise typer.Exit(1)
            summary = audit_bank(db, c.id)
        typer.echo(
            f"audited={summary['audited']} recalibrate={summary['recalibrate']} "
            f"retired={summary['retired']} dead_distractors={summary['dead_distractors']}"
        )
        raise typer.Exit(0)

    @group_app.command("verify-bank")
    def verify_bank_cmd(
        course: str = typer.Option(None, "--course", "-c"),
        caselets: bool = typer.Option(False, "--caselets", help="Also apply the caselet bar."),
    ) -> None:
        """Run the M3 bank gate (schema/grounding/blind/near-dup/G4; --caselets adds the caselet bar)."""
        with session_scope() as db:
            c = _resolve_course(db, course)
            if c is None:
                typer.echo("No course found.")
                raise typer.Exit(1)
            report = verify_bank(db, c.id, caselets=caselets)
        typer.echo(
            f"active={report.total_active} grounding_ok={report.grounding_ok} "
            f"blind_agreement={report.blind_agreement:.3f} near_dup={report.near_dup_rate:.3f} "
            f"g4_hits={report.g4_hits} disputed={report.disputed_count} "
            f"caselets_active={report.caselets_active}"
        )
        if report.passed:
            typer.echo("M3 bank gate: PASS")
            raise typer.Exit(0)
        for f in report.failures:
            typer.echo(f"FAIL {f}")
        raise typer.Exit(1)
