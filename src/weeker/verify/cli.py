"""The ``weeker verify`` command group (016 §T-14/T-17/T-24/T-35).

``register`` attaches the milestone gates to the group created in
``cli/main.py`` (which imports this module and calls ``register`` — no edit to
that file needed). Each command wires an already-built gate function:

* ``weeker verify schema``   — schema introspection assertion (``verify.schema``).
* ``weeker verify ingest``   — the M1 ingestion gate (``--stage parse`` for S1 only).
* ``weeker verify concepts`` — the M2 concept gate.
* ``weeker verify bank``     — the M3 bank gate (``--caselets`` folds in the caselet bar).
* ``weeker verify adaptive`` — the M4 synthetic-replay gate (no DB required).
* ``weeker verify all``      — every gate in order, rendered as a PASS/FAIL/SKIP table.

Like ``weeker doctor``, each gate reports PASS, SKIP (prerequisite absent — DB
unreachable, not migrated, or no course ingested yet) or FAIL (present but the
gate did not hold). Commands exit non-zero only on a hard FAIL, so a bare dev
box with an empty database still exits 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from weeker.core.db import get_engine, session_scope
from weeker.core.models import Course, Source
from weeker.learn.verify_adaptive import verify_adaptive
from weeker.verify import schema as schema_verify
from weeker.verify.bank_checks import verify_bank
from weeker.verify.concept_checks import verify_concepts
from weeker.verify.ingest_checks import check_parse, verify_ingest

console = Console()

_COLOR = {"PASS": "green", "SKIP": "yellow", "FAIL": "red"}


@dataclass
class GateResult:
    """Outcome of one milestone gate."""

    name: str
    status: str  # PASS | SKIP | FAIL
    detail: str


def _join(failures: list[str], limit: int = 3) -> str:
    """One-line join of failure strings, truncated to keep the table readable."""
    head = "; ".join(failures[:limit])
    return head + (f" (+{len(failures) - limit} more)" if len(failures) > limit else "")


def _db_reachable() -> tuple[bool, str]:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, ""
    except Exception as e:  # noqa: BLE001 — any connect failure ⇒ SKIP, like doctor
        return False, str(e)


def _resolve_course(db: Session, course: str | None) -> Course | None:
    stmt = select(Course)
    if course:
        stmt = stmt.where((Course.slug == course) | (Course.title == course))
    return db.scalars(stmt.order_by(Course.created_at.desc())).first()


def _course_gate(
    name: str, course: str | None, run: Callable[[Session, Course], GateResult]
) -> GateResult:
    """Run a course-scoped gate, degrading to SKIP when the DB isn't ready."""
    ok, err = _db_reachable()
    if not ok:
        return GateResult(name, "SKIP", f"db not reachable: {err}")
    try:
        with session_scope() as db:
            row = _resolve_course(db, course)
            if row is None:
                return GateResult(name, "SKIP", "no course ingested")
            return run(db, row)
    except (OperationalError, ProgrammingError) as e:
        return GateResult(name, "SKIP", f"schema not ready: {e.orig}")


# --- individual gates -------------------------------------------------------


def gate_schema() -> GateResult:
    ok, err = _db_reachable()
    if not ok:
        return GateResult("schema", "SKIP", f"db not reachable: {err}")
    engine = get_engine()
    try:
        tables = inspect(engine).get_table_names()
    except Exception as e:  # noqa: BLE001
        return GateResult("schema", "SKIP", f"introspection failed: {e}")
    if not tables:
        return GateResult("schema", "SKIP", "empty db (not migrated)")
    failures = schema_verify.check(engine)
    if failures:
        return GateResult("schema", "FAIL", _join(failures))
    n = len(schema_verify.EXPECTED_TABLES)
    return GateResult("schema", "PASS", f"{n} tables + one_correct_per_question present")


def gate_ingest(course: str | None = None, *, stage: str | None = None) -> GateResult:
    def run(db: Session, c: Course) -> GateResult:
        if stage == "parse":
            failures: list[str] = []
            for src_id in db.scalars(select(Source.id).where(Source.course_id == c.id)):
                failures += check_parse(db, src_id)
            label = "S1 parse gate"
        else:
            failures = verify_ingest(db, c.id)
            label = "M1 ingestion gate"
        if failures:
            return GateResult("ingest", "FAIL", _join(failures))
        return GateResult("ingest", "PASS", f"{label} ({c.slug or c.title})")

    return _course_gate("ingest", course, run)


def gate_concepts(course: str | None = None) -> GateResult:
    def run(db: Session, c: Course) -> GateResult:
        failures = verify_concepts(db, c.id)
        if failures:
            return GateResult("concepts", "FAIL", _join(failures))
        return GateResult("concepts", "PASS", f"M2 concept gate ({c.slug or c.title})")

    return _course_gate("concepts", course, run)


def _bank_detail(report) -> str:
    return (
        f"active={report.total_active} schema_ok={report.schema_ok} "
        f"grounding_ok={report.grounding_ok} blind={report.blind_agreement:.3f} "
        f"near_dup={report.near_dup_rate:.3f} g4={report.g4_hits} "
        f"disputed={report.disputed_count} caselets={report.caselets_active}"
    )


def gate_bank(course: str | None = None, *, caselets: bool = False) -> GateResult:
    def run(db: Session, c: Course) -> GateResult:
        report = verify_bank(db, c.id, caselets=caselets)
        detail = _bank_detail(report)
        if report.passed:
            return GateResult("bank", "PASS", detail)
        return GateResult("bank", "FAIL", f"{detail} | {_join(report.failures)}")

    return _course_gate("bank", course, run)


def gate_adaptive() -> GateResult:
    report = verify_adaptive()
    if report.ok:
        detail = (
            f"r={report.correlation:.3f}, {report.n_attempts} attempts, "
            f"{report.n_concepts} concepts"
        )
        return GateResult("adaptive", "PASS", detail)
    fails = [c for c in report.checks if c.startswith("FAIL")]
    return GateResult("adaptive", "FAIL", _join(fails))


# --- rendering + exit -------------------------------------------------------


def _exit_code(results: list[GateResult]) -> int:
    return 1 if any(r.status == "FAIL" for r in results) else 0


def _emit_one(result: GateResult) -> None:
    console.print(f"[{_COLOR[result.status]}]{result.status}[/] {result.name}: {result.detail}")
    raise typer.Exit(_exit_code([result]))


def _render_table(results: list[GateResult]) -> int:
    table = Table(title="weeker verify all")
    table.add_column("gate")
    table.add_column("status")
    table.add_column("detail")
    for r in results:
        table.add_row(r.name, f"[{_COLOR[r.status]}]{r.status}[/]", r.detail)
    console.print(table)
    fails = [r for r in results if r.status == "FAIL"]
    skips = [r for r in results if r.status == "SKIP"]
    if fails:
        console.print(f"[red]{len(fails)} gate(s) FAILED[/] ({len(skips)} skipped)")
    else:
        console.print(f"[green]all gates green[/] ({len(skips)} skipped)")
    return _exit_code(results)


def register(group_app: typer.Typer) -> None:
    """Attach the milestone gates to the ``weeker verify`` group."""

    @group_app.command("schema")
    def schema_cmd() -> None:
        """Assert the schema (all tables + one_correct_per_question, HNSW on PG)."""
        _emit_one(gate_schema())

    @group_app.command("ingest")
    def ingest_cmd(
        course: str = typer.Option(None, "--course", "-c", help="Course slug or title."),
        stage: str = typer.Option(
            None, "--stage", help="Run only one stage; currently supports 'parse' (S1)."
        ),
    ) -> None:
        """Run the M1 ingestion gate (or the S1 parse stage with --stage parse)."""
        if stage not in (None, "parse"):
            typer.echo(f"unknown --stage {stage!r} (only 'parse' is supported)")
            raise typer.Exit(2)
        _emit_one(gate_ingest(course, stage=stage))

    @group_app.command("concepts")
    def concepts_cmd(
        course: str = typer.Option(None, "--course", "-c", help="Course slug or title."),
    ) -> None:
        """Run the M2 concept gate (objectives + blueprint + no orphans)."""
        _emit_one(gate_concepts(course))

    @group_app.command("bank")
    def bank_cmd(
        course: str = typer.Option(None, "--course", "-c", help="Course slug or title."),
        caselets: bool = typer.Option(False, "--caselets", help="Also apply the caselet bar."),
    ) -> None:
        """Run the M3 bank gate; prints the full BankReport, exits 1 if not passed."""
        _emit_one(gate_bank(course, caselets=caselets))

    @group_app.command("adaptive")
    def adaptive_cmd() -> None:
        """Run the M4 synthetic adaptive-replay gate (no DB required)."""
        _emit_one(gate_adaptive())

    @group_app.command("all")
    def all_cmd(
        course: str = typer.Option(None, "--course", "-c", help="Course slug or title."),
        caselets: bool = typer.Option(False, "--caselets", help="Include the caselet bar in bank."),
    ) -> None:
        """Run every gate in order and print a PASS/FAIL/SKIP summary table."""
        results = [
            gate_schema(),
            gate_ingest(course),
            gate_concepts(course),
            gate_bank(course, caselets=caselets),
            gate_adaptive(),
        ]
        raise typer.Exit(_render_table(results))
