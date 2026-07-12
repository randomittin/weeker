"""Weeker CLI root.

Wires the ``weeker`` console entrypoint and registers the six command groups
(ingest, generate, study, mock, status, verify) as real Typer sub-apps. The
groups are empty now; later waves attach commands by providing an
``weeker.<pkg>.cli`` module exposing ``register(app: typer.Typer)`` — no edit to
this file required. ``weeker doctor`` is the M0 gate.
"""

from __future__ import annotations

import importlib

import typer

from weeker.cli.doctor import render, run_checks

app = typer.Typer(help="Weeker — pass your certification exam in a week.", no_args_is_help=True)

# Real, empty sub-apps. Later waves import these and add commands.
ingest_app = typer.Typer(help="Corpus ingestion pipeline.", no_args_is_help=True)
generate_app = typer.Typer(help="Question and caselet generation.", no_args_is_help=True)
study_app = typer.Typer(help="Adaptive study and flashcards.", no_args_is_help=True)
mock_app = typer.Typer(help="Mock exams.", no_args_is_help=True)
status_app = typer.Typer(help="Progress and predictions.", no_args_is_help=True)
verify_app = typer.Typer(help="Verification gates.", no_args_is_help=True)

_GROUPS: dict[str, tuple[typer.Typer, str]] = {
    "ingest": (ingest_app, "weeker.ingest.cli"),
    "generate": (generate_app, "weeker.generate.cli"),
    "study": (study_app, "weeker.learn.study_cli"),
    "mock": (mock_app, "weeker.learn.mock_cli"),
    "status": (status_app, "weeker.learn.status_cli"),
    "verify": (verify_app, "weeker.verify.cli"),
}

for name, (group_app, module_path) in _GROUPS.items():
    app.add_typer(group_app, name=name)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        continue  # wave hasn't provided commands yet — group stays empty but real
    register = getattr(module, "register", None)
    if callable(register):
        register(group_app)


@app.command()
def doctor() -> None:
    """Check environment readiness (db, migrations, model keys, live calls)."""
    code = render(run_checks())
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
