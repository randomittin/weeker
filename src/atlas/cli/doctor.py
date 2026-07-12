"""``atlas doctor`` — the M0 readiness gate.

Runs a fixed set of environment checks and prints a table. Each check reports
OK, SKIP (prerequisite absent — e.g. no DB, no keys), or FAIL (present but
broken). The command exits non-zero only on a hard FAIL, so a bare dev box with
no Postgres and no API keys still exits 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from atlas.core import config
from atlas.core.db import database_url, get_engine
from atlas.core.embed import embed
from atlas.core.llm import complete

console = Console()


@dataclass
class CheckResult:
    name: str
    status: str  # OK | SKIP | FAIL
    detail: str


def _alembic_head() -> str | None:
    """Head revision id from the alembic scripts, or None if not locatable."""
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        repo_root = Path(__file__).resolve().parents[3]
        alembic_dir = repo_root / "alembic"
        if not alembic_dir.exists():
            return None
        cfg = Config()
        cfg.set_main_option("script_location", str(alembic_dir))
        return ScriptDirectory.from_config(cfg).get_current_head()
    except Exception:
        return None


def check_db() -> CheckResult:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return CheckResult("db_reachable", "OK", database_url())
    except Exception as e:
        return CheckResult("db_reachable", "SKIP", f"not reachable: {e}")


def check_migrations() -> CheckResult:
    head = _alembic_head()
    if head is None:
        return CheckResult("migrations_current", "SKIP", "alembic scripts not found")
    try:
        from alembic.runtime.migration import MigrationContext

        with get_engine().connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
    except Exception as e:
        return CheckResult("migrations_current", "SKIP", f"db not reachable: {e}")
    if current is None:
        return CheckResult("migrations_current", "SKIP", "no migrations applied yet")
    if current == head:
        return CheckResult("migrations_current", "OK", f"at head {head}")
    return CheckResult("migrations_current", "FAIL", f"db {current} != head {head}")


def check_model_keys() -> CheckResult:
    if not config.models_configured():
        return CheckResult("model_keys", "SKIP", "MODEL_* still placeholders")
    if not config.llm_key_present():
        return CheckResult("model_keys", "SKIP", "ATLAS_LLM_API_KEY not set")
    return CheckResult("model_keys", "OK", "models + key configured")


def check_live_completion() -> CheckResult:
    if not (config.models_configured() and config.llm_key_present()):
        return CheckResult("live_completion", "SKIP", "no models/keys configured")
    models = [config.MODEL_GENERATOR, config.MODEL_VERIFIER_CHEAP, config.MODEL_SOLVER_BLIND]
    for model in models:
        try:
            complete(model, "Reply with the single word: ok", "ok")
        except Exception as e:
            return CheckResult("live_completion", "FAIL", f"{model}: {e}")
    return CheckResult("live_completion", "OK", f"{len(models)} models responded")


def check_embed_dim() -> CheckResult:
    if not config.embed_key_present():
        return CheckResult("embed_dim", "SKIP", "ATLAS_EMBED_API_KEY not set")
    try:
        vec = embed(["dimension probe"])
    except Exception as e:
        return CheckResult("embed_dim", "FAIL", f"embed call failed: {e}")
    dim = vec.shape[1] if vec.ndim == 2 else 0
    if dim == config.EMBED_DIM:
        return CheckResult("embed_dim", "OK", f"dim={dim}")
    return CheckResult("embed_dim", "FAIL", f"dim {dim} != EMBED_DIM {config.EMBED_DIM}")


def run_checks() -> list[CheckResult]:
    return [
        check_db(),
        check_migrations(),
        check_model_keys(),
        check_live_completion(),
        check_embed_dim(),
    ]


_COLOR = {"OK": "green", "SKIP": "yellow", "FAIL": "red"}


def render(results: list[CheckResult]) -> int:
    table = Table(title="atlas doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for r in results:
        table.add_row(r.name, f"[{_COLOR[r.status]}]{r.status}[/]", r.detail)
    console.print(table)
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        console.print(f"[red]{len(failures)} hard failure(s)[/]")
        return 1
    console.print("[green]no hard failures[/]")
    return 0
