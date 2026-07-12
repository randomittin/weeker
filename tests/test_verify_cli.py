"""weeker verify: every gate degrades to SKIP (never crashes) on a bare/empty DB,
and `weeker verify all` renders the summary table. Adaptive is DB-free and PASSes.

Each command is driven through the real Typer app via CliRunner against an empty
SQLite database, mirroring the M0 doctor gate's bare-env contract.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from typer.testing import CliRunner

from weeker.cli.main import app
from weeker.core import db as core_db
from weeker.core.models import Base

runner = CliRunner()


def _empty_db(monkeypatch, tmp_path, *, migrated: bool = False) -> str:
    """Point DATABASE_URL at a fresh SQLite file; optionally create the schema."""
    url = f"sqlite:///{tmp_path}/weeker.db"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.delenv("WEEKER_LLM_API_KEY", raising=False)
    monkeypatch.delenv("WEEKER_EMBED_API_KEY", raising=False)
    # get_engine/_sessionmaker are @cache-keyed on the raw arg (None), so a prior
    # invocation would pin a stale engine — clear so each test resolves fresh.
    core_db.get_engine.cache_clear()
    core_db._sessionmaker.cache_clear()
    if migrated:
        Base.metadata.create_all(create_engine(url))
    return url


def test_verify_group_registered():
    result = runner.invoke(app, ["verify", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("schema", "ingest", "concepts", "bank", "adaptive", "all"):
        assert cmd in result.output


def test_schema_skips_on_empty_db(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path)
    result = runner.invoke(app, ["verify", "schema"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output


def test_schema_passes_on_migrated_db(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "schema"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_ingest_skips_without_course(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "ingest"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output


def test_ingest_parse_stage_skips_without_course(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "ingest", "--stage", "parse"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output


def test_ingest_rejects_unknown_stage(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "ingest", "--stage", "bogus"])
    assert result.exit_code == 2, result.output


def test_concepts_skips_without_course(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "concepts"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output


def test_bank_skips_without_course(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "bank"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output


def test_bank_caselets_flag_skips_without_course(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "bank", "--caselets"])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output


def test_adaptive_passes_without_db(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path)
    result = runner.invoke(app, ["verify", "adaptive"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_all_renders_summary_table_and_exits_zero_on_bare_env(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path)
    result = runner.invoke(app, ["verify", "all"])
    assert result.exit_code == 0, result.output
    assert "weeker verify all" in result.output
    for gate in ("schema", "ingest", "concepts", "bank", "adaptive"):
        assert gate in result.output
    # adaptive is DB-free ⇒ always exercised; the rest SKIP on a bare env.
    assert "PASS" in result.output
    assert "SKIP" in result.output


def test_all_on_migrated_empty_db_skips_course_gates(monkeypatch, tmp_path):
    _empty_db(monkeypatch, tmp_path, migrated=True)
    result = runner.invoke(app, ["verify", "all"])
    assert result.exit_code == 0, result.output
    # schema now PASSes (tables present), course gates SKIP (no course), adaptive PASSes.
    assert "PASS" in result.output
    assert "SKIP" in result.output
