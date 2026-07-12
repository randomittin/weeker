"""atlas doctor: SKIPs cleanly (never crashes) when db/keys/docker absent."""

from __future__ import annotations

from typer.testing import CliRunner

from atlas.cli.doctor import run_checks
from atlas.cli.main import app

runner = CliRunner()


def test_doctor_exits_zero_in_bare_env(monkeypatch, tmp_path):
    # Unreachable DB + no model keys: every check SKIPs, nothing hard-fails.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/none.db")
    monkeypatch.delenv("ATLAS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ATLAS_EMBED_API_KEY", raising=False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output


def test_run_checks_reports_skip_for_missing_keys(monkeypatch):
    monkeypatch.delenv("ATLAS_LLM_API_KEY", raising=False)
    checks = run_checks()
    by_name = {c.name: c for c in checks}
    assert by_name["model_keys"].status == "SKIP"
    assert by_name["live_completion"].status == "SKIP"
    # no check should ever be in an unknown state
    assert all(c.status in {"OK", "SKIP", "FAIL"} for c in checks)


def test_subcommand_groups_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("ingest", "generate", "study", "mock", "status", "verify"):
        assert group in result.output
