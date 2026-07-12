# BUILDLOG — Wave 0 (Foundation)

Task → what → gate command → result. Gates run from `./.venv`.

| Task | What | Gate | Result |
|---|---|---|---|
| T-01 | Repo layout, `pyproject.toml` (py 3.11), deps, ruff, pre-commit (ruff+gitleaks), prompts copied, README | `pip install -e ".[dev]" && ruff check . && pytest -q` | PASS (17 tests, ruff clean) |
| T-03 | `core/config.py` — every 016 §T-03 constant verbatim; `MODEL_*`/`EMBED_*` from env with placeholder defaults; single source of tunables | `python -c "from atlas.core import config"` + `tests/test_config_is_single_source.py` (warn-only grep lint) | PASS |
| T-02 | Full schema (21 tables) `core/models.py` (SQLAlchemy 2.0); `core/db.py` engine/session; `core/types.py` cross-dialect `Vector`/JSONB/arrays; alembic `0001` full DDL (pgvector guarded to Postgres); `verify/schema.py` | `alembic upgrade head && python -m atlas.verify.schema` (SQLite) + `tests/test_models_sqlite.py`, `tests/test_schema_verify.py` | PASS (21 tables + `one_correct_per_question`) |
| T-04 | `core/llm.py` `complete()` strict-JSON retry + injectable transport; `core/embed.py` batching + sha256 on-disk cache | `pytest tests/test_llm.py tests/test_embed.py` | PASS (retry path + cache-hit proven) |
| T-05 | `cli/main.py` typer app + 6 sub-app groups (ingest/generate/study/mock/status/verify) with wave-attach hook; `cli/doctor.py` (`atlas doctor`) db/migrations/keys/live/embed-dim checks, SKIP when absent | `atlas doctor` exit 0 — **M0 gate** | PASS (all live checks SKIP, exit 0) |

## Environment notes
- Python 3.11 (pinned `>=3.11,<3.12`); no PEP695 / 3.12-only syntax.
- No Docker: Postgres+pgvector is production; SQLite is the test fallback via the
  `Vector` TypeDecorator (→ `vector(dim)` on Postgres, JSON on SQLite) and
  `.with_variant` for JSONB/arrays. pgvector HNSW indexes are guarded to Postgres
  in migration 0001; `verify.schema` skips the pgvector-index assertion on SQLite.
- No LLM/embed keys: transports are injectable; tests use fakes. Real HTTP path
  present (OpenAI-compatible) but not exercised.
