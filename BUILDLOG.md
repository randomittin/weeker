# BUILDLOG — Wave 0 (Foundation)

Task → what → gate command → result. Gates run from `./.venv`.

| Task | What | Gate | Result |
|---|---|---|---|
| T-01 | Repo layout, `pyproject.toml` (py 3.11), deps, ruff, pre-commit (ruff+gitleaks), prompts copied, README | `pip install -e ".[dev]" && ruff check . && pytest -q` | PASS (17 tests, ruff clean) |
| T-03 | `core/config.py` — every 016 §T-03 constant verbatim; `MODEL_*`/`EMBED_*` from env with placeholder defaults; single source of tunables | `python -c "from atlas.core import config"` + `tests/test_config_is_single_source.py` (warn-only grep lint) | PASS |
| T-02 | Full schema (21 tables) `core/models.py` (SQLAlchemy 2.0); `core/db.py` engine/session; `core/types.py` cross-dialect `Vector`/JSONB/arrays; alembic `0001` full DDL (pgvector guarded to Postgres); `verify/schema.py` | `alembic upgrade head && python -m atlas.verify.schema` (SQLite) + `tests/test_models_sqlite.py`, `tests/test_schema_verify.py` | PASS (21 tables + `one_correct_per_question`) |
| T-04 | `core/llm.py` `complete()` strict-JSON retry + injectable transport; `core/embed.py` batching + sha256 on-disk cache | `pytest tests/test_llm.py tests/test_embed.py` | PASS (retry path + cache-hit proven) |
| T-05 | `cli/main.py` typer app + 6 sub-app groups (ingest/generate/study/mock/status/verify) with wave-attach hook; `cli/doctor.py` (`atlas doctor`) db/migrations/keys/live/embed-dim checks, SKIP when absent | `atlas doctor` exit 0 — **M0 gate** | PASS (all live checks SKIP, exit 0) |

# BUILDLOG — Wave 1a (Ingestion pipeline)

| Task | What | Gate | Result |
|---|---|---|---|
| T-16 | `ingest/review.py` S5 review TUI (Textual) — a/e/m/s/d accept/edit/merge/split/drop + `p` peek source; each decision writes `review_status`+`reviewed_at` and commits → resumable (relaunch re-queries `pending`). Fix: `action_edit` uses `push_screen(...,callback)` (not `push_screen_wait`, which needs a worker) | `pytest -q tests/test_ingest_review.py` (Textual Pilot drives a/e/d over 3 fixtures; asserts DB status transitions + resumable + peek) | PASS (3 tests) |
| T-17 | `ingest/objectives.py` S6 — `generate_objectives` (batch-of-10 via `prompts/objectives.txt`, response keyed by concept id, per-concept validation, bloom/style coerced) + `compute_blueprint_weights` (base ∝ reviewed-concept count, manifest overrides authoritative, remainder split by base, Σ=1.0) + `persist_blueprint_weights`; `verify/concept_checks.py` `verify_concepts` = **M2 gate** (≥1 objective/reviewed concept, blueprint normalized, no orphan chapter/source-chunk) | `pytest -q tests/test_ingest_objectives.py` (fake LLM parse, batch-of-10, override exact + normalized, verify pass + 3 fail modes) | PASS (8 tests) |
| —   | `ingest/cli.py` `register(group_app)` → `atlas ingest run` (S0-S6 orchestration: manifest→parse→structure→chunk→embed→concepts→objectives+blueprint), `review` (launch TUI), `verify-parse`, `verify` (M1), `verify-concepts` (M2); real functions, no stub | `atlas ingest --help` lists 5 commands; imports resolve | PASS |

# BUILDLOG — Wave 2 (Adaptive learn loop)

| Task | What | Gate | Result |
|---|---|---|---|
| T-34 | `learn/study_cli.py` — `StudyApp` (one Q/screen, a-d answer → 1/2/3 confidence → grade via `record_attempt`, explanation block, `p` page-ranged source peek, end-of-session Δθ report) + `FlashApp` (space-flip drain, 1/2/3 grade reschedules via `retention.stability_update`); `register` attaches `atlas study run/flash/verify/replay` | `pytest -q tests/test_learn_study.py` (Textual Pilot 5-Q session asserts 5 Attempt rows) | PASS (4 tests) |
| T-35 | `learn/verify_adaptive.py` synthetic 200-attempt Elo replay (Pearson r≥0.7 vs true ability, bounds + [0,1] held) + `learn/replay.py` mastery-recompute diff vs stored `Mastery` rows; `verify_adaptive()`/`replay()` exit 0 = **M4 gate** | `pytest -q tests/test_learn_verify.py`; `atlas study verify` exit 0 (r=0.863) | PASS (5 tests) |

Note: `attempt_service._record` now stamps `Attempt.created_at=now` — replay orders the log by `(created_at, id)`, so the write path must persist the caller's timestamp for deterministic recompute.

## Environment notes
- Python 3.11 (pinned `>=3.11,<3.12`); no PEP695 / 3.12-only syntax.
- No Docker: Postgres+pgvector is production; SQLite is the test fallback via the
  `Vector` TypeDecorator (→ `vector(dim)` on Postgres, JSON on SQLite) and
  `.with_variant` for JSONB/arrays. pgvector HNSW indexes are guarded to Postgres
  in migration 0001; `verify.schema` skips the pgvector-index assertion on SQLite.
- No LLM/embed keys: transports are injectable; tests use fakes. Real HTTP path
  present (OpenAI-compatible) but not exercised.
