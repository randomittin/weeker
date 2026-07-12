# Weeker

**Pass your certification exam in a week.**

Adaptive NISM exam-prep learning engine. Ingests a certification workbook (PDF),
extracts a reviewed concept inventory, generates original gated questions and
caselets, then drives an adaptive study loop (confidence-weighted Elo ability +
FSRS-lite retention) with mock exams and a score prediction.

The **learning engine is the permanent asset**; the CLI/TUI are disposable shells
around `ingest/`, `generate/`, `learn/`.

## Layout

```
src/weeker/
  core/      config, db, models, cross-dialect types, llm + embed clients
  ingest/    corpus → pages → structure → chunks → embeddings → concepts (later waves)
  generate/  question + caselet generation and gate chain (later waves)
  learn/     Elo/retention, attempts, scheduler, prediction, study loop (later waves)
  cli/       typer app; `weeker doctor` + command groups
  verify/    schema introspection + verification gates
alembic/     migration 0001 = full DDL
prompts/     the six LLM prompt files (structure, concept_extract, objectives,
             question_gen, grounding_verify, caselet_gen)
tests/       pytest suite (runs on SQLite, no Postgres/keys needed)
```

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./weeker.db` | Postgres+pgvector in production; SQLite locally/tests |
| `WEEKER_MODEL_GENERATOR` / `_VERIFIER_CHEAP` / `_SOLVER_BLIND` / `WEEKER_MODEL_EMBED` | `…` | Provider model ids (G3 solver must be a different family) |
| `WEEKER_EMBED_DIM` | `1536` | Embedding dimension; sets `courses.embedding_dim` and the `vector(dim)` columns |
| `WEEKER_LLM_API_KEY` / `WEEKER_EMBED_API_KEY` | _(empty)_ | API keys; absent ⇒ live checks SKIP |
| `WEEKER_LLM_BASE_URL` / `WEEKER_EMBED_BASE_URL` | OpenAI-compatible | Chat + embeddings endpoints |

Postgres+pgvector is the real path. Unit tests run on SQLite: the `Vector(dim)`
type emits `vector(dim)` on Postgres and JSON on SQLite, and arrays/JSONB degrade
to JSON, so models import and the suite runs without Postgres or API keys.

## Gates

```bash
.venv/bin/ruff check .                 # lint (zero warnings)
.venv/bin/python -m pytest -q          # tests
DATABASE_URL=sqlite:///./weeker.db .venv/bin/alembic upgrade head
.venv/bin/python -m weeker.verify.schema   # asserts all tables + one_correct_per_question
.venv/bin/weeker doctor                 # M0 readiness gate (exit 0)
```
