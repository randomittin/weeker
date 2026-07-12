# Atlas Build Plan — executed from 016 ultraplan + 017 patch

Adaptive NISM exam-prep learning engine. Task DAG in `atlas-spec/016_ultraplan.md`, patch in `atlas-spec/017_...md`. Docs 010–015 not in source drop → schema/interfaces derived from 016+017 (config defaults, DDL deltas, interfaces, gates all present there).

## Environment constraints
- Python 3.11 (spec says 3.12 → pin 3.11, no 3.12-only syntax)
- No docker locally → Postgres+pgvector is the real path; unit tests use SQLite fallback (vector cols degrade to JSON). Live gates (`atlas doctor`, `verify ingest` on real PDF, live `generate`) wired + runnable but skip cleanly when db/LLM keys/docker absent.
- No LLM keys → llm/embed clients use fake transport in tests.

## Waves (sequential; parallel within)
- **W0 Foundation** (1 agent): repo layout, pyproject, config.py (all T-03 constants), FULL db schema + SQLAlchemy 2.0 models + alembic 0001, llm.py + embed.py, `atlas doctor`, verify.schema. Builds ALL shared files → later waves only ADD modules.  Tasks T-01..T-05.
- **W1a Ingestion** (parallel): manifest→parse→structure→chunk→embeddings→concepts→review TUI→objectives. `src/atlas/ingest/`. Tasks T-10..T-17.
- **W1b Adaptive** (parallel): elo+retention, attempt service, scheduler, predict, study/flash TUI, verify adaptive+replay. `src/atlas/learn/`. Tasks T-30..T-35.
- **W2 Generation+caselets**: targets, prompts+gen call, gates G1/G4/G5, gates G2/G3, generate+dispute TUI, calibrate, caselet engine+G6. `src/atlas/generate/`. Tasks T-20..T-26.
- **W3 Mock/status/diagnostic/refill**: diagnose, mock, status, full-pattern mock+strategy, refill. Tasks T-36,T-40,T-41,T-42,T-50.
- **W4 Verify**: full pytest + ruff + every ⊢ pytest-level gate + coverage report + BUILDLOG.md.

## Invariants (README)
1. Nothing ships unproven — gate written as code before feature; gate evidence in gate_log; mastery replayable.
2. Concepts are the unit of mastery (keys everywhere).
3. Original content only — sources ground, never copied (G4) / redistributed.
4. Engine is the permanent asset; TUI/CLI disposable shells.
5. NO stub/dummy/placeholder/TODO code — production-real or say so.
