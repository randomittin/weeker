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

# BUILDLOG — Wave 2b (Generation engine + caselets)

| Task | What | Gate | Result |
|---|---|---|---|
| T-20 | `generate/targets.py` — `allocate_targets` (largest-remainder Hamilton apportionment, min-3 floor, mastery≥0.9 refill exclusion, refill eff-weight = weight×(1−mastery)) + `compute_targets` (blueprint chapter weight shared per concept, mastery from `elo.mastery(theta)`) | `pytest -q tests/test_generate_targets.py` | PASS (6 tests) |
| T-21 | `generate/prompts.py` — `GeneratedQuestion` pydantic contract (exactly 4 opts A–D, one keyed-correct, explanation, named misconception per distractor) + `render_question_prompt` (prompts/question_gen.txt verbatim) + `generate_batch`/`generate_for_concept` (top-4 concept→chunk grounding, batch-of-GEN_BATCH parse) | `pytest -q tests/test_generate_prompts.py` (contract round-trip + recorded 5-object fixture) | PASS (7 tests) |
| T-22 | `generate/gates.py` — G1 (structural + all/none ban + option distinctness OPTION_DISTINCT_MAX_COS), G4 (`build_ngram_index` 8-gram scan OCR-excluded + stem↔chunk G4_STEM_CHUNK_MAX_COS), G5 (dedup G5_DUP_COS + variant tag G5_VARIANT_COS); each records reason into gate_log | `pytest -q tests/test_generate_gates.py` (crafted 3-opt / two-correct / verbatim lift / 0.91 dup / 0.84 variant) | PASS (11 tests) |
| T-23 | `generate/gates.py` — G2 grounding verifier (prompts/grounding_verify.txt → supported/ambiguous), G3 blind solve on MODEL_SOLVER_BLIND with `--consensus` (pass iff unanimous-with-key; else `disputed`) | `pytest -q tests/test_generate_gates_llm.py` (G2 supported/ambiguous/unsupported; G3 agree/disagree/2-of-3/3-of-3) | PASS (8 tests) |
| T-24 | `generate/pipeline.py` (overgenerate ×GEN_OVERGEN_FACTOR → gate chain → persist active/disputed; course n-gram index OCR-excluded), `generate/review_tui.py` `DisputeApp` (k keep / 1-4 fix-key / x discard, resumable), `verify/bank_checks.py` `verify_bank` = **M3 gate** (schema/grounding 100%, blind ≥95%, near-dup <2%, zero G4), `generate/cli.py` run/review/verify-bank | `pytest -q tests/test_generate_pipeline.py tests/test_generate_bank.py tests/test_generate_review.py` | PASS (3+6+2 tests) |
| T-25 | `generate/calibrate.py` — `initial_q_rating` (difficulty→logit, consumed by pipeline), `apply_rating`/`apply_rating_value` (live Elo hook off `UpdateResult.q_rating`), `needs_recalibration` (>1.5 logit drift), `dead_distractors` (zero pulls @≥20), `compute_discrimination`/`should_retire` (point-biserial <0.05 @≥20), `audit_bank`; CLI `recalibrate` | `pytest -q tests/test_generate_calibrate.py` (30-attempt convergence, dead-distractor + low-disc streams) | PASS (8 tests) |
| T-26 | `generate/caselets.py` — `GeneratedCaselet` contract (120-220-word scenario + 5 positioned 1|2-mark Qs), G4-on-scenario, per-Q G1/G4/G5/G2, **G6** scenario-dependence (blind-solve sans scenario → regenerate independents), combined caselet G3, `run_caselets` (concept-group select + regen loop), `caselet_drill`; `verify_bank(caselets=True)` (≥25 active / 100% G6 / per-Q green); CLI `caselets` + `verify-bank --caselets` | `pytest -q tests/test_generate_caselets.py` | PASS (10 tests) |

Wave 2b totals: +61 tests (103 → 164), ruff clean. Commits: one per task, `feat(generate): T-xx …`.

Wave-2b notes:
- `CaseGroup.concept_ids` (UuidArray) is stored as **stringified** UUIDs — `ARRAY(Uuid())` accepts them on Postgres and the SQLite JSON fallback cannot serialize raw `uuid.UUID`.
- Question-contract field list (Wave 3 mock composer): `stem:str`, `options:list[{key:'A'|'B'|'C'|'D', text:str, misconception:str|None}]`, `correct_key:'A'|'B'|'C'|'D'`, `explanation:str`, `difficulty:1|2|3`; caselet adds `case_position:1..5`, `marks:1|2`. `GeneratedQuestion.option_dicts()` yields the `Question.options` jsonb shape.
- `verify_bank(session, course_id, *, caselets=False) -> BankReport(.passed, .total_active, .schema_ok, .grounding_ok, .blind_agreement, .near_dup_rate, .g4_hits, .disputed_count, .caselets_active, .failures)`.

## Environment notes
- Python 3.11 (pinned `>=3.11,<3.12`); no PEP695 / 3.12-only syntax.
- No Docker: Postgres+pgvector is production; SQLite is the test fallback via the
  `Vector` TypeDecorator (→ `vector(dim)` on Postgres, JSON on SQLite) and
  `.with_variant` for JSONB/arrays. pgvector HNSW indexes are guarded to Postgres
  in migration 0001; `verify.schema` skips the pgvector-index assertion on SQLite.
- No LLM/embed keys: transports are injectable; tests use fakes. Real HTTP path
  present (OpenAI-compatible) but not exercised.
