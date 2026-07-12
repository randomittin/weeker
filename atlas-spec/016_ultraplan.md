# 016 Ultraplan — Executable Task Graph

Version: 1.0
Status: Active
Consumes: 010–015. This is the layer between spec and code: a dependency-ordered task
DAG fine-grained enough to hand to a coding agent (Claude Code) one task at a time,
with an acceptance command per task. A task is done when its gate command exits 0 —
no gate, no merge. Run it exactly like a Heimdall build queue.

Legend: `T-xx` task id · `⇐` depends on · `⊢` acceptance gate (must be written/run
as part of the task) · effort in agent-sessions (one focused Claude Code run).

---

## Phase 0 — Scaffold (M0)

**T-01 · Repo + toolchain** · ⇐ — · 0.5
Create layout per 015 §1. `pyproject.toml` (python 3.12; deps: sqlalchemy, alembic,
psycopg, pgvector, pymupdf, typer, rich, textual, httpx, numpy, pydantic v2, pytest,
pytest-asyncio, ruff). Pre-commit: ruff + a gitleaks hook (you already know why).
⊢ `pip install -e . && ruff check . && pytest -q` (0 tests ok)

**T-02 · docker-compose + migrations** · ⇐ T-01 · 0.5
Postgres 16 + pgvector image; alembic init; migration 0001 = full DDL from 014 verbatim.
⊢ `docker compose up -d && alembic upgrade head && python -m atlas.verify.schema`
(verify.schema introspects and asserts all 20 tables + the partial unique index
`one_correct_per_question` exist)

**T-03 · config.py — every constant, named** · ⇐ T-01 · 0.5
Single source of tunables. Defaults (change here, nowhere else):

```python
# ingestion
CHUNK_TARGET_TOKENS = 900;  CHUNK_MIN = 300;  CHUNK_MAX = 1400;  CHUNK_OVERLAP = 0.15
HEADER_FOOTER_PAGE_SHARE = 0.30
OCR_MIN_CHARS = 40;  OCR_DPI = 200
CONCEPT_MERGE_COS = 0.92;  CONCEPT_CHUNK_MIN_COS = 0.60;  CONCEPT_CHUNK_TOPK = 6
# generation gates
GEN_BATCH = 5;  GEN_OVERGEN_FACTOR = 1.3
G4_NGRAM = 8;  G4_STEM_CHUNK_MAX_COS = 0.90
G5_DUP_COS = 0.90;  G5_VARIANT_COS = 0.83
OPTION_DISTINCT_MAX_COS = 0.85
# adaptive (013)
ELO_K_BASE = 0.6;  ELO_K_MIN = 0.08;  ELO_K_MAX = 0.6;  ELO_KQ_BASE = 0.15
CONF_MULT = {("correct","sure"):1.0, ("correct","unsure"):0.7, ("correct","guessing"):0.25,
             ("wrong","sure"):1.6, ("wrong","unsure"):1.0, ("wrong","guessing"):0.6}
MASTERY_TARGET_LOGIT = 0.3
STAB_GROW_BASE = 1.4;  STAB_GROW_MASTERY = 1.8;  STAB_FAIL_MULT = 0.35;  STAB_MIN = 0.5
DUE_R_THRESHOLD = 0.85
PRIORITY_W = dict(weak=0.45, overdue=0.30, misconception=0.20, novelty=0.05)
SOFTMAX_T = 0.7;  DESIRABLE_DIFFICULTY_OFFSET = 0.3;  RESERVE_DAYS = 3
SESSION_MIX = dict(warmup=3, core=21, mixed=4, review=2)
PREDICT_SE_BASE = 1.2;  PREDICT_BOOTSTRAP = 1000;  PREDICT_MIN_COVERAGE = 0.30
# models (fill in your providers)
MODEL_GENERATOR = "…";  MODEL_VERIFIER_CHEAP = "…";  MODEL_SOLVER_BLIND = "…-different-family"
MODEL_EMBED = "…";  EMBED_DIM = …
```
⊢ `python -c "from atlas.core import config"` + a test asserting no module besides
config defines a float/int literal used by gates (grep-based lint, warn-only).

**T-04 · llm.py + embed.py thin clients** · ⇐ T-03 · 0.5
`complete(model, system, user, json_schema=None, max_retries=2) -> dict|str` with
strict-JSON retry (re-prompt with the validation error appended). `embed(list[str])
-> np.ndarray` with batching + on-disk cache keyed by sha256(text).
⊢ pytest with a fake transport: malformed-JSON-then-valid retry path covered;
embed cache hit avoids second call.

**T-05 · `atlas doctor`** · ⇐ T-02,T-04 · 0.25
Checks: db reachable, migrations current, model keys present, one live 1-token
completion per configured model, embed dim matches EMBED_DIM.
⊢ `atlas doctor` exit 0 — **this is the M0 gate**

## Phase 1 — Ingestion (M1–M2, doc 011)

**T-10 · manifest.py (S0)** · ⇐ T-05 · 0.5
Pydantic models for manifest.yaml; loader; course+sources rows upsert by sha256.
⊢ pytest: valid manifest loads; missing file, bad blueprint sum, dup sha rejected.

**T-11 · parse.py (S1)** · ⇐ T-10 · 1
PyMuPDF page extraction with blocks/spans; de-hyphenation; header/footer strip;
table serialization; OCR fallback behind `OCR_MIN_CHARS`.
⊢ `atlas verify ingest --stage parse` on the NISM PDF: coverage ≥ 0.98, zero
silent-empty pages; unit tests on de-hyphenation and header detection with fixtures.

**T-12 · structure.py (S2)** · ⇐ T-11 · 1
Strategy chain: outline → font heuristic → LLM (prompt file `prompts/structure.txt`).
⊢ structure gate from 011 §S2 as code; fixture PDFs exercising each strategy
(build tiny synthetic PDFs with fitz in the test itself).

**T-13 · chunk.py (S3)** · ⇐ T-12 · 1
Section-bounded chunking, atomic tables/examples, content_hash, overlap.
⊢ token histogram in range; coverage ≥ 0.97; a property test: no chunk crosses a
chapter boundary for randomized synthetic chapter maps.

**T-14 · embeddings (S4)** · ⇐ T-13 · 0.5
Chunk embeddings → pgvector; HNSW index build; smoke retrieval per chapter.
⊢ `atlas verify ingest` full — **M1 gate**

**T-15 · concepts.py (S5 map+reduce)** · ⇐ T-14 · 1.5
Chapter-batched extraction (prompt `prompts/concept_extract.txt` = 011 §S5 verbatim),
embedding merge at `CONCEPT_MERGE_COS`, prereq resolution, DAG check,
concept→chunk retrieval mapping.
⊢ pytest on merge logic with synthetic near-dupes; DAG cycle detection test;
end-to-end run leaves review_status='pending' rows only.

**T-16 · review TUI (S5 human pass)** · ⇐ T-15 · 1
Textual app per 011 sketch: accept/edit/merge/split/drop/peek-source; writes
review_status + reviewed_at; resumable.
⊢ scripted textual pilot test drives a/e/d keys against 3 fixture concepts and
asserts DB state. Then **you** run the real 45-minute pass.

**T-17 · objectives + blueprint (S6)** · ⇐ T-16 · 0.5
Prompt `prompts/objectives.txt`; weight formula + manifest overrides; normalization.
⊢ `atlas verify concepts` — **M2 gate**

## Phase 2 — Generation (M3, doc 012)

**T-20 · targets.py** · ⇐ T-17 · 0.5
Initial-fill and refill quota logic per 012 §2.
⊢ pytest: quotas sum to N; min-3 floor; mastery≥0.9 exclusion; refill weights follow
`weight×(1−mastery)` on a synthetic mastery table.

**T-21 · prompts.py + generation call** · ⇐ T-20 · 1
Prompt file `prompts/question_gen.txt` (012 §3 verbatim); retrieval of top-4 chunks;
batch-of-5 parsing into the question contract (pydantic).
⊢ contract round-trip test; a recorded-fixture generation parses to 5 valid objects.

**T-22 · gates.py — G1, G4, G5 (pure code)** · ⇐ T-21 · 1
G1 invariants incl. option-distinctness and the all/none-of-the-above ban; G4 8-gram
scan (build a corpus n-gram set once per course, normalized, OCR-excluded) + stem-chunk
cosine; G5 stem dedup + variant tagging.
⊢ pytest per gate with crafted violations: 3-option question, two-correct, verbatim
8-gram lift, 0.91-cos near-dupe, 0.85 variant. Each must fail the right gate and
record the reason into gate_log.

**T-23 · gates.py — G2, G3 (LLM)** · ⇐ T-22 · 1
G2 grounding verifier (prompt `prompts/grounding_verify.txt`); G3 blind solve on a
different model family, `--consensus 3` mode, disputed-status routing.
⊢ fixture-transport tests: supported/ambiguous G2 paths; G3 agree, disagree→disputed,
2/3 consensus→disputed, 3/3→pass.

**T-24 · `atlas generate` + disputed review TUI** · ⇐ T-23 · 0.5
Pipeline wiring with `GEN_OVERGEN_FACTOR`; side-by-side dispute screen (keyed answer
rationale vs blind solver rationale; keep/fix-key/discard).
⊢ `atlas generate --count 40` live against the real corpus, then
`atlas verify bank` — **M3 gate** (schema 100%, grounding 100%, blind agreement ≥95%
after dispute resolution, near-dup <2%, zero G4 hits among active)

**T-25 · calibrate.py** · ⇐ T-24 · 0.5
q_rating init from authored difficulty; live Elo-side update hook (shared with T-31);
`recalibrate` flagging at >1.5 logit divergence; distractor pull counters; retirement
rules (discrimination <0.05 @20).
⊢ pytest: simulated 30 attempts move q_rating toward empirical; dead-distractor and
low-discrimination flags fire on crafted streams.

## Phase 3 — Adaptive engine + study loop (M4, doc 013)

**T-30 · elo.py + retention.py** · ⇐ T-05 · 1  *(parallelizable with Phase 2)*
Exact formulas from 013 §2–3. Public interface:

```python
def update(theta, q_rating, attempts, q_attempts, correct: bool,
           confidence: Literal["sure","unsure","guessing"]) -> UpdateResult
    # returns theta', q_rating', deltas — pure function, no I/O
def mastery(theta) -> float
def retrievability(stability, days_since) -> float
def stability_update(stability, mastery, correct, confidence) -> float
```
⊢ property tests: mastery∈[0,1]; theta bounded; conf-wrong |Δ| > guess-wrong |Δ|;
stability monotone over 3 successes; K decays with attempts.

**T-31 · attempt service** · ⇐ T-30,T-02 · 0.5
`record_attempt(...)` — transactional: attempts row (theta_before/after), mastery
upsert, q_rating + pull_count update, misconception flow steps 1–3 (flag, tag,
flashcard upsert), analytics event.
⊢ pytest: one call mutates exactly the expected rows; misconception fires only on
wrong+sure; flag clears after 2 confident-correct.

**T-32 · scheduler.py** · ⇐ T-31 · 1
Priority formula, softmax sampling, desirable-difficulty pick, 3-day reserve,
variant exclusion, session mix; mock composer with blueprint stratification,
difficulty ramp, adjacency rule.
⊢ statistical test: over 500 sampled sessions on a synthetic state, serve share per
concept correlates (ρ>0.8) with priority; zero variant pairs; mock per-chapter counts
match blueprint ±1.

**T-33 · predict.py** · ⇐ T-30 · 0.5
Expected score + bootstrap CI + P(pass) + negative marking; coverage suppression.
⊢ the 013 §8 calibration assertion (uniform 0.8 learner → within ±8%).

**T-34 · study TUI (`atlas study`, `atlas flash`)** · ⇐ T-32 · 1.5
Per 015 §3: one question per screen, a–d answer, 1/2/3 confidence, explanation block,
`p` peek-source (page-ranged chunk render), end-of-session delta report. Flashcard
drain mode.
⊢ textual pilot test scripts a full 5-question session against fixtures; DB attempts
verified. Then a real session by you.

**T-35 · `atlas verify adaptive` + `atlas replay`** · ⇐ T-31–T-33 · 0.5
Synthetic 200-attempt replay (013 §8) and full mastery-recompute diff.
⊢ both exit 0 — **M4 gate**

## Phase 4 — Mock + status (M5)

**T-40 · `atlas mock`** · ⇐ T-32,T-33 · 1
Timed exam mode, pause/resume (session config jsonb), negative marking, no feedback
until submit, confidence capture, post-mock review ordered confident-wrong-first,
prediction snapshot stored in `predicted_before`.
⊢ e2e: scripted 135-answer run produces mock_exams row with correct raw score and
deduction against a known answer set.

**T-41 · `atlas status`** · ⇐ T-33 · 0.5
Heatmap by chapter/concept, due counts, flags, prediction band, tonight's refill plan.
⊢ golden-output test on a fixture state — **M5 gate** = M4 green + T-40 + T-41 green.

## Phase 5 — nightly loop glue

**T-50 · `atlas refill` cron-able command** · ⇐ T-24,T-32 · 0.25
`generate --refill` + bank verify + status print, one command for the evening ritual.
⊢ runs green end-to-end on live data.

---

## Critical path & parallelism

```
T-01→02→05→10→11→12→13→14→15→16(you)→17→20→21→22→23→24→(study begins)
                     T-30→31→32→34 runs in parallel from T-05
```
Critical path ≈ 8.5 agent-sessions; with the adaptive branch parallel, wall-clock
lands on the Day-1/Day-2 split in 015 §5 if you run two worktrees. If you must
serialize, cut T-16's polish (accept/drop only) and T-34's flashcard mode to Day 3.

## Pre-written ADR stubs (decide-by-doing; record one paragraph each)

- ADR-001 embedding model + dim (locks `courses.embedding_dim`)
- ADR-002 generator/verifier/solver model assignment (G3 requires ≠ family)
- ADR-003 Postgres-in-Docker confirmed over SQLite (default: yes, per 014)
- ADR-004 NISM official mark distribution → blueprint overrides (enter real numbers)
- ADR-005 consensus mode default for calculation questions (default: on, 3-way)

## Agent execution protocol (how to actually run this)

One task per agent run. The run prompt is always:
*"Implement T-xx per atlas-spec/016 §T-xx and the referenced spec section. Write the
acceptance gate first, then the implementation. Do not modify config constants or
other tasks' files. Stop when `⊢` passes; print the gate output."*
Keep a `BUILDLOG.md` with task id → commit → gate output hash. If a later task breaks
an earlier gate, the earlier gate wins — revert or fix forward, never skip. That's the
whole discipline; it's the same one you run on Heimdall, pointed at your own exam.
