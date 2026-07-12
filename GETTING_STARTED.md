# Getting Started — Weeker (local, NISM Series X-A)

Run Weeker **locally** (no Docker, SQLite) to prep for the **NISM Series X-A:
Investment Adviser (Level 1)** exam. This runbook is verified end-to-end: the
full pipeline (ingest → generate → study → mock) was smoke-proven with fake
transports before you add the two things only you can supply — the **workbook
PDF** and **API keys**.

## Prereqs

- Python 3.11, the venv already created at `./.venv`.
- No Docker. State lives in a local SQLite file `./weeker.db` (the default
  `DATABASE_URL`, see `src/weeker/core/db.py`). It is already created, migrated,
  and empty — ready for your data.

```bash
cd /Users/rj/Downloads/code/weeker
source .venv/bin/activate
pip install -e ".[dev]" -q          # once
export DATABASE_URL=sqlite:///./weeker.db   # default; export makes it explicit
alembic upgrade head                # idempotent; already at head 0001
weeker doctor                       # db + migrations OK; model/embed keys SKIP until step 1
```

`weeker doctor` on the clean box:

```
db_reachable        OK    sqlite:///./weeker.db
migrations_current  OK    at head 0001
model_keys          SKIP  MODEL_* still placeholders
live_completion     SKIP  no models/keys configured
embed_dim           SKIP  WEEKER_EMBED_API_KEY not set
no hard failures
```

---

## STEP 1 — API keys (free tier, OpenAI-compatible)

Weeker talks to any **OpenAI-compatible** `/chat/completions` and `/embeddings`
endpoint (`src/weeker/core/llm.py`, `embed.py`). Config is 100% env-driven
(`src/weeker/core/config.py`) — no code changes.

**Free option: Google Gemini** (get a key at https://aistudio.google.com/apikey).
Its OpenAI-compatible base serves both chat and embeddings.

Paste these exports (env var names are the ACTUAL names read by `config.py`):

```bash
# ── LLM (generator + cheap verifier + blind solver all share this base+key) ──
export WEEKER_LLM_API_KEY="<your-gemini-key>"
export WEEKER_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
export WEEKER_MODEL_GENERATOR="gemini-2.0-flash"
export WEEKER_MODEL_VERIFIER_CHEAP="gemini-2.0-flash-lite"
export WEEKER_MODEL_SOLVER_BLIND="gemini-2.5-flash"   # a DIFFERENT model from the generator

# ── Embeddings ──
export WEEKER_EMBED_API_KEY="<your-gemini-key>"       # same key is fine
export WEEKER_EMBED_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
export WEEKER_MODEL_EMBED="text-embedding-004"        # NOTE: var is WEEKER_MODEL_EMBED
export WEEKER_EMBED_DIM=768                            # text-embedding-004 → 768 dims
```

Then:

```bash
weeker doctor    # model_keys → OK, live_completion → OK (3 models reply), embed_dim → OK (dim=768)
```

> **Blind-solver limitation (be honest about this).** The G3 blind solver uses
> the **same** `WEEKER_LLM_BASE_URL` + `WEEKER_LLM_API_KEY` as the generator —
> there is no separate solver-base-url env var. So you can only make the solver a
> different **model** on the **same provider** (as above), not a truly different
> **family** on a different provider (e.g. Groq Llama). The G3 gate does not
> enforce family difference in code; using a distinct model still gives an
> independent second opinion. A cross-provider solver would need a code change to
> give the solver its own transport.

Any OpenAI-compatible provider works the same way — just change the two
`*_BASE_URL`s, the model names, and `WEEKER_EMBED_DIM` to match your embedding
model's dimension. All optional knobs: `WEEKER_EMBED_BATCH` (default 64),
`WEEKER_CACHE_DIR` (default `.weeker_cache`).

---

## STEP 2 — Add the workbook PDF (and, optionally, real weightage)

1. Drop the official NISM X-A workbook PDF here (exact path):

   ```
   courses/nism-xa/nism-xa-workbook.pdf
   ```

   The manifest at `courses/nism-xa/manifest.yaml` already references it and
   encodes the verified exam pattern (90 standalone 1-mark + 6×5 1-mark caselets
   + 3×5 2-mark caselets = 150 marks, pass 90, 180 min, 25% negative marking).

2. (Optional, after your first ingest) Fix the chapter weightage. The
   `blueprint.overrides` in the manifest are **EVEN placeholders (0.1 each)** —
   not the official NISM weightage. After `weeker ingest run` prints the detected
   chapter titles, edit the manifest so each override key matches a detected
   title and the weights reflect the real per-chapter exam weightage (they must
   sum to 1.0). Leave `blueprint: {}` to auto-derive weights from concept counts.

---

## STEP 3 — Run the real prep flow (exact commands)

Commands/flags below are verified against the registered CLI (`weeker --help`,
`weeker <group> --help`).

```bash
# 1) Ingest: parse → structure → chunk → embed → concepts → objectives+blueprint
weeker ingest run --manifest courses/nism-xa/manifest.yaml

# 2) Human concept review (~45 min): accept / edit / drop each pending concept
weeker ingest review --course nism-xa

# 3) Generate the question bank, then the caselets, then gate-verify both
weeker generate run --count 300 --course nism-xa
weeker generate caselets --count 30 --two-mark 10 --course nism-xa
weeker generate verify-bank --course nism-xa --caselets

# 4) Day-0 diagnostic (silent measurement; seeds your initial ability)
weeker mock diagnose --course nism-xa          # spec name is "weeker diagnose"

# 5) Daily adaptive study + due flashcards
weeker study run --course nism-xa
weeker study flash --course nism-xa

# 6) Full-pattern 150-mark mock, then the dashboard (heatmap + P(pass) + skip strategy)
weeker mock full --course nism-xa
weeker status show --course nism-xa
```

Useful extras:

- `weeker status refill --course nism-xa` — nightly loop: generate refill → bank
  verify → status print (spec name `weeker refill`).
- `weeker mock run --course nism-xa` / `weeker mock resume` — timed standalone
  mock, pausable/resumable.
- `weeker verify all` — run every gate (schema/ingest/concepts/bank/adaptive).

> Command-name deviations (both documented in code): the spec's `weeker diagnose`
> is wired as `weeker mock diagnose`, and `weeker refill` as
> `weeker status refill` — the CLI root has fixed groups.

---

## What costs money / free-tier limits

- Every ingest **embedding** call and every **generation/verification/solve**
  call hits your provider. Ingesting a full workbook (embed all chunks + extract
  concepts + write objectives) and generating a few hundred questions + caselets
  (each question runs G2 grounding and G3 blind-solve LLM calls) is the heavy
  part — pace it against your free-tier rate/day limits.
- Studying, mocks, `status show`, and scoring are **pure local compute** — no API
  calls, no cost.
- Embeddings are cached on disk (`WEEKER_CACHE_DIR`), so re-ingesting the same
  text does not re-bill.

---

## Was this proven?

Yes. Before you add keys/PDF, the whole wiring was smoke-run against injectable
fake transports and a tiny synthetic PDF into a throwaway `/tmp` SQLite DB
(the real `./weeker.db` stayed empty): ingest produced chunks + concepts +
objectives, generation produced gate-logged questions and caselets, a study
attempt recorded through the real write path, the Day-0 diagnostic seeded
mastery, and `weeker mock full` composed the full pattern with a P(pass)
prediction — with `weeker status show` rendering the dashboard from a fresh
process reading the file DB.
