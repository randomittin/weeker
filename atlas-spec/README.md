# Atlas Spec — Execution Set (010–015)

These five documents extend your 000–009 foundation set and are written to be executed,
not admired. Reading order = build order.

| Doc | What it specifies | You can start when |
|---|---|---|
| [010 Execution Plan](010_execution_plan.md) | Track A vs Track B split, milestones M0–M6, gates, anti-goals, copyright policy, calendar | now |
| [011 Ingestion Pipeline](011_ingestion_pipeline.md) | PDF corpus → parsed pages → chapter tree → chunks → embeddings → reviewed concepts → objectives + blueprint weights. Full prompts, JSON schemas, per-stage gates, failure playbook | M0 done |
| [012 Generation Engine](012_generation_engine.md) | Question contract, generation prompts, gate chain G1–G5 (schema, grounding, blind solve, originality, dedup), Elo difficulty calibration, telemetry & retirement | M2 done |
| [013 Adaptive Engine](013_adaptive_engine.md) | Concrete math: confidence-weighted Elo ability, FSRS-lite retention, scheduler priority formula, session/mock composition, misconception flow, score prediction with CI, synthetic-replay verification | M3 done |
| [014 Database Schema](014_database_schema.md) | Complete DDL — 20 tables, pgvector indexes, gate_log auditability, replayable mastery | M0 (migrate first) |
| [015 Build Guide](015_build_guide.md) | Repo layout, CLI surface, TUI sketches, Track B API contract, day-by-day plan, tuning notes | alongside everything |
| [016 Ultraplan](016_ultraplan.md) | The executable task DAG: 28 tasks T-01…T-50 with per-task acceptance gates, config defaults, interfaces, critical path, ADR stubs, agent execution protocol | now — this is what you hand to Claude Code |
| [prompts/](prompts/) | The five LLM prompt files referenced by 011/012/016, ready to drop into the repo | with T-12/T-15/T-17/T-21/T-23 |

Core invariants across the set:

1. **Nothing ships unproven.** Every milestone has a `atlas verify …` gate written as
   code before the feature. Every question carries its gate evidence in `gate_log`.
   Mastery state is replayable from the attempt log.
2. **Concepts are the unit of mastery.** Users master concepts, never questions
   (per 005). Questions, flashcards, sessions, predictions all key on concept ids.
3. **Original content only.** Source PDFs ground and explain; they are never copied
   into questions (012 §G4) and never redistributed (010 §5).
4. **The learning engine is the permanent asset** (per 000). UIs — TUI now, web later —
   are disposable shells around `ingest/`, `generate/`, `learn/`.
