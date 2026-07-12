# 017 Patch — Caselets, Diagnostic, Exam Strategy

Version: 1.0
Status: Active
Patches: 010 (manifest), 012 (generation), 013 (scheduler/prediction), 014 (schema),
016 (adds tasks T-26, T-36, T-42)

Reason: the verified NISM XA structure is 90 standalone MCQs (1 mark) + 9 caselets
(6 caselets × 5 questions × 1 mark, 3 caselets × 5 questions × 2 marks) = 135
questions, 150 marks, 180 minutes, pass at 90/150, negative marking 25% of the
question's marks. Caselets are 40% of available marks and were unmodeled. This patch
closes that, plus the diagnostic and attempt/skip strategy gaps.

---

## 1. Corrected manifest exam block (replaces 010 §S0)

```yaml
exam:
  duration_minutes: 180
  passing_marks: 90
  total_marks: 150
  negative_marking_fraction: 0.25       # of the question's own marks
  sections:
    - kind: standalone
      questions: 90
      marks_each: 1
    - kind: caselet
      count: 6
      questions_each: 5
      marks_each: 1
    - kind: caselet
      count: 3
      questions_each: 5
      marks_each: 2
```

Confirm chapter-wise weightage from the official exam outline in the enrolled
workbook and enter it under `blueprint.overrides` (ADR-004).

## 2. Schema delta (append to 014)

```sql
create table case_groups (
  id uuid primary key default gen_random_uuid(),
  course_id uuid not null references courses(id) on delete cascade,
  scenario text not null,                  -- the caselet narrative, 120-220 words
  concept_ids uuid[] not null,             -- 3-5 concepts the caselet spans
  status text not null default 'active'
    check (status in ('active','disputed','retired')),
  gate_log jsonb not null default '{}',
  generator_model text not null,
  created_at timestamptz not null default now()
);

alter table questions add column case_group_id uuid references case_groups(id);
alter table questions add column marks numeric(3,1) not null default 1;
alter table questions add column case_position int;   -- 1..5 within the caselet
```

Serving rule: caselet questions are served only inside mocks and caselet-drill
sessions — never mixed into adaptive practice singles (the scenario context is the
point). Each caselet question still keys to exactly one primary concept, so all
mastery math in 013 applies unchanged; a caselet is a *packaging* of five concept
probes behind one scenario.

## 3. Caselet generation (012 addendum, prompt `prompts/caselet_gen.txt`)

Target selection: pick 3–5 concepts that plausibly co-occur in one client situation
(constraint: same or adjacent chapters, or the classic cross-cutting sets — e.g.
suitability + asset allocation + taxation). Weight toward weak concepts on refill,
same as standalone.

Prompt core:

```
Write ONE original client scenario (an Indian investor case: age, income, goals,
holdings, constraints — 120-220 words, realistic numbers) and exactly 5 questions
answerable ONLY by applying the scenario's facts to the listed concepts. Each
question follows the standard 4-option contract. At least 2 questions must require
combining two scenario facts (not lookup). No question may be answerable without
reading the scenario. Marks per question: {1|2}. 2-mark questions must require
calculation or multi-step reasoning.
```

Gate chain: G1/G4/G5 unchanged per question (G4 additionally runs on the scenario
text). G2 grounding verifies each keyed answer against scenario + source chunks.
G3 blind solve receives **scenario + all 5 questions together** (that's how the
exam presents them). New **G6 — scenario dependence**: the blind solver answers each
question a second time *without* the scenario; any question it still answers
correctly with confidence is flagged "scenario-independent" and regenerated —
otherwise your caselets are just five MCQs wearing a costume.

Bank target: ≥ 25 active caselets (≥ 10 at 2-mark difficulty) before the first
full-pattern mock.

## 4. Mock composer update (013 §4.3 replacement)

A full mock is now: 90 standalone (blueprint-stratified as before) + 6 one-mark
caselets + 3 two-mark caselets, caselets sampled to maximize coverage of currently
weak concepts, no caselet reused within 5 days. Scoring: per-question marks with
negative deduction = 0.25 × marks of that question. Mock review orders
confident-wrong first, caselets grouped.

Prediction update (013 §7): expected score is computed per section —
`Σ P(correct|theta_c) × marks − Σ P(attempt ∧ wrong) × 0.25 × marks` — using the
attempt policy in §6 below rather than assuming every question is attempted.

## 5. Day-0 diagnostic (new, closes the theta-seeding stub)

`atlas diagnose` — runs once, before the first study session:

- 36 standalone questions: 3 per chapter (12 chapters), one per authored difficulty
  tier, chosen for maximum concept spread (greedy max-coverage over concept ids).
- No feedback during the run (it's measurement, not practice); full review after.
- Seeding: for each *tested* concept run the normal Elo update from theta=0 with
  K fixed at 0.5. For untested concepts in the same chapter, seed
  `theta = 0.6 × mean(theta of tested chapter-mates)` (shrunk toward 0).
- Sets `misconception_pending` on confident-wrong exactly as normal.
- Output: the first real `atlas status` heatmap + tonight's refill plan.

Takes ~35 minutes and converts day 1 from blind coverage into targeted work.

## 6. Attempt/skip policy (exam strategy layer)

Negative marking EV, per question of m marks with subjective correctness p:
`EV(attempt) = p·m − (1−p)·0.25·m` → breakeven at **p = 0.2**. Practical policy,
coached in every mock review and printed on the mock summary:

- Can eliminate ≥ 1 option → attempt (p ≥ 1/3 > 0.2).
- True blind guess (p = 0.25) → marginally +EV but noise; skip on 2-mark caselet
  questions when time-pressured, attempt 1-markers.
- Track realized calibration: the confidence keystroke gives empirical p per
  confidence level ("your 'unsure' is actually 61% correct — always attempt
  unsure"). Print this calibration table on `atlas status` from day 3.

## 7. New ultraplan tasks

**T-26 · caselet engine** · ⇐ T-24 · 1.5 — case_groups schema migration, caselet
prompt + parsing, G6 scenario-dependence gate, caselet drill mode.
⊢ `atlas verify bank --caselets`: ≥25 active, 100% G6-passed, per-question gates green.

**T-36 · `atlas diagnose`** · ⇐ T-32 · 0.5 — selection, silent mode, seeding math.
⊢ pytest: coverage-greedy picks ≥ 30 distinct concepts; chapter-mate seeding
shrinks toward 0; reseeding is idempotent (refuses to run twice without --force).

**T-42 · full-pattern mock + strategy report** · ⇐ T-40,T-26 · 0.5 — section-aware
composer, marks-weighted scoring, calibration table, skip-policy summary.
⊢ e2e scripted run reproduces a hand-computed score including 2-mark negatives.

Revised calendar impact: T-26 adds ~1 session — run it Day 3 morning in parallel
with studying the standalone bank; first *full-pattern* mock moves to Day 3 evening
or Day 4 morning. Standalone-only mocks remain available from Day 2 night.
