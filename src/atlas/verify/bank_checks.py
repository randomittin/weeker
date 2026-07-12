"""Question-bank gates as code — the M3 gate (016 §T-24, 017 §T-26).

``verify_bank`` audits the *active* bank against the M3 acceptance bar:

* **schema 100%** — every active question re-passes the G1 structural invariants
  (exactly 4 options keyed A–D, exactly one correct, no all/none-of-the-above);
* **grounding 100%** — every active question recorded a passing G2;
* **blind agreement ≥ 95%** — at most 5% of the active bank is still G3-disputed
  after review;
* **near-duplicate < 2%** — recorded G5 duplicate flags among active questions;
* **zero G4 hits** — no active question carries a failed G4.

With ``caselets=True`` it additionally applies the caselet bar (017 §3): ≥ 25
active caselets, 100% G6-passed, and every caselet question green on its
per-question gates. Reads persisted rows via an injected session, so it runs
identically on SQLite (tests) and Postgres (production).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core.models import CaseGroup, Question
from atlas.generate import gates

BLIND_AGREEMENT_MIN = 0.95
NEAR_DUP_MAX = 0.02
CASELET_ACTIVE_MIN = 25
CASELET_TWO_MARK_MIN = 10


@dataclass
class BankReport:
    """M3 bank audit result."""

    total_active: int = 0
    schema_ok: bool = True
    grounding_ok: bool = True
    blind_agreement: float = 1.0
    near_dup_rate: float = 0.0
    g4_hits: int = 0
    disputed_count: int = 0
    caselets_active: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def _gate_passed(q: Question, gate: str) -> bool:
    return bool((q.gate_log or {}).get(gate, {}).get("passed", False))


def _g3_disputed(q: Question) -> bool:
    return bool((q.gate_log or {}).get("G3", {}).get("disputed", False))


def _g5_duplicate(q: Question) -> bool:
    return "duplicate_of" in (q.gate_log or {}).get("G5", {})


def verify_bank(
    session: Session, course_id: uuid.UUID, *, caselets: bool = False
) -> BankReport:
    """Run the M3 bank gate; returns a :class:`BankReport` (``.passed`` is the gate)."""
    report = BankReport()
    active = list(
        session.scalars(
            select(Question).where(
                Question.course_id == course_id,
                Question.status == "active",
                Question.case_group_id.is_(None),
            )
        )
    )
    report.total_active = len(active)
    if not active:
        report.failures.append("no active standalone questions in bank")
        return report

    # 1. schema 100% — re-run G1 structurally (no embedding needed for the invariants).
    schema_bad = []
    for q in active:
        res = gates.check_g1(
            stem=q.stem, options=q.options or [], correct_key=q.correct_key, gate_log={}
        )
        if not res.passed:
            schema_bad.append((q.id, res.reason))
    if schema_bad:
        report.schema_ok = False
        report.failures.append(
            f"schema: {len(schema_bad)}/{len(active)} active fail G1 "
            f"(e.g. {schema_bad[0][1]})"
        )

    # 2. grounding 100% — every active question passed G2.
    ungrounded = [q.id for q in active if not _gate_passed(q, "G2")]
    if ungrounded:
        report.grounding_ok = False
        report.failures.append(f"grounding: {len(ungrounded)}/{len(active)} active without G2 pass")

    # 3. blind agreement ≥ 95% post-dispute.
    disputed = [q.id for q in active if _g3_disputed(q)]
    report.disputed_count = len(disputed)
    report.blind_agreement = 1.0 - len(disputed) / len(active)
    if report.blind_agreement < BLIND_AGREEMENT_MIN:
        report.failures.append(
            f"blind agreement {report.blind_agreement:.3f} < {BLIND_AGREEMENT_MIN} "
            f"({len(disputed)} still disputed)"
        )

    # 4. near-duplicate < 2%.
    dups = [q.id for q in active if _g5_duplicate(q)]
    report.near_dup_rate = len(dups) / len(active)
    if report.near_dup_rate >= NEAR_DUP_MAX:
        report.failures.append(
            f"near-dup rate {report.near_dup_rate:.3f} ≥ {NEAR_DUP_MAX} ({len(dups)} flagged)"
        )

    # 5. zero G4 hits among active.
    g4 = [q.id for q in active if q.gate_log and "G4" in q.gate_log and not _gate_passed(q, "G4")]
    report.g4_hits = len(g4)
    if g4:
        report.failures.append(f"{len(g4)} active question(s) carry a failed G4")

    if caselets:
        _verify_caselets(session, course_id, report)

    return report


def _verify_caselets(session: Session, course_id: uuid.UUID, report: BankReport) -> None:
    """Caselet bar (017 §3), folded into an existing :class:`BankReport`."""
    groups = list(
        session.scalars(
            select(CaseGroup).where(
                CaseGroup.course_id == course_id, CaseGroup.status == "active"
            )
        )
    )
    report.caselets_active = len(groups)
    if len(groups) < CASELET_ACTIVE_MIN:
        report.failures.append(
            f"caselets: {len(groups)} active < {CASELET_ACTIVE_MIN} target"
        )

    two_mark = 0
    for g in groups:
        qs = list(
            session.scalars(
                select(Question).where(Question.case_group_id == g.id).order_by(Question.case_position)
            )
        )
        if len(qs) != 5:
            report.failures.append(f"caselet {g.id}: {len(qs)} questions (expected 5)")
        if any(float(q.marks) >= 2 for q in qs):
            two_mark += 1
        if not (g.gate_log or {}).get("G6", {}).get("passed", False):
            report.failures.append(f"caselet {g.id}: G6 scenario-dependence not passed")
        for q in qs:
            for gate in ("G1", "G2", "G4"):
                if not _gate_passed(q, gate):
                    report.failures.append(
                        f"caselet {g.id} q{q.case_position}: {gate} not passed"
                    )
                    break

    if groups and two_mark < CASELET_TWO_MARK_MIN:
        report.failures.append(
            f"caselets: {two_mark} two-mark caselets < {CASELET_TWO_MARK_MIN} target"
        )
