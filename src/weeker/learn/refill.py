"""``weeker refill`` — the nightly loop glue (016 §T-50).

One cron-able command for the evening ritual: overgenerate weak-concept questions
(``generate --refill``), re-run the bank gate, and build the status snapshot — so
a single invocation both replenishes the bank and reports where the learner
stands. Every dependency (generation, verify, status) is the committed real
function; the LLM/embed transports are injectable so the same code runs live
(real providers) and green in tests (fakes).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from weeker.core.embed import EmbedTransport
from weeker.core.llm import Transport
from weeker.core.models import Course
from weeker.generate.pipeline import GenerationReport, run_generation
from weeker.learn import status
from weeker.verify.bank_checks import BankReport, verify_bank


@dataclass
class RefillResult:
    """The three-step outcome: what was generated, the bank gate, and the status."""

    generation: GenerationReport
    bank: BankReport
    status: status.StatusReport


def run_refill(
    db: Session,
    course: Course,
    user_id: str,
    *,
    count: int = 40,
    consensus: int = 1,
    caselets: bool = False,
    now: datetime | None = None,
    rng: random.Random | None = None,
    generator_model: str | None = None,
    verifier_model: str | None = None,
    solver_model: str | None = None,
    llm_transport: Transport | None = None,
    embed_transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
) -> RefillResult:
    """Generate-refill → verify bank → build status, in one transaction-friendly pass."""
    now = now or datetime.now(UTC)
    generation = run_generation(
        db,
        course,
        user_id,
        count=count,
        refill=True,
        consensus=consensus,
        generator_model=generator_model,
        verifier_model=verifier_model,
        solver_model=solver_model,
        llm_transport=llm_transport,
        embed_transport=embed_transport,
        cache_dir=cache_dir,
    )
    bank = verify_bank(db, course.id, caselets=caselets)
    report = status.build_status(db, user_id=user_id, course=course, now=now, rng=rng)
    return RefillResult(generation=generation, bank=bank, status=report)
