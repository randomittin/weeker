"""Mastery replay / consistency check (013 §8, T-35).

Mastery must be reconstructable from the immutable attempt log. For each learner
concept we replay the log: final ability is the latest attempt's ``theta_after``,
and stability is recomputed from scratch by re-running ``stability_update`` over
every attempt (each attempt stores ``theta_after``, ``correct`` and ``confidence``
— everything the retention update needs). The recomputed values are diffed
against the stored :class:`Mastery` rows; any drift beyond tolerance is a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from weeker.core.db import get_session
from weeker.core.models import Attempt, Mastery
from weeker.learn import elo, retention


@dataclass
class ReplayDiff:
    """Result of replaying the attempt log against stored mastery."""

    concepts_checked: int = 0
    max_theta_diff: float = 0.0
    max_stability_diff: float = 0.0
    mismatches: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def _recompute(attempts: list[Attempt]) -> tuple[float, float]:
    """Return ``(final_theta, final_stability)`` from an ordered attempt list."""
    stability = 1.0
    theta_after = 0.0
    for a in attempts:
        theta_after = float(a.theta_after)
        stability = retention.stability_update(
            stability, elo.mastery(theta_after), bool(a.correct), a.confidence
        )
    return theta_after, stability


def replay(
    db: Session | None = None,
    user_id: str | None = None,
    *,
    tol: float = 1e-4,
) -> ReplayDiff:
    """Replay attempts and diff recomputed mastery against the stored rows."""
    own = db is None
    db = db or get_session()
    try:
        q = select(Attempt)
        if user_id is not None:
            q = q.where(Attempt.user_id == user_id)
        attempts = db.scalars(q.order_by(Attempt.created_at, Attempt.id)).all()

        grouped: dict = {}
        for a in attempts:
            grouped.setdefault((a.user_id, a.concept_id), []).append(a)

        mastery_rows = {
            (m.user_id, m.concept_id): m
            for m in db.scalars(select(Mastery)).all()
        }

        diff = ReplayDiff()
        for key, rows in grouped.items():
            diff.concepts_checked += 1
            theta, stability = _recompute(rows)
            m = mastery_rows.get(key)
            if m is None:
                diff.mismatches.append({"key": _fmt(key), "reason": "no mastery row"})
                continue
            dt = abs(theta - float(m.theta))
            ds = abs(stability - float(m.stability))
            diff.max_theta_diff = max(diff.max_theta_diff, dt)
            diff.max_stability_diff = max(diff.max_stability_diff, ds)
            if dt > tol or ds > tol:
                diff.mismatches.append(
                    {
                        "key": _fmt(key),
                        "theta_diff": dt,
                        "stability_diff": ds,
                        "stored_theta": float(m.theta),
                        "replayed_theta": theta,
                    }
                )
        return diff
    finally:
        if own:
            db.close()


def _fmt(key) -> str:
    user_id, concept_id = key
    return f"{user_id}:{concept_id}"
