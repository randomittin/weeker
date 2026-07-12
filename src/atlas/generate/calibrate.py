"""T-25 — question calibration: rating init, live Elo hook, retirement.

A question's ``q_rating`` starts from its authored difficulty and thereafter moves
on the shared Elo scale with every attempt (the attempt write path applies the
update; :func:`live_update` here reads the same :class:`~atlas.learn.elo.UpdateResult`).
Over time the bank is audited: a question whose rating has drifted far from its
authored difficulty is flagged for ``recalibrate``; a distractor no one ever picks
is a dead distractor; and a question that no longer separates strong from weak
learners (discrimination below the floor at enough attempts) is retired.

Only :func:`initial_q_rating` is consumed by the generation pipeline (T-24); the
audit functions are driven by ``atlas`` maintenance commands.
"""

from __future__ import annotations

# Authored difficulty (1 easy … 3 hard) → starting q_rating on the logit scale.
# Centered on 0 (an average question) so an average learner (theta≈0) sees ~50%.
DIFFICULTY_RATING = {1: -0.6, 2: 0.0, 3: 0.6}


def initial_q_rating(difficulty: int) -> float:
    """Starting question rating for an authored difficulty tier (1/2/3)."""
    return DIFFICULTY_RATING.get(int(difficulty), 0.0)
