"""Generation gate chain (012 §3).

Gates are the wall between a *generated* question and an *active* one. This module
holds the pure-code gates (G1/G4/G5); the LLM gates (G2/G3) live alongside them in
:mod:`atlas.generate.gates` too (added by T-23). Every gate returns a
:class:`GateResult` and records ``{"passed", "reason", ...}`` into the question's
``gate_log`` under its own key, so a rejection is always auditable.

* **G1 — structural invariants**: exactly 4 options keyed A–D, exactly one correct,
  the "all/none of the above" ban, and option distinctness (no two options within
  :data:`~atlas.core.config.OPTION_DISTINCT_MAX_COS`).
* **G4 — anti-plagiarism**: no verbatim :data:`~atlas.core.config.G4_NGRAM`-gram
  lifted from the (OCR-excluded) course corpus, and no stem within
  :data:`~atlas.core.config.G4_STEM_CHUNK_MAX_COS` cosine of a source chunk.
* **G5 — dedup / variant**: reject a stem within
  :data:`~atlas.core.config.G5_DUP_COS` of an existing stem; tag one within
  :data:`~atlas.core.config.G5_VARIANT_COS` as a variant so the scheduler can
  avoid serving near-twins together.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from atlas.core import config
from atlas.ingest.textutil import cosine_matrix

_WORD = re.compile(r"[a-z0-9]+")
_BANNED_OPTION_PHRASES = ("all of the above", "none of the above")
OPTION_KEYS = ("A", "B", "C", "D")


@dataclass
class GateResult:
    """Outcome of one gate: pass/fail, a human reason, and any tags to persist."""

    gate: str
    passed: bool
    reason: str
    tags: dict = field(default_factory=dict)

    def record(self, gate_log: dict) -> GateResult:
        """Write this result under ``gate_log[self.gate]`` and return self."""
        gate_log[self.gate] = {"passed": self.passed, "reason": self.reason, **self.tags}
        return self


def _normalize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ── G1 ──────────────────────────────────────────────────────────────────────────
def check_g1(
    *,
    stem: str,
    options: Sequence[dict],
    correct_key: str,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    gate_log: dict,
) -> GateResult:
    """Structural invariants + option distinctness. Records into ``gate_log['G1']``."""
    if len(options) != 4:
        return GateResult("G1", False, f"must have exactly 4 options, got {len(options)}").record(
            gate_log
        )

    keys = [str(o.get("key", "")) for o in options]
    if sorted(keys) != list(OPTION_KEYS):
        return GateResult(
            "G1", False, f"option keys must be exactly A,B,C,D distinct, got {keys}"
        ).record(gate_log)

    # Exactly one correct: an explicit per-option 'correct' flag (if present) must be
    # singular and agree with correct_key; otherwise correct_key alone names the answer.
    flagged = [str(o.get("key")) for o in options if o.get("correct")]
    if flagged:
        if len(flagged) != 1:
            return GateResult(
                "G1", False, f"exactly one correct option required, {len(flagged)} flagged"
            ).record(gate_log)
        if flagged[0] != correct_key:
            return GateResult(
                "G1", False, f"flagged-correct {flagged[0]} disagrees with correct_key {correct_key}"
            ).record(gate_log)
    if correct_key not in keys:
        return GateResult(
            "G1", False, f"correct_key {correct_key} not among option keys"
        ).record(gate_log)

    for o in options:
        norm = " ".join(_normalize_words(str(o.get("text", ""))))
        for banned in _BANNED_OPTION_PHRASES:
            if banned in norm:
                return GateResult(
                    "G1", False, f"banned option phrase '{banned}' in option {o.get('key')}"
                ).record(gate_log)

    if not str(stem).strip():
        return GateResult("G1", False, "stem must be non-empty").record(gate_log)

    if embed_fn is not None:
        texts = [str(o.get("text", "")) for o in options]
        matrix = np.asarray(embed_fn(texts), dtype=np.float32)
        for i in range(len(texts)):
            sims = cosine_matrix(matrix[i], matrix)
            for j in range(len(texts)):
                if i != j and float(sims[j]) >= config.OPTION_DISTINCT_MAX_COS:
                    return GateResult(
                        "G1",
                        False,
                        f"options {keys[i]}/{keys[j]} not distinct "
                        f"(cos={float(sims[j]):.3f} ≥ {config.OPTION_DISTINCT_MAX_COS})",
                    ).record(gate_log)

    return GateResult("G1", True, "ok").record(gate_log)


# ── G4 ──────────────────────────────────────────────────────────────────────────
def _ngrams(words: list[str], n: int) -> set[str]:
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_ngram_index(texts: Sequence[str], n: int = config.G4_NGRAM) -> set[str]:
    """Normalized ``n``-gram set over a corpus (feed it OCR-excluded chunk text)."""
    index: set[str] = set()
    for t in texts:
        index |= _ngrams(_normalize_words(t), n)
    return index


def check_g4(
    stem: str,
    ngram_index: set[str],
    *,
    stem_vec: np.ndarray | None = None,
    chunk_matrix: np.ndarray | None = None,
    n: int = config.G4_NGRAM,
    gate_log: dict,
) -> GateResult:
    """Anti-plagiarism: verbatim n-gram lift + stem↔chunk cosine ceiling."""
    words = _normalize_words(stem)
    hits = _ngrams(words, n) & ngram_index
    if hits:
        return GateResult(
            "G4", False, f"verbatim {n}-gram lift from source: '{sorted(hits)[0]}'"
        ).record(gate_log)

    if stem_vec is not None and chunk_matrix is not None and np.asarray(chunk_matrix).size:
        sims = cosine_matrix(np.asarray(stem_vec, dtype=np.float32), np.asarray(chunk_matrix))
        top = float(sims.max()) if sims.size else 0.0
        if top >= config.G4_STEM_CHUNK_MAX_COS:
            return GateResult(
                "G4",
                False,
                f"stem too close to source chunk (cos={top:.3f} ≥ {config.G4_STEM_CHUNK_MAX_COS})",
            ).record(gate_log)

    return GateResult("G4", True, "ok").record(gate_log)


# ── G5 ──────────────────────────────────────────────────────────────────────────
def check_g5(
    stem_vec: np.ndarray,
    existing: Sequence[tuple[object, np.ndarray]],
    *,
    gate_log: dict,
) -> GateResult:
    """Dedup + variant tagging against existing stem vectors.

    ``existing`` is ``(id, vector)`` pairs. A max cosine ≥ ``G5_DUP_COS`` rejects
    the stem as a near-duplicate; a max cosine in ``[G5_VARIANT_COS, G5_DUP_COS)``
    passes but tags the variant group (the matched stem's id) so the scheduler
    keeps variants apart.
    """
    if not existing:
        return GateResult("G5", True, "no existing stems to compare").record(gate_log)

    matrix = np.vstack([np.asarray(v, dtype=np.float32) for _, v in existing])
    sims = cosine_matrix(np.asarray(stem_vec, dtype=np.float32), matrix)
    top_i = int(np.argmax(sims))
    top = float(sims[top_i])
    match_id = existing[top_i][0]

    if top >= config.G5_DUP_COS:
        return GateResult(
            "G5",
            False,
            f"near-duplicate of {match_id} (cos={top:.3f} ≥ {config.G5_DUP_COS})",
            tags={"duplicate_of": str(match_id)},
        ).record(gate_log)

    if top >= config.G5_VARIANT_COS:
        return GateResult(
            "G5",
            True,
            f"variant of {match_id} (cos={top:.3f} ≥ {config.G5_VARIANT_COS})",
            tags={"variant_of": str(match_id), "variant_group": str(match_id)},
        ).record(gate_log)

    return GateResult("G5", True, f"distinct (max cos={top:.3f})").record(gate_log)
