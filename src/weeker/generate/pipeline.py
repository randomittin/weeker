"""T-24 — the generation pipeline: overgenerate, gate, persist (012 §3).

For each concept's authoring quota the pipeline overgenerates by
:data:`~weeker.core.config.GEN_OVERGEN_FACTOR`, runs every candidate through the
full gate chain (G1 → G4 → G5 → G2 → G3), and persists the survivors:

* all hard gates pass and the blind solver agrees → ``status='active'``;
* the blind solver disputes the key → ``status='disputed'`` (kept for the review
  TUI, not counted toward the active quota);
* any hard gate fails → the candidate is dropped (counted in the report only).

Embeddings (option distinctness, stem↔chunk cosine, stem dedup) come from the
shared :func:`weeker.core.embed.embed` client with an injectable transport; the
LLM calls (generation, G2, G3) go through :func:`weeker.core.llm.complete`, also
injectable, so the whole pipeline runs against fakes in tests.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from weeker.core import config
from weeker.core.embed import EmbedTransport, embed
from weeker.core.llm import Transport
from weeker.core.models import Chunk, Concept, Course, Page, Question, Source
from weeker.generate import gates
from weeker.generate.calibrate import initial_q_rating
from weeker.generate.prompts import GeneratedQuestion, generate_for_concept, top_chunks
from weeker.generate.targets import compute_targets

# Statuses the pipeline writes.
STATUS_ACTIVE = "active"
STATUS_DISPUTED = "disputed"


@dataclass
class GenerationReport:
    """What one generation run produced, with per-gate rejection tallies."""

    generated: int = 0
    accepted: int = 0
    disputed: int = 0
    rejected: int = 0
    by_gate: dict[str, int] = field(default_factory=dict)

    def _reject(self, gate: str) -> None:
        self.rejected += 1
        self.by_gate[gate] = self.by_gate.get(gate, 0) + 1


def _embedder(transport: EmbedTransport | None, cache_dir: str | Path | None):
    def fn(texts: list[str]) -> np.ndarray:
        return embed(texts, transport=transport, cache_dir=cache_dir)

    return fn


def _ocr_pages(session: Session, course_id: uuid.UUID) -> dict[uuid.UUID, set[int]]:
    """Map source_id → set of OCR'd page numbers for the course."""
    out: dict[uuid.UUID, set[int]] = {}
    rows = session.execute(
        select(Page.source_id, Page.page_number)
        .join(Source, Page.source_id == Source.id)
        .where(Source.course_id == course_id, Page.ocr_used.is_(True))
    ).all()
    for src_id, pno in rows:
        out.setdefault(src_id, set()).add(pno)
    return out


def _chunk_is_ocr(chunk: Chunk, ocr: dict[uuid.UUID, set[int]]) -> bool:
    """True when every page a chunk spans was produced by OCR (exclude from n-grams)."""
    pages = ocr.get(chunk.source_id)
    if not pages or chunk.page_start is None or chunk.page_end is None:
        return False
    span = range(chunk.page_start, chunk.page_end + 1)
    return bool(span) and all(p in pages for p in span)


def course_ngram_index(session: Session, course_id: uuid.UUID) -> set[str]:
    """Normalized G4 n-gram set over the course's non-OCR chunk text (built once)."""
    ocr = _ocr_pages(session, course_id)
    texts = [
        c.content
        for c in session.scalars(select(Chunk).where(Chunk.course_id == course_id))
        if not _chunk_is_ocr(c, ocr)
    ]
    return gates.build_ngram_index(texts)


def _existing_stems(session: Session, course_id: uuid.UUID) -> list[tuple[str, str]]:
    """(question_id, stem) for every active/disputed question in the course."""
    return [
        (str(qid), stem)
        for qid, stem in session.execute(
            select(Question.id, Question.stem).where(
                Question.course_id == course_id,
                Question.status.in_((STATUS_ACTIVE, STATUS_DISPUTED)),
            )
        ).all()
    ]


def gate_candidate(
    gq: GeneratedQuestion,
    *,
    chunk_texts: list[str],
    chunk_matrix: np.ndarray,
    ngram_index: set[str],
    existing: list[tuple[object, np.ndarray]],
    embed_fn,
    llm_transport: Transport | None,
    solver_transport: Transport | None,
    verifier_model: str | None,
    solver_model: str | None,
    consensus: int,
) -> tuple[dict, str]:
    """Run one candidate through the full gate chain.

    Returns ``(gate_log, status)`` where status is ``active``, ``disputed`` or
    ``rejected``. Hard gates (G1/G4/G5-dup/G2) reject; a G3 dispute yields
    ``disputed``; a clean pass yields ``active``.
    """
    gate_log: dict = {}
    options = gq.option_dicts()

    if not gates.check_g1(
        stem=gq.stem, options=options, correct_key=gq.correct_key, embed_fn=embed_fn, gate_log=gate_log
    ).passed:
        return gate_log, "rejected"

    stem_vec = np.asarray(embed_fn([gq.stem])[0], dtype=np.float32)

    if not gates.check_g4(
        gq.stem, ngram_index, stem_vec=stem_vec, chunk_matrix=chunk_matrix, gate_log=gate_log
    ).passed:
        return gate_log, "rejected"

    if not gates.check_g5(stem_vec, existing, gate_log=gate_log).passed:
        return gate_log, "rejected"

    if not gates.check_g2(
        stem=gq.stem,
        options=options,
        correct_key=gq.correct_key,
        chunk_texts=chunk_texts,
        model=verifier_model,
        transport=llm_transport,
        gate_log=gate_log,
    ).passed:
        return gate_log, "rejected"

    g3 = gates.check_g3(
        stem=gq.stem,
        options=options,
        correct_key=gq.correct_key,
        consensus=consensus,
        model=solver_model,
        transport=solver_transport or llm_transport,
        gate_log=gate_log,
    )
    return gate_log, (STATUS_ACTIVE if g3.passed else STATUS_DISPUTED)


def _persist(
    session: Session,
    course: Course,
    concept: Concept,
    gq: GeneratedQuestion,
    gate_log: dict,
    status: str,
    generator_model: str | None,
) -> Question:
    q = Question(
        course_id=course.id,
        concept_id=concept.id,
        stem=gq.stem,
        options=gq.option_dicts(),
        correct_key=gq.correct_key,
        explanation=gq.explanation,
        difficulty=gq.difficulty,
        q_rating=initial_q_rating(gq.difficulty),
        status=status,
        gate_log=gate_log,
        generator_model=generator_model,
        marks=1,
    )
    session.add(q)
    session.flush()
    return q


def run_generation(
    session: Session,
    course: Course,
    user_id: str,
    *,
    count: int,
    refill: bool = False,
    consensus: int = 1,
    generator_model: str | None = None,
    verifier_model: str | None = None,
    solver_model: str | None = None,
    llm_transport: Transport | None = None,
    solver_transport: Transport | None = None,
    embed_transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
) -> GenerationReport:
    """Generate, gate and persist ``count`` questions across the course's concepts.

    ``solver_transport`` targets the G3 blind solver independently of
    ``llm_transport`` (generator + G2). When omitted it falls back to
    ``llm_transport``, and when both are omitted the blind solver uses its own
    default endpoint (``WEEKER_SOLVER_*`` → :func:`gates.get_solver_transport`).
    """
    report = GenerationReport()
    embed_fn = _embedder(embed_transport, cache_dir)
    ngram_index = course_ngram_index(session, course.id)

    # Course-wide existing stems (extended as we accept, so intra-run dedup holds).
    existing_pairs = _existing_stems(session, course.id)
    existing: list[tuple[object, np.ndarray]] = []
    if existing_pairs:
        vecs = embed_fn([s for _, s in existing_pairs])
        existing = [(qid, np.asarray(vecs[i], dtype=np.float32)) for i, (qid, _) in enumerate(existing_pairs)]

    for target in compute_targets(session, course.id, user_id, count, refill=refill):
        concept = session.get(Concept, target.concept_id)
        if concept is None or target.quota <= 0:
            continue
        chunk_texts = [c.content for c in top_chunks(session, concept.id)]
        chunk_matrix = (
            np.asarray(embed_fn(chunk_texts), dtype=np.float32)
            if chunk_texts
            else np.empty((0, 0), dtype=np.float32)
        )

        want = math.ceil(target.quota * config.GEN_OVERGEN_FACTOR)
        candidates: list[GeneratedQuestion] = []
        while len(candidates) < want:
            batch = generate_for_concept(
                session,
                concept,
                batch=config.GEN_BATCH,
                model=generator_model,
                transport=llm_transport,
            )
            if not batch:
                break
            candidates.extend(batch)

        accepted_here = 0
        for gq in candidates:
            report.generated += 1
            if accepted_here >= target.quota:
                break
            gate_log, status = gate_candidate(
                gq,
                chunk_texts=chunk_texts,
                chunk_matrix=chunk_matrix,
                ngram_index=ngram_index,
                existing=existing,
                embed_fn=embed_fn,
                llm_transport=llm_transport,
                solver_transport=solver_transport,
                verifier_model=verifier_model,
                solver_model=solver_model,
                consensus=consensus,
            )
            if status == "rejected":
                failed = next(
                    (g for g in ("G1", "G4", "G5", "G2") if not gate_log.get(g, {}).get("passed", True)),
                    "G?",
                )
                report._reject(failed)
                continue

            _persist(session, course, concept, gq, gate_log, status, generator_model)
            stem_vec = np.asarray(embed_fn([gq.stem])[0], dtype=np.float32)
            existing.append((gq.stem, stem_vec))
            if status == STATUS_ACTIVE:
                report.accepted += 1
                accepted_here += 1
            else:
                report.disputed += 1

    session.flush()
    return report
