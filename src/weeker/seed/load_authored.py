"""Load hand-authored concept/question/caselet JSON into the DB (keyless).

The authoring contract (one file per chapter) is::

    {"chapter_ordinal": int, "chapter_title": str,
     "concepts": [{"title", "description", "difficulty", "prerequisites",
                   "keywords", "misconceptions", "source_pages",
                   "objectives": [{"text", "bloom", "assessment_style"}],
                   "questions": [<question>]}],
     "caselets": [{"scenario", "concept_titles", "questions": [<question + case_position + marks>]}]}

Each ``<question>`` is validated against the existing
:class:`~weeker.generate.prompts.GeneratedQuestion` contract (caselet questions
against :class:`~weeker.generate.caselets.CaseletQuestion`), then run through the
PURE-CODE gates only:

* **G1** structural invariants + option distinctness,
* **G4** anti-plagiarism (8-gram lift + stem↔chunk cosine) vs the real course chunks,
* **G5** dedup / variant vs the existing bank.

G2/G3 (the LLM grounding/blind-solve gates) are *not* run — authored content is
pre-grounded — and are stamped ``{"skipped": True}`` in the ``gate_log`` so a
reader can tell an authored item from a generated one. A question that fails a
pure-code gate is skipped and tallied under the failing gate; the rest are
persisted ``status='active'`` with ``generator_model='claude-authored'``.

The loader is idempotent (re-running adds nothing: concepts dedup by
title-in-chapter, objectives by text, questions by stem, caselets by scenario)
and transactional per file (a bad file rolls back only its own rows).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from weeker.core.db import get_session
from weeker.core.embed import EmbedTransport
from weeker.core.models import CaseGroup, Chapter, Concept, Course, Objective, Question
from weeker.generate import gates
from weeker.generate.calibrate import initial_q_rating
from weeker.generate.caselets import CaseletQuestion, GeneratedCaselet
from weeker.generate.pipeline import (
    STATUS_ACTIVE,
    _embedder,
    _existing_stems,
    course_ngram_index,
)
from weeker.generate.prompts import GeneratedQuestion, top_chunks
from weeker.ingest.concepts import map_concept_chunks

GENERATOR_MODEL = "claude-authored"


@dataclass
class ChapterLoad:
    """What one authored chapter file contributed to the DB."""

    ordinal: int
    title: str
    concepts_inserted: int = 0
    concepts_skipped: int = 0
    objectives_inserted: int = 0
    questions_inserted: int = 0
    questions_skipped: int = 0
    caselets_inserted: int = 0
    caselets_skipped: int = 0
    caselet_questions_inserted: int = 0
    rejected_by_gate: dict[str, int] = field(default_factory=dict)

    def reject(self, gate: str) -> None:
        self.rejected_by_gate[gate] = self.rejected_by_gate.get(gate, 0) + 1

    @property
    def questions_rejected(self) -> int:
        return sum(self.rejected_by_gate.values())


@dataclass
class LoadReport:
    """Aggregate outcome of a :func:`load_authored` run."""

    chapters: list[ChapterLoad] = field(default_factory=list)
    file_errors: dict[str, str] = field(default_factory=dict)

    @property
    def concepts_inserted(self) -> int:
        return sum(c.concepts_inserted for c in self.chapters)

    @property
    def objectives_inserted(self) -> int:
        return sum(c.objectives_inserted for c in self.chapters)

    @property
    def questions_inserted(self) -> int:
        return sum(c.questions_inserted for c in self.chapters)

    @property
    def questions_rejected(self) -> int:
        return sum(c.questions_rejected for c in self.chapters)

    @property
    def caselets_inserted(self) -> int:
        return sum(c.caselets_inserted for c in self.chapters)

    @property
    def rejected_by_gate(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for c in self.chapters:
            for gate, n in c.rejected_by_gate.items():
                totals[gate] = totals.get(gate, 0) + n
        return totals


# ── helpers ──────────────────────────────────────────────────────────────────
def _resolve_course(session: Session, course: str | None) -> Course | None:
    stmt = select(Course)
    if course:
        stmt = stmt.where((Course.slug == course) | (Course.title == course))
    return session.scalars(stmt.order_by(Course.created_at.desc())).first()


def _resolve_files(dir_or_files: str | Path | list) -> list[Path]:
    """Normalize a dir / file / list-of-files into a sorted list of JSON paths."""
    if isinstance(dir_or_files, (list, tuple)):
        return [Path(p) for p in dir_or_files]
    p = Path(dir_or_files)
    if p.is_dir():
        return sorted(p.glob("*.json"))
    return [p]


def _norm_stem(stem: str) -> str:
    return " ".join(str(stem).split()).lower()


def _coerce_pages(raw) -> list[int]:
    """Keep integer-parseable page entries, drop the rest (tolerant).

    Some authors put section labels (``"11.1.1"``, ``"12.1"``) in ``source_pages``
    instead of int pages. Those are not pages and must not crash the load — they
    are silently dropped; genuine int pages (``13``, ``"14"``) are kept.
    """
    pages: list[int] = []
    for x in raw or []:
        try:
            pages.append(int(x))
        except (TypeError, ValueError):
            continue
    return pages


def _skip_llm_gates(gate_log: dict) -> None:
    """Stamp the un-run LLM gates so authored items stay auditable."""
    reason = "skipped: authored content is pre-grounded"
    gate_log["G2"] = {"passed": True, "reason": reason, "skipped": True}
    gate_log["G3"] = {"passed": True, "reason": reason, "skipped": True, "disputed": False}


def _gate_authored(
    stem: str,
    options: list[dict],
    correct_key: str,
    *,
    ngram_index: set[str],
    chunk_matrix: np.ndarray,
    existing: list[tuple[object, np.ndarray]],
    embed_fn,
) -> tuple[dict, str | None]:
    """Run the pure-code gate chain on one authored question.

    Returns ``(gate_log, failed_gate)`` — ``failed_gate`` is ``None`` on a clean
    pass, otherwise the name of the gate that rejected the question.
    """
    gate_log: dict = {}
    if not gates.check_g1(
        stem=stem, options=options, correct_key=correct_key, embed_fn=embed_fn, gate_log=gate_log
    ).passed:
        return gate_log, "G1"
    stem_vec = np.asarray(embed_fn([stem])[0], dtype=np.float32)
    if not gates.check_g4(
        stem, ngram_index, stem_vec=stem_vec, chunk_matrix=chunk_matrix, gate_log=gate_log
    ).passed:
        return gate_log, "G4"
    if not gates.check_g5(stem_vec, existing, gate_log=gate_log).passed:
        return gate_log, "G5"
    _skip_llm_gates(gate_log)
    return gate_log, None


def _chunk_matrix_for(session: Session, concept: Concept, embed_fn) -> np.ndarray:
    """Embed the concept's grounding chunks in the stem's embedding space (for G4)."""
    texts = [c.content for c in top_chunks(session, concept.id)]
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(embed_fn(texts), dtype=np.float32)


# ── per-question / per-concept / per-caselet loading ───────────────────────────
def _load_question(
    q_raw: dict,
    concept: Concept,
    course: Course,
    session: Session,
    *,
    ngram_index: set[str],
    chunk_matrix: np.ndarray,
    existing: list[tuple[object, np.ndarray]],
    embed_fn,
    stem_set: set[str],
    chapter_report: ChapterLoad,
) -> None:
    """Validate + gate + persist one standalone authored question."""
    key = _norm_stem(q_raw.get("stem", ""))
    if key in stem_set:  # exact-stem idempotency
        chapter_report.questions_skipped += 1
        return

    try:
        gq = GeneratedQuestion.model_validate(q_raw)
    except ValidationError:
        chapter_report.reject("contract")
        return

    options = gq.option_dicts()
    gate_log, failed = _gate_authored(
        gq.stem, options, gq.correct_key,
        ngram_index=ngram_index, chunk_matrix=chunk_matrix, existing=existing, embed_fn=embed_fn,
    )
    if failed is not None:
        chapter_report.reject(failed)
        return

    session.add(
        Question(
            course_id=course.id,
            concept_id=concept.id,
            stem=gq.stem,
            options=options,
            correct_key=gq.correct_key,
            explanation=gq.explanation,
            difficulty=gq.difficulty,
            q_rating=initial_q_rating(gq.difficulty),
            status=STATUS_ACTIVE,
            gate_log=gate_log,
            generator_model=GENERATOR_MODEL,
            marks=1.0,
        )
    )
    session.flush()
    stem_set.add(key)
    existing.append((gq.stem, np.asarray(embed_fn([gq.stem])[0], dtype=np.float32)))
    chapter_report.questions_inserted += 1


def _load_concept(
    c_raw: dict,
    course: Course,
    chapter: Chapter | None,
    session: Session,
    *,
    review_status: str,
    ngram_index: set[str],
    existing: list[tuple[object, np.ndarray]],
    embed_fn,
    stem_set: set[str],
    concept_index: dict[str, Concept],
    chapter_report: ChapterLoad,
) -> Concept:
    """Insert (or reuse) one authored concept, its objectives and its questions."""
    title = str(c_raw["title"]).strip()
    idx_key = title.lower()
    concept = concept_index.get(idx_key)
    if concept is None:
        concept = Concept(
            course_id=course.id,
            chapter_id=chapter.id if chapter else None,
            title=title,
            description=str(c_raw.get("description", "")).strip(),
            difficulty=int(c_raw.get("difficulty", 2)),
            prerequisites=[str(x) for x in c_raw.get("prerequisites", []) or []],
            keywords=[str(x) for x in c_raw.get("keywords", []) or []],
            misconceptions=[str(x) for x in c_raw.get("misconceptions", []) or []],
            source_pages=_coerce_pages(c_raw.get("source_pages")),
            review_status=review_status,
            reviewed_at=datetime.now(UTC),
        )
        concept.embedding = np.asarray(
            embed_fn([f"{title}. {concept.description}"])[0], dtype=np.float32
        )
        session.add(concept)
        session.flush()
        map_concept_chunks(session, course, concept)
        concept_index[idx_key] = concept
        chapter_report.concepts_inserted += 1
    else:
        chapter_report.concepts_skipped += 1

    # Objectives — dedup by text within the concept.
    existing_obj = {
        t.strip().lower()
        for t in session.scalars(
            select(Objective.text).where(Objective.concept_id == concept.id)
        )
    }
    for obj in c_raw.get("objectives", []) or []:
        otext = str(obj["text"]).strip()
        if otext.lower() in existing_obj:
            continue
        session.add(
            Objective(
                concept_id=concept.id,
                text=otext,
                bloom=str(obj["bloom"]),
                assessment_style=str(obj["assessment_style"]),
            )
        )
        existing_obj.add(otext.lower())
        chapter_report.objectives_inserted += 1
    session.flush()

    # Questions — validate, gate, persist.
    chunk_matrix = _chunk_matrix_for(session, concept, embed_fn)
    for q_raw in c_raw.get("questions", []) or []:
        _load_question(
            q_raw, concept, course, session,
            ngram_index=ngram_index, chunk_matrix=chunk_matrix, existing=existing,
            embed_fn=embed_fn, stem_set=stem_set, chapter_report=chapter_report,
        )
    return concept


def _load_caselet(
    cl_raw: dict,
    course: Course,
    session: Session,
    *,
    ngram_index: set[str],
    existing: list[tuple[object, np.ndarray]],
    embed_fn,
    stem_set: set[str],
    scenario_set: set[str],
    concept_index: dict[str, Concept],
    chapter_report: ChapterLoad,
) -> None:
    """Validate + gate + persist one authored caselet (scenario + five questions)."""
    scenario = str(cl_raw.get("scenario", "")).strip()
    skey = _norm_stem(scenario)
    if skey in scenario_set:  # scenario idempotency
        chapter_report.caselets_skipped += 1
        return

    try:
        caselet = GeneratedCaselet.model_validate(
            {"scenario": scenario, "questions": cl_raw.get("questions", [])}
        )
    except ValidationError:
        chapter_report.reject("contract")
        return

    # Resolve the caselet's concepts (must already be authored in this course).
    concepts = [
        concept_index[t.strip().lower()]
        for t in cl_raw.get("concept_titles", [])
        if t.strip().lower() in concept_index
    ]
    if not concepts:
        chapter_report.reject("contract")
        return

    # Grounding for G4: union of the concepts' chunks.
    chunk_texts: list[str] = []
    for c in concepts:
        chunk_texts += [ch.content for ch in top_chunks(session, c.id)]
    chunk_matrix = (
        np.asarray(embed_fn(chunk_texts), dtype=np.float32)
        if chunk_texts
        else np.empty((0, 0), dtype=np.float32)
    )

    # G4 on the scenario narrative (no verbatim lift), then per-question G1/G4/G5.
    scenario_log: dict = {}
    scen_vec = np.asarray(embed_fn([scenario])[0], dtype=np.float32)
    if not gates.check_g4(
        scenario, ngram_index, stem_vec=scen_vec, chunk_matrix=chunk_matrix, gate_log=scenario_log
    ).passed:
        chapter_report.reject("G4")
        return
    _skip_llm_gates(scenario_log)

    gated: list[tuple[CaseletQuestion, dict]] = []
    for q in sorted(caselet.questions, key=lambda x: x.case_position):
        if _norm_stem(q.stem) in stem_set:
            chapter_report.reject("G5")  # a repeated caselet stem is a dup
            return
        gate_log, failed = _gate_authored(
            q.stem, q.option_dicts(), q.correct_key,
            ngram_index=ngram_index, chunk_matrix=chunk_matrix, existing=existing, embed_fn=embed_fn,
        )
        if failed is not None:
            chapter_report.reject(failed)
            return
        gated.append((q, gate_log))

    cg = CaseGroup(
        course_id=course.id,
        scenario=scenario,
        concept_ids=[str(c.id) for c in concepts],
        status=STATUS_ACTIVE,
        gate_log=scenario_log,
        generator_model=GENERATOR_MODEL,
    )
    session.add(cg)
    session.flush()

    for i, (q, gate_log) in enumerate(gated):
        concept = concepts[i % len(concepts)]
        session.add(
            Question(
                course_id=course.id,
                concept_id=concept.id,
                case_group_id=cg.id,
                stem=q.stem,
                options=q.option_dicts(),
                correct_key=q.correct_key,
                explanation=q.explanation,
                difficulty=q.difficulty,
                q_rating=initial_q_rating(q.difficulty),
                status=STATUS_ACTIVE,
                gate_log=gate_log,
                generator_model=GENERATOR_MODEL,
                marks=float(q.marks),
                case_position=q.case_position,
            )
        )
        stem_set.add(_norm_stem(q.stem))
        existing.append((q.stem, np.asarray(embed_fn([q.stem])[0], dtype=np.float32)))
        chapter_report.caselet_questions_inserted += 1
    session.flush()
    scenario_set.add(skey)
    chapter_report.caselets_inserted += 1


def _load_file(
    path: Path,
    course: Course,
    session: Session,
    *,
    review_status: str,
    ngram_index: set[str],
    existing: list[tuple[object, np.ndarray]],
    embed_fn,
    stem_set: set[str],
    scenario_set: set[str],
) -> ChapterLoad:
    """Load one authored chapter file into the DB (called inside a savepoint)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ordinal = int(data["chapter_ordinal"])
    report = ChapterLoad(ordinal=ordinal, title=str(data.get("chapter_title", "")))

    chapter = session.scalars(
        select(Chapter).where(Chapter.course_id == course.id, Chapter.ordinal == ordinal)
    ).first()

    # Concepts already in the course (for reuse / FK linking during caselets).
    concept_index: dict[str, Concept] = {
        c.title.strip().lower(): c
        for c in session.scalars(select(Concept).where(Concept.course_id == course.id))
    }

    for c_raw in data.get("concepts", []) or []:
        _load_concept(
            c_raw, course, chapter, session,
            review_status=review_status, ngram_index=ngram_index, existing=existing,
            embed_fn=embed_fn, stem_set=stem_set, concept_index=concept_index,
            chapter_report=report,
        )

    for cl_raw in data.get("caselets", []) or []:
        _load_caselet(
            cl_raw, course, session,
            ngram_index=ngram_index, existing=existing, embed_fn=embed_fn,
            stem_set=stem_set, scenario_set=scenario_set, concept_index=concept_index,
            chapter_report=report,
        )
    return report


def load_authored(
    dir_or_files: str | Path | list,
    *,
    db: Session | None = None,
    review_status: str = "accepted",
    course: str | None = None,
    embed_transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
) -> LoadReport:
    """Ingest hand-authored chapter JSON into the DB, keyless, gate-enforced.

    ``dir_or_files`` is a directory (all ``*.json`` loaded, sorted), a single
    file, or a list of files. Each file is loaded in its own savepoint so a bad
    file rolls back only its own rows. Concepts land ``review_status`` (default
    ``accepted``); questions land ``status='active'`` after passing G1/G4/G5.
    """
    owns = db is None
    session = db or get_session()
    report = LoadReport()
    try:
        c = _resolve_course(session, course)
        if c is None:
            raise ValueError("no course in DB — ingest a course before loading authored content")

        embed_fn = _embedder(embed_transport, cache_dir)

        # Derive the embedder's TRUE output dim from a real probe (never trust a
        # hand-set WEEKER_EMBED_DIM — it can silently mismatch the model). It must
        # match the course's embedding_dim (fixed at ingestion) so authored
        # concept/question vectors share the chunk cosine space. No pad/truncate:
        # a mismatch means the wrong embedding model → hard error.
        true_dim = int(np.asarray(embed_fn(["dimension probe"]), dtype=np.float32).shape[1])
        if c.embedding_dim and true_dim != c.embedding_dim:
            raise ValueError(
                f"embedder emits {true_dim}-dim vectors but course "
                f"{c.slug or c.title!r} was ingested at {c.embedding_dim}-dim; "
                "authored embeddings would not share the chunk cosine space — "
                "load with the same embedding model used at ingestion"
            )

        ngram_index = course_ngram_index(session, c.id)

        # G5 dedup vectors (active/disputed bank) + exact-stem / scenario idempotency sets.
        existing_pairs = _existing_stems(session, c.id)
        existing: list[tuple[object, np.ndarray]] = []
        if existing_pairs:
            vecs = embed_fn([s for _, s in existing_pairs])
            existing = [
                (qid, np.asarray(vecs[i], dtype=np.float32))
                for i, (qid, _) in enumerate(existing_pairs)
            ]
        stem_set = {
            _norm_stem(s)
            for s in session.scalars(select(Question.stem).where(Question.course_id == c.id))
        }
        scenario_set = {
            _norm_stem(s)
            for s in session.scalars(select(CaseGroup.scenario).where(CaseGroup.course_id == c.id))
        }

        for path in _resolve_files(dir_or_files):
            try:
                with session.begin_nested():
                    report.chapters.append(
                        _load_file(
                            path, c, session,
                            review_status=review_status, ngram_index=ngram_index,
                            existing=existing, embed_fn=embed_fn,
                            stem_set=stem_set, scenario_set=scenario_set,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 — record + continue to next file
                report.file_errors[str(path)] = f"{type(exc).__name__}: {exc}"

        if owns:
            session.commit()
    finally:
        if owns:
            session.close()
    return report
