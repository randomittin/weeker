"""FastAPI application wrapping the Weeker study engine (keyless / local).

Every route is a thin adapter: it loads the single local course, calls an engine
service in :mod:`weeker.learn`, and shapes the result into a
:mod:`weeker.api.schemas` model. No learning logic lives here. The web/ build (if
present) is served at ``/`` as static files; otherwise ``/`` returns a short JSON
placeholder so the API is usable before the frontend exists.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from weeker.api import schemas as S
from weeker.core.db import session_scope
from weeker.core.models import (
    CaseGroup,
    Chapter,
    Chunk,
    Concept,
    ConceptChunk,
    Course,
    Flashcard,
    Question,
)
from weeker.learn import diagnose, full, mock, predict, retention, scheduler, status
from weeker.learn.attempt_service import record_attempt

USER_ID = "me"

# Flashcard grade → (correct, confidence), mirroring learn.study_cli.FlashApp.GRADE.
FLASH_GRADE = {1: (False, "sure"), 2: (True, "unsure"), 3: (True, "sure")}

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


# ── request-scoped session ─────────────────────────────────────────────────────
def get_db() -> Iterator[Session]:
    """Yield a transactional session; commits on success, rolls back on error."""
    with session_scope() as db:
        yield db


# Module-level dependency singleton (avoids a Depends() call in argument defaults).
DB = Depends(get_db)


# ── course + label helpers ─────────────────────────────────────────────────────
def _course(db: Session) -> Course:
    course = db.scalars(select(Course).order_by(Course.created_at.desc())).first()
    if course is None:
        raise HTTPException(status_code=404, detail="no course loaded")
    return course


def _chapter_titles(db: Session, course_id: object) -> dict:
    return {
        c.id: c.title
        for c in db.scalars(select(Chapter).where(Chapter.course_id == course_id)).all()
    }


def _concept_meta(db: Session, course_id: object) -> dict:
    """Map concept id → (title, chapter_id) for the course."""
    return {
        c.id: (c.title, c.chapter_id)
        for c in db.scalars(select(Concept).where(Concept.course_id == course_id)).all()
    }


def _options(question: Question) -> list[S.Option]:
    """Public options — key + text only (never a misconception label)."""
    out: list[S.Option] = []
    for opt in question.options or []:
        if isinstance(opt, dict):
            out.append(S.Option(key=str(opt.get("key", "")), text=str(opt.get("text", ""))))
    return out


def _source_peek(db: Session, concept_id: object) -> S.SourcePeek | None:
    """Best-matching source chunk for a concept, as a page-ranged peek."""
    row = (
        db.query(Chunk)
        .join(ConceptChunk, ConceptChunk.chunk_id == Chunk.id)
        .filter(ConceptChunk.concept_id == concept_id)
        .order_by(ConceptChunk.cos.desc())
        .first()
    )
    if row is None:
        return None
    return S.SourcePeek(pages=f"{row.page_start}-{row.page_end}", text=(row.content or "")[:1200])


def _prediction_out(p: predict.Prediction) -> S.PredictionOut:
    return S.PredictionOut(
        expected=float(p.expected_marks),
        band=[float(p.ci_low), float(p.ci_high)],
        p_pass=float(p.p_pass),
        coverage=float(p.coverage),
        suppressed=bool(p.suppressed),
    )


def _status_out(report: status.StatusReport) -> S.StatusOut:
    heatmap: list[S.HeatCell] = []
    for row in report.chapters:
        heatmap.append(
            S.HeatCell(chapter=row.title, concept=None, mastery=float(row.mean_mastery), flags=0)
        )
        for cell in row.cells:
            heatmap.append(
                S.HeatCell(
                    chapter=row.title,
                    concept=cell.title,
                    mastery=float(cell.mastery),
                    due=None if cell.due_in_days is None else float(cell.due_in_days),
                    flags=1 if cell.misconception else 0,
                )
            )
    calibration = (
        [
            S.CalibrationRow(
                confidence=c.confidence,
                n=c.n,
                correct=c.correct,
                rate=float(c.rate),
                recommendation=c.recommendation,
            )
            for c in report.calibration
        ]
        if report.calibration_ready
        else None
    )
    return S.StatusOut(
        heatmap=heatmap,
        prediction=_prediction_out(report.prediction),
        due_counts=S.DueCounts(concepts=report.due_concepts, flashcards=report.due_flashcards),
        refill_plan=[
            S.RefillItem(concept_id=str(e.concept_id), title=e.title, quota=e.quota)
            for e in report.refill_plan
        ],
        calibration=calibration,
    )


# ── app ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Weeker", description="Local study engine web API.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=S.HealthOut)
def health(db: Session = DB) -> S.HealthOut:
    course = _course(db)
    concepts = db.scalar(
        select(func.count())
        .select_from(Concept)
        .where(Concept.course_id == course.id, Concept.review_status != "dropped")
    )
    questions = db.scalar(
        select(func.count())
        .select_from(Question)
        .where(
            Question.course_id == course.id,
            Question.status == "active",
            Question.case_group_id.is_(None),
        )
    )
    caselets = db.scalar(
        select(func.count())
        .select_from(CaseGroup)
        .where(CaseGroup.course_id == course.id, CaseGroup.status == "active")
    )
    return S.HealthOut(
        ok=True,
        bank=S.BankCounts(
            concepts=int(concepts or 0),
            questions=int(questions or 0),
            caselets=int(caselets or 0),
        ),
    )


@app.get("/api/status", response_model=S.StatusOut)
def get_status(db: Session = DB) -> S.StatusOut:
    course = _course(db)
    report = status.build_status(db, user_id=USER_ID, course=course, now=datetime.now(UTC))
    return _status_out(report)


@app.post("/api/diagnostic/start", response_model=S.DiagnosticStartOut)
def diagnostic_start(db: Session = DB) -> S.DiagnosticStartOut:
    course = _course(db)
    questions = diagnose.select_diagnostic_questions(db, course.id, random.Random())
    meta = _concept_meta(db, course.id)
    titles = _chapter_titles(db, course.id)
    out = []
    for q in questions:
        _title, chapter_id = meta.get(q.concept_id, (None, None))
        out.append(
            S.DiagnosticQuestion(
                id=str(q.id),
                stem=q.stem,
                options=_options(q),
                chapter=titles.get(chapter_id),
            )
        )
    return S.DiagnosticStartOut(session_id=str(uuid.uuid4()), questions=out)


@app.post("/api/diagnostic/submit", response_model=S.DiagnosticSubmitOut)
def diagnostic_submit(
    payload: S.DiagnosticSubmitIn, db: Session = DB
) -> S.DiagnosticSubmitOut:
    course = _course(db)
    qmap = {
        q.id: q
        for q in db.scalars(
            select(Question).where(
                Question.id.in_([a.question_id for a in payload.answers])
            )
        ).all()
    }
    graded = [
        diagnose.Graded(question=qmap[a.question_id], chosen_key=a.chosen_key, confidence=a.confidence)
        for a in payload.answers
        if a.question_id in qmap
    ]
    result = diagnose.seed_diagnostic(
        db,
        user_id=USER_ID,
        course=course,
        graded=graded,
        now=datetime.now(UTC),
        force=True,
        build_status=True,
    )
    return S.DiagnosticSubmitOut(seeded=True, status=_status_out(result.status))


@app.get("/api/study/next", response_model=S.StudyNextOut)
def study_next(response: Response, db: Session = DB) -> S.StudyNextOut:
    course = _course(db)
    questions = scheduler.next_session(db, USER_ID, course.id, datetime.now(UTC), random.Random())
    if not questions:
        response.status_code = 204
        return S.StudyNextOut(question=None)
    q = questions[0]
    meta = _concept_meta(db, course.id)
    titles = _chapter_titles(db, course.id)
    concept_title, chapter_id = meta.get(q.concept_id, (None, None))
    return S.StudyNextOut(
        question=S.StudyQuestion(
            id=str(q.id),
            stem=q.stem,
            options=_options(q),
            chapter=titles.get(chapter_id),
            concept=concept_title,
        )
    )


@app.post("/api/study/answer", response_model=S.StudyAnswerOut)
def study_answer(payload: S.StudyAnswerIn, db: Session = DB) -> S.StudyAnswerOut:
    question = db.get(Question, payload.question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="question not found")
    outcome = record_attempt(
        user_id=USER_ID,
        question_id=payload.question_id,
        chosen_key=payload.chosen_key,
        confidence=payload.confidence,
        db=db,
    )
    return S.StudyAnswerOut(
        correct=outcome.correct,
        correct_key=question.correct_key,
        explanation=question.explanation or "",
        delta=S.MasteryDelta(
            theta_before=float(outcome.theta_before),
            theta_after=float(outcome.theta_after),
            mastery_before=float(outcome.mastery_before),
            mastery_after=float(outcome.mastery_after),
        ),
        misconception_fired=outcome.misconception_fired,
        source_peek=_source_peek(db, question.concept_id),
    )


@app.get("/api/flash/next", response_model=S.FlashNextOut)
def flash_next(db: Session = DB) -> S.FlashNextOut:
    card = db.scalars(
        select(Flashcard)
        .where(Flashcard.user_id == USER_ID)
        .order_by(Flashcard.due_at.is_(None).desc(), Flashcard.due_at)
    ).first()
    if card is None:
        return S.FlashNextOut(card=None)
    return S.FlashNextOut(
        card=S.FlashCard(
            id=str(card.id),
            front=card.front,
            back=card.back,
            concept_id=str(card.concept_id),
            stability=float(card.stability),
            due_at=card.due_at.isoformat() if card.due_at is not None else None,
        )
    )


@app.post("/api/flash/grade", response_model=S.FlashGradeOut)
def flash_grade(payload: S.FlashGradeIn, db: Session = DB) -> S.FlashGradeOut:
    card = db.get(Flashcard, payload.card_id)
    if card is None:
        raise HTTPException(status_code=404, detail="flashcard not found")
    if payload.grade not in FLASH_GRADE:
        raise HTTPException(status_code=422, detail="grade must be 1, 2, or 3")
    correct, confidence = FLASH_GRADE[payload.grade]
    proxy_mastery = 1.0 if correct else 0.0
    new_stability = retention.stability_update(
        float(card.stability), proxy_mastery, correct, confidence
    )
    from datetime import timedelta

    due = datetime.now(UTC) + timedelta(days=retention.days_until_due(new_stability))
    card.stability = new_stability
    card.due_at = due
    db.add(card)
    db.flush()
    return S.FlashGradeOut(
        rescheduled=S.Rescheduled(due_at=due.isoformat(), stability=float(new_stability))
    )


@app.post("/api/mock/start", response_model=S.MockStartOut)
def mock_start(payload: S.MockStartIn, db: Session = DB) -> S.MockStartOut:
    course = _course(db)
    now = datetime.now(UTC)
    rng = random.Random()
    params = mock.exam_params(course)
    if payload.kind == "full":
        items = full.compose_full_items(db, USER_ID, course, now, rng)
    elif payload.kind == "standalone":
        items = mock.compose_standalone_items(db, USER_ID, course, now, rng, total=90)
    else:
        raise HTTPException(status_code=422, detail="kind must be 'standalone' or 'full'")
    if not items:
        raise HTTPException(status_code=409, detail="not enough questions to compose a mock")

    predicted = mock.snapshot_prediction(
        db,
        USER_ID,
        course,
        passing_marks=params["passing_marks"],
        total_marks=params["total_marks"],
        neg_fraction=params["neg_fraction"],
        rng=rng,
    )
    exam = mock.start_mock(
        db,
        user_id=USER_ID,
        course=course,
        items=items,
        predicted_before=predicted,
        duration_minutes=params["duration_minutes"],
        kind=payload.kind,
        now=now,
    )

    qmap = {
        q.id: q
        for q in db.scalars(
            select(Question).where(Question.id.in_([it.question_id for it in items]))
        ).all()
    }
    scenarios = {
        cg.id: cg.scenario
        for cg in db.scalars(
            select(CaseGroup).where(
                CaseGroup.id.in_([it.case_group_id for it in items if it.case_group_id])
            )
        ).all()
    }
    meta = _concept_meta(db, course.id)
    titles = _chapter_titles(db, course.id)

    questions: list[S.MockQuestion] = []
    for it in items:
        q = qmap.get(it.question_id)
        if q is None:
            continue
        _t, chapter_id = meta.get(q.concept_id, (None, None))
        questions.append(
            S.MockQuestion(
                id=str(q.id),
                stem=q.stem,
                options=_options(q),
                chapter=titles.get(chapter_id),
                marks=float(it.marks),
                case_group_id=str(it.case_group_id) if it.case_group_id else None,
                case_scenario=scenarios.get(it.case_group_id) if it.case_group_id else None,
            )
        )
    return S.MockStartOut(
        mock_id=str(exam.id),
        kind=payload.kind,
        questions=questions,
        total_marks=float(sum(float(it.marks) for it in items)),
        duration_minutes=float(params["duration_minutes"]),
    )


@app.post("/api/mock/submit", response_model=S.MockSubmitOut)
def mock_submit(payload: S.MockSubmitIn, db: Session = DB) -> S.MockSubmitOut:
    course = _course(db)
    resumed = mock.resume_mock(db, payload.mock_id)
    if resumed.mock.user_id != USER_ID:
        raise HTTPException(status_code=404, detail="mock not found")
    answers = {
        a.question_id: mock.MockAnswer(chosen_key=a.chosen_key, confidence=a.confidence)
        for a in payload.answers
    }
    params = mock.exam_params(course)
    result = mock.submit_mock(
        db,
        mock=resumed.mock,
        items=resumed.items,
        answers=answers,
        passing_marks=params["passing_marks"],
        neg_fraction=params["neg_fraction"],
        now=datetime.now(UTC),
    )

    # Per-section scores (kind = item.section: "standalone" | "caselet").
    section_score: dict = {}
    section_max: dict = {}
    for it in resumed.items:
        section_score.setdefault(it.section, 0.0)
        section_max.setdefault(it.section, 0.0)
        section_score[it.section] += mock.score_item(
            it, answers.get(it.question_id), params["neg_fraction"]
        )
        section_max[it.section] += float(it.marks)
    per_section = [
        S.SectionScore(kind=k, score=round(section_score[k], 4), max=section_max[k])
        for k in section_score
    ]

    prediction = predict.predict_from_db(
        db,
        user_id=USER_ID,
        course_id=course.id,
        passing_marks=params["passing_marks"],
        total_marks=params["total_marks"],
        neg_fraction=params["neg_fraction"],
    )

    qmap = {
        q.id: q
        for q in db.scalars(
            select(Question).where(Question.id.in_([it.question_id for it in result.review]))
        ).all()
    }
    review: list[S.ReviewItem] = []
    for it in result.review:
        q = qmap.get(it.question_id)
        if q is None:
            continue
        ans = answers.get(it.question_id)
        review.append(
            S.ReviewItem(
                question_id=str(q.id),
                stem=q.stem,
                options=_options(q),
                correct_key=q.correct_key,
                chosen_key=ans.chosen_key if ans else None,
                explanation=q.explanation or "",
                marks=float(it.marks),
            )
        )
    return S.MockSubmitOut(
        raw_score=float(result.raw_score),
        per_section=per_section,
        prediction=_prediction_out(prediction),
        review=review,
    )


@app.get("/api/concept/{concept_id}/source", response_model=S.ConceptSourceOut)
def concept_source(concept_id: uuid.UUID, db: Session = DB) -> S.ConceptSourceOut:
    rows = (
        db.query(Chunk)
        .join(ConceptChunk, ConceptChunk.chunk_id == Chunk.id)
        .filter(ConceptChunk.concept_id == concept_id)
        .order_by(ConceptChunk.cos.desc())
        .limit(5)
        .all()
    )
    return S.ConceptSourceOut(
        chunks=[
            S.SourceChunk(pages=f"{r.page_start}-{r.page_end}", text=r.content or "")
            for r in rows
        ]
    )


# ── static frontend (mounted last so /api/* wins) ───────────────────────────────
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:

    @app.get("/")
    def root() -> dict:
        """Placeholder root shown until the web/ frontend is built."""
        return {
            "message": "Weeker API is running. Build the web/ frontend to serve the UI here.",
            "docs": "/docs",
            "health": "/api/health",
        }
