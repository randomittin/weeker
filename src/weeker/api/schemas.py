"""Pydantic request/response schemas for the Weeker web API.

Response models are the leak guard: a question shown *before* it is answered is
serialised through :class:`DiagnosticQuestion` / :class:`StudyQuestion` /
:class:`MockQuestion`, none of which carry ``correct_key``. Only the post-answer
review shapes (:class:`StudyAnswerOut`, :class:`ReviewItem`) expose the key.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel

# ── shared ─────────────────────────────────────────────────────────────────────
Confidence = str  # "sure" | "unsure" | "guessing" (validated by the engine)


class Option(BaseModel):
    """A public answer option — key + text only, never a misconception label."""

    key: str
    text: str


# ── health ─────────────────────────────────────────────────────────────────────
class BankCounts(BaseModel):
    concepts: int
    questions: int
    caselets: int


class HealthOut(BaseModel):
    ok: bool
    bank: BankCounts


# ── status dashboard ───────────────────────────────────────────────────────────
class HeatCell(BaseModel):
    """One heatmap entry. A chapter aggregate row has ``concept=None``."""

    chapter: str
    concept: str | None = None
    mastery: float
    due: float | None = None
    flags: int = 0


class PredictionOut(BaseModel):
    expected: float
    band: list[float]
    p_pass: float
    coverage: float
    suppressed: bool


class DueCounts(BaseModel):
    concepts: int
    flashcards: int


class RefillItem(BaseModel):
    concept_id: str
    title: str
    quota: int


class CalibrationRow(BaseModel):
    confidence: str
    n: int
    correct: int
    rate: float
    recommendation: str


class StatusOut(BaseModel):
    heatmap: list[HeatCell]
    prediction: PredictionOut
    due_counts: DueCounts
    refill_plan: list[RefillItem]
    calibration: list[CalibrationRow] | None = None


# ── diagnostic ─────────────────────────────────────────────────────────────────
class DiagnosticQuestion(BaseModel):
    id: str
    stem: str
    options: list[Option]
    chapter: str | None = None


class DiagnosticStartOut(BaseModel):
    session_id: str
    questions: list[DiagnosticQuestion]


class AnswerIn(BaseModel):
    question_id: uuid.UUID
    chosen_key: str | None = None
    confidence: Confidence = "guessing"


class DiagnosticSubmitIn(BaseModel):
    answers: list[AnswerIn]


class DiagnosticSubmitOut(BaseModel):
    seeded: bool
    status: StatusOut


# ── study ──────────────────────────────────────────────────────────────────────
class StudyQuestion(BaseModel):
    id: str
    stem: str
    options: list[Option]
    chapter: str | None = None
    concept: str | None = None


class StudyNextOut(BaseModel):
    question: StudyQuestion | None = None


class StudyAnswerIn(BaseModel):
    question_id: uuid.UUID
    chosen_key: str
    confidence: Confidence = "guessing"


class MasteryDelta(BaseModel):
    theta_before: float
    theta_after: float
    mastery_before: float
    mastery_after: float


class SourcePeek(BaseModel):
    pages: str
    text: str


class StudyAnswerOut(BaseModel):
    correct: bool
    correct_key: str
    explanation: str
    delta: MasteryDelta
    misconception_fired: bool
    source_peek: SourcePeek | None = None


# ── flashcards ─────────────────────────────────────────────────────────────────
class FlashCard(BaseModel):
    id: str
    front: str
    back: str
    concept_id: str
    stability: float
    due_at: str | None = None


class FlashNextOut(BaseModel):
    card: FlashCard | None = None


class FlashGradeIn(BaseModel):
    card_id: uuid.UUID
    grade: int  # 1 again · 2 hard · 3 good


class Rescheduled(BaseModel):
    due_at: str | None = None
    stability: float


class FlashGradeOut(BaseModel):
    rescheduled: Rescheduled


# ── mock exam ──────────────────────────────────────────────────────────────────
class MockStartIn(BaseModel):
    kind: str = "standalone"  # "standalone" | "full"


class MockQuestion(BaseModel):
    id: str
    stem: str
    options: list[Option]
    chapter: str | None = None
    marks: float
    case_group_id: str | None = None
    case_scenario: str | None = None


class MockStartOut(BaseModel):
    mock_id: str
    kind: str
    questions: list[MockQuestion]
    total_marks: float
    duration_minutes: float


class MockSubmitIn(BaseModel):
    mock_id: uuid.UUID
    answers: list[AnswerIn]


class SectionScore(BaseModel):
    kind: str
    score: float
    max: float


class ReviewItem(BaseModel):
    question_id: str
    stem: str
    options: list[Option]
    correct_key: str
    chosen_key: str | None = None
    explanation: str
    marks: float


class MockSubmitOut(BaseModel):
    raw_score: float
    per_section: list[SectionScore]
    prediction: PredictionOut
    review: list[ReviewItem]


# ── concept source ─────────────────────────────────────────────────────────────
class SourceChunk(BaseModel):
    pages: str
    text: str


class ConceptSourceOut(BaseModel):
    chunks: list[SourceChunk]
