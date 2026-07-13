"""FastAPI web-layer contract tests (TestClient against a temp SQLite bank).

Each test builds an isolated on-disk SQLite database (a unique path → a fresh
cached engine, so no cross-test bleed), seeds a tiny fixture bank, and drives the
real API. The load-bearing guarantees checked here: every *pre-answer* question
payload is free of ``correct_key``; ``/api/study/answer`` reveals the key and
actually writes an :class:`Attempt`; the mock flow composes, scores per section,
and reviews with the key exposed only post-submit.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from weeker.core.models import (
    Attempt,
    Base,
    CaseGroup,
    Chapter,
    Chunk,
    Concept,
    ConceptChunk,
    Course,
    Flashcard,
    Question,
)

USER = "me"


def _seed(db):
    course = Course(title="NISM XA", exam_config={})
    db.add(course)
    db.flush()
    concepts = []
    for ci in range(2):
        ch = Chapter(
            course_id=course.id,
            ordinal=ci + 1,
            title=f"Ch{ci + 1}",
            page_start=ci * 10,
            page_end=ci * 10 + 9,
        )
        db.add(ch)
        db.flush()
        for k in range(3):
            concept = Concept(
                course_id=course.id,
                chapter_id=ch.id,
                title=f"C{ci}-{k}",
                difficulty=(k % 3) + 1,
                review_status="accepted",
            )
            db.add(concept)
            db.flush()
            concepts.append(concept)
            Question_ = Question(
                course_id=course.id,
                concept_id=concept.id,
                stem=f"Stem {concept.title}?",
                options=[
                    {"key": "A", "text": "right", "misconception": "should-not-leak"},
                    {"key": "B", "text": "wrong"},
                    {"key": "C", "text": "wrong"},
                    {"key": "D", "text": "wrong"},
                ],
                correct_key="A",
                explanation="because grounded reasons",
                difficulty=(k % 3) + 1,
                q_rating=0.0,
                marks=1,
                status="active",
            )
            db.add(Question_)
    db.flush()

    # A source chunk mapped to the first concept (for source peek + concept source).
    chunk = Chunk(
        course_id=course.id,
        source_id=uuid.uuid4(),
        content="Regulation 3 requires registration before advising clients.",
        content_hash="h",
        page_start=12,
        page_end=13,
    )
    db.add(chunk)
    db.flush()
    db.add(ConceptChunk(concept_id=concepts[0].id, chunk_id=chunk.id, cos=0.9))

    # One 1-mark and one 2-mark caselet (5 questions each) so `full` has caselets.
    for tier, marks in (("one", 1), ("two", 2)):
        cg = CaseGroup(
            course_id=course.id,
            scenario="An investor case scenario. " * 15,
            concept_ids=[str(concepts[0].id)],
            status="active",
            generator_model="test",
        )
        db.add(cg)
        db.flush()
        for pos in range(1, 6):
            db.add(
                Question(
                    course_id=course.id,
                    concept_id=concepts[pos % len(concepts)].id,
                    case_group_id=cg.id,
                    stem=f"Caselet {tier} q{pos}",
                    options=[
                        {"key": "A", "text": "right"},
                        {"key": "B", "text": "wrong"},
                        {"key": "C", "text": "wrong"},
                        {"key": "D", "text": "wrong"},
                    ],
                    correct_key="A",
                    explanation="because",
                    difficulty=2,
                    q_rating=0.0,
                    marks=marks,
                    case_position=pos,
                    status="active",
                )
            )
    db.flush()

    # A due flashcard for the flash endpoints.
    db.add(
        Flashcard(
            user_id=USER,
            concept_id=concepts[0].id,
            front="front",
            back="back",
            stability=1.0,
        )
    )
    db.flush()
    return course


@pytest.fixture
def client(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'api.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    from weeker.core import db as dbmod

    engine = dbmod.get_engine(url)
    Base.metadata.create_all(engine)
    session = dbmod.get_session(url)
    _seed(session)
    session.commit()
    session.close()

    from weeker.api.app import app

    with TestClient(app) as c:
        c._url = url  # noqa: SLF001 — stash for post-request DB assertions
        yield c


def _fresh_session(client):
    from weeker.core import db as dbmod

    return dbmod.get_session(client._url)


def _assert_no_key(question: dict) -> None:
    assert "correct_key" not in question
    for opt in question["options"]:
        assert set(opt.keys()) == {"key", "text"}  # no misconception label leaks


# ── endpoints ───────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["bank"] == {"concepts": 6, "questions": 6, "caselets": 2}


def test_status_shape(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert {"heatmap", "prediction", "due_counts", "refill_plan", "calibration"} <= set(body)
    assert set(body["prediction"]) == {"expected", "band", "p_pass", "coverage", "suppressed"}
    assert len(body["prediction"]["band"]) == 2
    assert set(body["due_counts"]) == {"concepts", "flashcards"}
    assert body["calibration"] is None  # < 3 attempt-days on a fresh bank


def test_diagnostic_start_hides_key(client):
    r = client.post("/api/diagnostic/start")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["questions"]
    for q in body["questions"]:
        _assert_no_key(q)


def test_diagnostic_submit_seeds(client):
    start = client.post("/api/diagnostic/start").json()
    answers = [
        {"question_id": q["id"], "chosen_key": "A", "confidence": "sure"}
        for q in start["questions"]
    ]
    r = client.post("/api/diagnostic/submit", json={"answers": answers})
    assert r.status_code == 200
    body = r.json()
    assert body["seeded"] is True
    assert "heatmap" in body["status"]
    db = _fresh_session(client)
    assert db.scalar(select(func.count()).select_from(Attempt).where(Attempt.user_id == USER)) > 0
    db.close()


def test_study_next_hides_key(client):
    r = client.get("/api/study/next")
    assert r.status_code in (200, 204)
    if r.status_code == 200 and r.json().get("question"):
        _assert_no_key(r.json()["question"])


def test_study_answer_reveals_key_and_writes_attempt(client):
    q = client.get("/api/study/next").json()["question"]
    r = client.post(
        "/api/study/answer",
        json={"question_id": q["id"], "chosen_key": "A", "confidence": "sure"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["correct"] is True
    assert body["correct_key"] == "A"
    assert body["explanation"]
    assert set(body["delta"]) == {
        "theta_before",
        "theta_after",
        "mastery_before",
        "mastery_after",
    }
    db = _fresh_session(client)
    n = db.scalar(
        select(func.count()).select_from(Attempt).where(Attempt.question_id == uuid.UUID(q["id"]))
    )
    db.close()
    assert n == 1


def test_flash_next_and_grade(client):
    card = client.get("/api/flash/next").json()["card"]
    assert card is not None
    r = client.post("/api/flash/grade", json={"card_id": card["id"], "grade": 3})
    assert r.status_code == 200
    resched = r.json()["rescheduled"]
    assert resched["stability"] > 0
    assert resched["due_at"]


def test_mock_standalone_flow(client):
    start = client.post("/api/mock/start", json={"kind": "standalone"}).json()
    assert start["mock_id"]
    assert start["questions"]
    assert start["total_marks"] > 0
    for q in start["questions"]:
        _assert_no_key(q)
    answers = [
        {"question_id": q["id"], "chosen_key": "A", "confidence": "sure"}
        for q in start["questions"]
    ]
    r = client.post(
        "/api/mock/submit", json={"mock_id": start["mock_id"], "answers": answers}
    )
    assert r.status_code == 200
    body = r.json()
    assert "raw_score" in body
    assert body["per_section"]
    assert set(body["prediction"]) == {"expected", "band", "p_pass", "coverage", "suppressed"}
    # Review is the only place the key surfaces (post-submit).
    assert body["review"]
    assert all("correct_key" in item for item in body["review"])


def test_mock_full_includes_caselets(client):
    start = client.post("/api/mock/start", json={"kind": "full"}).json()
    assert start["questions"]
    caselet_qs = [q for q in start["questions"] if q.get("case_group_id")]
    # The fixture has one 1-mark + one 2-mark caselet (5 Qs each) → 10 caselet Qs.
    assert len(caselet_qs) == 10
    assert any(q["case_scenario"] for q in caselet_qs)
    assert any(q["marks"] == 2 for q in caselet_qs)
    for q in start["questions"]:
        _assert_no_key(q)


def test_concept_source(client):
    db = _fresh_session(client)
    concept_id = db.scalar(select(Concept.id).order_by(Concept.title))
    db.close()
    r = client.get(f"/api/concept/{concept_id}/source")
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert chunks
    assert chunks[0]["pages"] == "12-13"
