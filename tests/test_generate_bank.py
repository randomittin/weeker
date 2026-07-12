"""T-24 gate: verify_bank (M3 bank gate) — pass on a clean bank, fail per criterion."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atlas.core.models import Base, Question
from atlas.verify.bank_checks import verify_bank
from tests.test_learn_fixtures import seed_concept, seed_course


def _mem():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _good_options():
    return [
        {"key": "A", "text": "right"},
        {"key": "B", "text": "wrong b", "misconception": "b"},
        {"key": "C", "text": "wrong c", "misconception": "c"},
        {"key": "D", "text": "wrong d", "misconception": "d"},
    ]


def _clean_log():
    return {
        "G1": {"passed": True, "reason": "ok"},
        "G2": {"passed": True, "reason": "ok", "supported": True, "ambiguous_option": None},
        "G4": {"passed": True, "reason": "ok"},
        "G5": {"passed": True, "reason": "ok"},
        "G3": {"passed": True, "reason": "ok", "disputed": False, "votes": ["A"]},
    }


def _q(db, course, concept, *, options=None, gate_log=None, status="active"):
    q = Question(
        course_id=course.id,
        concept_id=concept.id,
        stem="stem",
        options=options or _good_options(),
        correct_key="A",
        explanation="e",
        difficulty=2,
        status=status,
        gate_log=gate_log or _clean_log(),
    )
    db.add(q)
    db.flush()
    return q


def test_clean_bank_passes():
    db = _mem()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    for _ in range(5):
        _q(db, course, concept)
    report = verify_bank(db, course.id)
    assert report.passed, report.failures
    assert report.total_active == 5
    assert report.blind_agreement == 1.0


def test_ungrounded_fails_grounding():
    db = _mem()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    _q(db, course, concept)
    log = _clean_log()
    log["G2"] = {"passed": False, "reason": "not supported", "supported": False}
    _q(db, course, concept, gate_log=log)
    report = verify_bank(db, course.id)
    assert not report.passed
    assert not report.grounding_ok
    assert any("grounding" in f for f in report.failures)


def test_disputed_active_fails_blind_agreement():
    db = _mem()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    log = _clean_log()
    log["G3"] = {"passed": False, "reason": "disagree", "disputed": True, "votes": ["B"]}
    _q(db, course, concept, gate_log=log)
    report = verify_bank(db, course.id)
    assert not report.passed
    assert report.blind_agreement < 0.95
    assert any("blind agreement" in f for f in report.failures)


def test_near_duplicate_flag_fails():
    db = _mem()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    log = _clean_log()
    log["G5"] = {"passed": True, "reason": "dup", "duplicate_of": "other"}
    _q(db, course, concept, gate_log=log)
    report = verify_bank(db, course.id)
    assert not report.passed
    assert report.near_dup_rate >= 0.02
    assert any("near-dup" in f for f in report.failures)


def test_g4_hit_among_active_fails():
    db = _mem()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    log = _clean_log()
    log["G4"] = {"passed": False, "reason": "verbatim lift"}
    _q(db, course, concept, gate_log=log)
    report = verify_bank(db, course.id)
    assert not report.passed
    assert report.g4_hits == 1
    assert any("G4" in f for f in report.failures)


def test_bad_schema_fails():
    db = _mem()
    course = seed_course(db)
    concept = seed_concept(db, course, "c")
    three = _good_options()[:3]
    _q(db, course, concept, options=three)
    report = verify_bank(db, course.id)
    assert not report.passed
    assert not report.schema_ok
    assert any("schema" in f for f in report.failures)
