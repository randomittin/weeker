"""T-17 gate (M2): objective generation, blueprint weights, verify_concepts.

* objective generation parses with a fake LLM transport keyed by concept id;
* blueprint normalization sums to 1.0 with and without manifest overrides, and
  overridden chapters keep their exact weight;
* ``verify_concepts`` passes on well-formed fixture data and fails on each of
  its three failure modes.
"""

from __future__ import annotations

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atlas.core.models import (
    Base,
    Chapter,
    Chunk,
    Concept,
    ConceptChunk,
    Course,
    Objective,
    Source,
)
from atlas.ingest.objectives import (
    compute_blueprint_weights,
    generate_objectives,
    run_objectives,
)
from atlas.verify.concept_checks import verify_concepts


class FakeLLM:
    """Returns one objective per requested concept id, echoing the batch block."""

    def __init__(self):
        self.calls = 0

    def request(self, model, system, user):
        self.calls += 1
        ids = [ln.split("id: ", 1)[1] for ln in user.splitlines() if ln.startswith("id: ")]
        out = {
            cid: {
                "objectives": [
                    {
                        "text": f"Learner can define concept {cid[:8]}",
                        "bloom": "understand",
                        "assessment_style": "definition",
                    }
                ]
            }
            for cid in ids
        }
        return json.dumps(out)


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed(s, *, n_ch1=2, n_ch2=1, link=True, chapter=True):
    course = Course(title="NISM")
    s.add(course)
    s.flush()
    source = Source(course_id=course.id, filename="wb.pdf", sha256="a" * 64)
    s.add(source)
    s.flush()
    ch1 = Chapter(course_id=course.id, ordinal=1, title="Basics", page_start=1, page_end=5)
    ch2 = Chapter(course_id=course.id, ordinal=2, title="Taxation", page_start=6, page_end=10)
    s.add_all([ch1, ch2])
    s.flush()
    chunk = Chunk(
        course_id=course.id,
        source_id=source.id,
        chapter_id=ch1.id,
        content="grounding passage",
        content_hash="c" * 64,
        token_count=3,
        page_start=1,
        page_end=1,
    )
    s.add(chunk)
    s.flush()

    concepts = []
    for i in range(n_ch1):
        concepts.append((ch1, f"B{i}"))
    for i in range(n_ch2):
        concepts.append((ch2, f"T{i}"))
    for ch, title in concepts:
        c = Concept(
            course_id=course.id,
            chapter_id=ch.id if chapter else None,
            title=title,
            description=f"{title} desc",
            review_status="accepted",
        )
        s.add(c)
        s.flush()
        if link:
            s.add(ConceptChunk(concept_id=c.id, chunk_id=chunk.id, cos=0.9))
    s.flush()
    return course


def test_generate_objectives_parses_fake_llm():
    with _session() as s:
        course = _seed(s)
        made = generate_objectives(s, course, transport=FakeLLM())
        s.flush()
        assert made == 3  # 2 + 1 reviewed concepts, one objective each
        for c in s.query(Concept).all():
            objs = s.query(Objective).filter(Objective.concept_id == c.id).all()
            assert len(objs) == 1
            assert objs[0].bloom == "understand"
            assert objs[0].assessment_style == "definition"


def test_generate_objectives_batches_over_ten():
    with _session() as s:
        course = _seed(s, n_ch1=12, n_ch2=0)
        fake = FakeLLM()
        made = generate_objectives(s, course, transport=fake)
        s.flush()
        assert made == 12
        assert fake.calls == 2  # 12 concepts -> two batches of 10


def test_blueprint_weights_normalize_without_overrides():
    with _session() as s:
        course = _seed(s, n_ch1=3, n_ch2=1)
        w = compute_blueprint_weights(s, course)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        # base ∝ concept counts: Basics(3) vs Taxation(1) -> 0.75 / 0.25
        assert abs(w["Basics"] - 0.75) < 1e-9
        assert abs(w["Taxation"] - 0.25) < 1e-9


def test_blueprint_override_is_authoritative_and_normalized():
    with _session() as s:
        course = _seed(s, n_ch1=3, n_ch2=1)
        w = compute_blueprint_weights(s, course, overrides={"Taxation": 0.6})
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert abs(w["Taxation"] - 0.6) < 1e-9  # exact override retained
        assert abs(w["Basics"] - 0.4) < 1e-9    # remaining mass, sole other chapter


def test_verify_concepts_passes_on_good_fixture():
    with _session() as s:
        course = _seed(s)
        run_objectives(s, course, transport=FakeLLM())
        s.commit()
        assert verify_concepts(s, course.id) == []


def test_verify_concepts_flags_missing_objective():
    with _session() as s:
        course = _seed(s)
        # blueprint present but no objectives generated
        run_objectives(s, course, transport=FakeLLM())
        s.query(Objective).delete()
        s.flush()
        fails = verify_concepts(s, course.id)
        assert any("without an objective" in f for f in fails)


def test_verify_concepts_flags_orphans():
    with _session() as s:
        course = _seed(s, link=False)
        run_objectives(s, course, transport=FakeLLM())
        s.commit()
        fails = verify_concepts(s, course.id)
        assert any("no source chunk" in f for f in fails)


def test_verify_concepts_flags_unnormalized_blueprint():
    with _session() as s:
        course = _seed(s)
        generate_objectives(s, course, transport=FakeLLM())
        # deliberately skip blueprint persistence
        s.commit()
        fails = verify_concepts(s, course.id)
        assert any("blueprint" in f for f in fails)
