"""S5 — concept extraction, merge, prerequisite DAG, concept→chunk map (016 §T-15).

Map: each chapter's chunks are batched under a token budget and sent to the LLM
with ``prompts/concept_extract.txt`` (verbatim). Reduce: concept drafts across
the whole course are embedded and near-duplicates (cosine ≥ ``CONCEPT_MERGE_COS``)
are collapsed, unioning their keywords/misconceptions/source pages. Prerequisite
titles are resolved against surviving concepts, cycles are broken to keep a DAG,
and each concept is linked to its top ``CONCEPT_CHUNK_TOPK`` supporting chunks
(cosine ≥ ``CONCEPT_CHUNK_MIN_COS``). Every persisted concept is left
``review_status='pending'`` for the human pass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core.config import (
    CONCEPT_CHUNK_MIN_COS,
    CONCEPT_CHUNK_TOPK,
    CONCEPT_MERGE_COS,
)
from atlas.core.embed import EmbedTransport, embed
from atlas.core.llm import Transport, complete
from atlas.core.models import Chapter, Chunk, Concept, ConceptChunk, Course
from atlas.ingest.embeddings import retrieve_by_vector
from atlas.ingest.textutil import cosine, count_tokens, load_prompt, render_prompt

# Operational batch budget for one extraction call (not a scoring gate).
EXTRACT_BATCH_TOKENS = 6000

_CONCEPT_SCHEMA = {
    "type": "object",
    "required": ["concepts"],
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "description", "difficulty"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "difficulty": {"type": "integer"},
                    "prerequisites": {"type": "array"},
                    "keywords": {"type": "array"},
                    "misconceptions": {"type": "array"},
                    "source_pages": {"type": "array"},
                },
            },
        }
    },
}


@dataclass
class ConceptDraft:
    title: str
    description: str
    difficulty: int
    keywords: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    chapter_id: uuid.UUID | None = None


def _batch_chunks(chunks: list[Chunk]) -> list[list[Chunk]]:
    batches: list[list[Chunk]] = []
    cur: list[Chunk] = []
    tok = 0
    for c in chunks:
        ct = c.token_count or count_tokens(c.content)
        if cur and tok + ct > EXTRACT_BATCH_TOKENS:
            batches.append(cur)
            cur, tok = [], 0
        cur.append(c)
        tok += ct
    if cur:
        batches.append(cur)
    return batches


def extract_chapter_concepts(
    session: Session,
    course: Course,
    chapter: Chapter,
    transport: Transport | None = None,
    model: str = "concept-extract",
) -> list[ConceptDraft]:
    """Run the extraction prompt over a chapter's chunks (batched)."""
    system, user_t = load_prompt("concept_extract")
    chunks = list(
        session.scalars(select(Chunk).where(Chunk.chapter_id == chapter.id).order_by(Chunk.page_start))
    )
    drafts: list[ConceptDraft] = []
    for batch in _batch_chunks(chunks):
        body = "\n\n".join(c.content for c in batch)
        user = render_prompt(
            user_t,
            course_title=course.title,
            chapter_title=chapter.title,
            page_start=chapter.page_start,
            page_end=chapter.page_end,
            chunks=body,
        )
        data = complete(model, system, user, json_schema=_CONCEPT_SCHEMA, transport=transport)
        assert isinstance(data, dict)
        for c in data["concepts"]:
            diff = int(c.get("difficulty", 2))
            drafts.append(
                ConceptDraft(
                    title=str(c["title"]).strip(),
                    description=str(c["description"]).strip(),
                    difficulty=diff if diff in (1, 2, 3) else 2,
                    keywords=[str(x) for x in c.get("keywords", []) or []],
                    misconceptions=[str(x) for x in c.get("misconceptions", []) or []],
                    prerequisites=[str(x) for x in c.get("prerequisites", []) or []],
                    source_pages=[int(x) for x in c.get("source_pages", []) or [] if str(x).isdigit()],
                    chapter_id=chapter.id,
                )
            )
    return drafts


def _dedup(seq: list) -> list:
    seen: dict = {}
    for x in seq:
        seen[x] = None
    return list(seen)


def merge_concepts(
    drafts: list[ConceptDraft], embeddings: np.ndarray, threshold: float = CONCEPT_MERGE_COS
) -> list[ConceptDraft]:
    """Collapse near-duplicate drafts (cosine ≥ threshold) into representatives.

    The lowest-index draft in each cluster is the representative; the others'
    keywords, misconceptions, prerequisites and source pages are unioned into it.
    """
    n = len(drafts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        for j in range(i + 1, n):
            if cosine(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    reps: dict[int, ConceptDraft] = {}
    for i in range(n):
        r = find(i)
        if r not in reps:
            reps[r] = ConceptDraft(
                title=drafts[r].title,
                description=drafts[r].description,
                difficulty=drafts[r].difficulty,
                keywords=list(drafts[r].keywords),
                misconceptions=list(drafts[r].misconceptions),
                prerequisites=list(drafts[r].prerequisites),
                source_pages=list(drafts[r].source_pages),
                chapter_id=drafts[r].chapter_id,
            )
        if i != r:
            rep = reps[r]
            rep.keywords = _dedup(rep.keywords + drafts[i].keywords)
            rep.misconceptions = _dedup(rep.misconceptions + drafts[i].misconceptions)
            rep.prerequisites = _dedup(rep.prerequisites + drafts[i].prerequisites)
            rep.source_pages = _dedup(rep.source_pages + drafts[i].source_pages)
    return [reps[k] for k in sorted(reps)]


def detect_cycle(edges: dict[str, list[str]]) -> list[str]:
    """Return the nodes of one cycle in the prereq graph, or [] if acyclic."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in edges}
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        color[node] = GRAY
        stack.append(node)
        for nxt in edges.get(node, []):
            if nxt not in color:
                continue
            if color[nxt] == GRAY:
                return stack[stack.index(nxt):]
            if color[nxt] == WHITE:
                found = dfs(nxt)
                if found:
                    return found
        color[node] = BLACK
        stack.pop()
        return None

    for n in edges:
        if color[n] == WHITE:
            found = dfs(n)
            if found:
                return found
    return []


def _resolve_prereqs(drafts: list[ConceptDraft]) -> None:
    """Keep only prereq titles that name a surviving concept; break cycles."""
    titles = {d.title.lower(): d.title for d in drafts}
    for d in drafts:
        resolved = []
        for p in d.prerequisites:
            key = p.strip().lower()
            if key in titles and titles[key] != d.title:
                resolved.append(titles[key])
        d.prerequisites = _dedup(resolved)

    # Break cycles by dropping the back-edge that closes each detected cycle.
    while True:
        edges = {d.title: list(d.prerequisites) for d in drafts}
        cyc = detect_cycle(edges)
        if not cyc:
            break
        offender, victim = cyc[-1], cyc[0]
        for d in drafts:
            if d.title == offender and victim in d.prerequisites:
                d.prerequisites.remove(victim)
                break


def map_concept_chunks(
    session: Session,
    course: Course,
    concept: Concept,
    topk: int = CONCEPT_CHUNK_TOPK,
    min_cos: float = CONCEPT_CHUNK_MIN_COS,
) -> int:
    """Link a concept to its top supporting chunks by cosine. Returns link count."""
    if concept.embedding is None:
        return 0
    session.query(ConceptChunk).filter(ConceptChunk.concept_id == concept.id).delete()
    session.flush()
    hits = retrieve_by_vector(
        session, course.id, np.asarray(concept.embedding, dtype=np.float32), topk, min_cos
    )
    for chunk, cos in hits:
        session.add(
            ConceptChunk(concept_id=concept.id, chunk_id=chunk.id, cos=round(float(cos), 5))
        )
    session.flush()
    return len(hits)


def run_concepts(
    session: Session,
    course: Course,
    transport: Transport | None = None,
    embed_transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
) -> list[Concept]:
    """Full S5: extract → merge → resolve prereqs → persist → map to chunks."""
    chapters = list(
        session.scalars(
            select(Chapter).where(Chapter.course_id == course.id).order_by(Chapter.page_start)
        )
    )
    drafts: list[ConceptDraft] = []
    for ch in chapters:
        drafts.extend(extract_chapter_concepts(session, course, ch, transport))
    if not drafts:
        return []

    embed_texts = [f"{d.title}. {d.description}" for d in drafts]
    draft_vecs = embed(embed_texts, transport=embed_transport, cache_dir=cache_dir)
    kept = merge_concepts(drafts, draft_vecs, CONCEPT_MERGE_COS)
    _resolve_prereqs(kept)

    # Recompute embeddings for the surviving (possibly merged) concepts.
    kept_texts = [f"{d.title}. {d.description}" for d in kept]
    kept_vecs = embed(kept_texts, transport=embed_transport, cache_dir=cache_dir)

    session.query(Concept).filter(Concept.course_id == course.id).delete()
    session.flush()
    rows: list[Concept] = []
    for d, vec in zip(kept, kept_vecs, strict=True):
        row = Concept(
            course_id=course.id,
            chapter_id=d.chapter_id,
            title=d.title,
            description=d.description,
            difficulty=d.difficulty,
            prerequisites=list(d.prerequisites),
            keywords=list(d.keywords),
            misconceptions=list(d.misconceptions),
            source_pages=list(d.source_pages),
            review_status="pending",
        )
        row.embedding = np.asarray(vec, dtype=np.float32)
        session.add(row)
        rows.append(row)
    session.flush()

    for row in rows:
        map_concept_chunks(session, course, row)
    return rows
