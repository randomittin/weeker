"""The ``atlas ingest`` command group (016 §T-10..T-17).

``register`` attaches the real ingestion commands to the group created in
``cli/main.py`` (which loads ``atlas.ingest.cli`` and calls ``register`` — no
edit to that file needed):

* ``atlas ingest run`` — the full S0-S6 pipeline: manifest → parse → structure →
  chunk → embeddings → concepts → objectives + blueprint.
* ``atlas ingest review`` — launch the S5 human review TUI.
* ``atlas ingest verify-parse`` — the S1 parse gate per source.
* ``atlas ingest verify`` — the full M1 ingestion gate.
* ``atlas ingest verify-concepts`` — the M2 concept gate.

Every command wires the real stage functions; there is no stub path.
"""

from __future__ import annotations

from pathlib import Path

import typer
from sqlalchemy import select

from atlas.core.db import get_engine, session_scope
from atlas.core.models import Course, Source
from atlas.ingest.chunk import chunk_course
from atlas.ingest.concepts import run_concepts
from atlas.ingest.embeddings import build_hnsw_index, embed_chunks
from atlas.ingest.manifest import load_manifest, upsert
from atlas.ingest.objectives import run_objectives
from atlas.ingest.parse import parse_source, persist_pages
from atlas.ingest.review import ReviewApp
from atlas.ingest.structure import build_structure, persist_structure
from atlas.verify.concept_checks import verify_concepts
from atlas.verify.ingest_checks import check_parse, verify_ingest


def _resolve_course(db, course: str | None) -> Course | None:
    stmt = select(Course)
    if course:
        stmt = stmt.where((Course.slug == course) | (Course.title == course))
    return db.scalars(stmt.order_by(Course.created_at.desc())).first()


def _emit(failures: list[str], ok_msg: str) -> None:
    """Print gate failures (or a success line) and exit with the right code."""
    if failures:
        for f in failures:
            typer.echo(f"FAIL {f}")
        raise typer.Exit(1)
    typer.echo(ok_msg)
    raise typer.Exit(0)


def register(group_app: typer.Typer) -> None:
    """Attach the ingestion commands to the ``atlas ingest`` group."""

    @group_app.command("run")
    def run(
        manifest: str = typer.Option(..., "--manifest", "-m", help="Path to manifest.yaml."),
        corpus_root: str = typer.Option(
            None, "--corpus-root", "-r", help="Root for source paths (default: manifest dir)."
        ),
    ) -> None:
        """Run the full S0-S6 ingestion pipeline for a course."""
        man = load_manifest(manifest)
        root = Path(corpus_root) if corpus_root else Path(manifest).resolve().parent
        engine = get_engine()
        with session_scope() as db:
            course, sources = upsert(db, man, root)

            # S1 parse — every source; track the primary (most pages) for structure.
            primary_source: Source | None = None
            primary_path: Path | None = None
            for src, spec in zip(sources, man.sources, strict=True):
                path = root / spec.path
                result = parse_source(path, page_offset=spec.page_offset)
                persist_pages(db, src, result)
                typer.echo(f"parsed {spec.filename}: {len(result.pages)} pages")
                if primary_source is None or (src.page_count or 0) > (
                    primary_source.page_count or 0
                ):
                    primary_source, primary_path = src, path

            # S2 structure — built from the primary workbook, persisted course-wide.
            structure = build_structure(primary_path, primary_source.filename)
            chapters, sections = persist_structure(db, course, structure)
            typer.echo(
                f"structure ({structure.strategy}): "
                f"{len(chapters)} chapters, {len(sections)} sections"
            )

            # S3 chunk → S4 embed.
            chunks = chunk_course(db, course)
            typer.echo(f"chunked: {len(chunks)} chunks")
            embedded = embed_chunks(db, course)
            build_hnsw_index(engine)
            typer.echo(f"embedded: {embedded} chunks")

            # S5 concepts (leaves review_status='pending').
            concepts = run_concepts(db, course)
            typer.echo(f"concepts: {len(concepts)} pending review")

            # S6 objectives + blueprint (over any already-reviewed concepts).
            made, weights = run_objectives(
                db, course, overrides=man.blueprint.overrides or None
            )
            typer.echo(
                f"objectives: {made} generated; blueprint over {len(weights)} chapters"
            )
        typer.echo("ingestion complete — run 'atlas ingest review' for the S5 human pass.")

    @group_app.command("review")
    def review(
        course: str = typer.Option(None, "--course", "-c", help="Course slug or title."),
    ) -> None:
        """Launch the S5 concept review TUI over pending concepts."""
        engine = get_engine()
        course_id = None
        if course:
            with session_scope() as db:
                row = _resolve_course(db, course)
                if row is None:
                    typer.echo("No matching course — ingest one first.")
                    raise typer.Exit(1)
                course_id = row.id
        ReviewApp(engine, course_id=course_id).run()

    @group_app.command("verify-parse")
    def verify_parse(
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Run the S1 parse gate for every source of a course (M1 stage)."""
        with session_scope() as db:
            row = _resolve_course(db, course)
            if row is None:
                typer.echo("No matching course.")
                raise typer.Exit(1)
            failures: list[str] = []
            for src_id in db.scalars(select(Source.id).where(Source.course_id == row.id)):
                failures += check_parse(db, src_id)
            _emit(failures, "parse gate: PASS")

    @group_app.command("verify")
    def verify(
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Run the full M1 ingestion gate (parse + structure + chunks + embeddings)."""
        with session_scope() as db:
            row = _resolve_course(db, course)
            if row is None:
                typer.echo("No matching course.")
                raise typer.Exit(1)
            _emit(verify_ingest(db, row.id), "M1 ingestion gate: PASS")

    @group_app.command("verify-concepts")
    def verify_concepts_cmd(
        course: str = typer.Option(None, "--course", "-c"),
    ) -> None:
        """Run the M2 concept gate (objectives + blueprint + no orphans)."""
        with session_scope() as db:
            row = _resolve_course(db, course)
            if row is None:
                typer.echo("No matching course.")
                raise typer.Exit(1)
            _emit(verify_concepts(db, row.id), "M2 concept gate: PASS")
