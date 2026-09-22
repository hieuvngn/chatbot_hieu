"""Tests for the structured entity tables (departments, instructors, programs, terms)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import generate_data as gd
import generate_entities as ge
from rag_core import RagCore
from rag_core.entities import (
    chunk_entity_bundle,
    load_entity_bundle,
)
from rag_core.models import Department, Instructor, Program, Term


@pytest.fixture()
def entity_data_dir(tmp_path: Path) -> Path:
    """A clean data dir with courses + documents + the four entity files."""
    gd.write(gd.generate(seed=gd.SEED), tmp_path)
    ge.write(ge.generate(seed=ge.SEED, data_dir=tmp_path), tmp_path)
    return tmp_path


def test_generator_emits_four_files(tmp_path: Path) -> None:
    gd.write(gd.generate(seed=gd.SEED), tmp_path)
    ge.write(ge.generate(seed=ge.SEED, data_dir=tmp_path), tmp_path)
    assert (tmp_path / "departments.json").exists()
    assert (tmp_path / "instructors.json").exists()
    assert (tmp_path / "programs.json").exists()
    assert (tmp_path / "terms.json").exists()


def test_load_entity_bundle_returns_all_four_lists(entity_data_dir: Path) -> None:
    bundle = load_entity_bundle(entity_data_dir)
    assert len(bundle.departments) > 0
    assert len(bundle.instructors) > 0
    assert len(bundle.programs) > 0
    assert len(bundle.terms) > 0


def test_load_entity_bundle_handles_missing_files(tmp_path: Path) -> None:
    bundle = load_entity_bundle(tmp_path)
    assert bundle.departments == ()
    assert bundle.instructors == ()
    assert bundle.programs == ()
    assert bundle.terms == ()


def test_instructor_courses_are_known_course_codes(entity_data_dir: Path) -> None:
    bundle = load_entity_bundle(entity_data_dir)
    course_codes = {c["code"] for c in json.loads((entity_data_dir / "courses.json").read_text())}
    for instructor in bundle.instructors:
        for code, _role in instructor.courses:
            assert code in course_codes, f"{instructor.id} teaches unknown {code}"


def test_program_courses_are_known(entity_data_dir: Path) -> None:
    bundle = load_entity_bundle(entity_data_dir)
    course_codes = {c["code"] for c in json.loads((entity_data_dir / "courses.json").read_text())}
    for program in bundle.programs:
        for code in program.required_courses + program.elective_courses:
            assert code in course_codes, f"{program.id} mentions unknown {code}"


def test_term_offered_courses_are_known(entity_data_dir: Path) -> None:
    bundle = load_entity_bundle(entity_data_dir)
    course_codes = {c["code"] for c in json.loads((entity_data_dir / "courses.json").read_text())}
    instructor_ids = {i.id for i in bundle.instructors}
    for term in bundle.terms:
        for code, instructor_id, _schedule in term.offered:
            assert code in course_codes
            assert instructor_id in instructor_ids


def test_chunk_entity_bundle_assigns_entity_source(entity_data_dir: Path) -> None:
    bundle = load_entity_bundle(entity_data_dir)
    chunks = chunk_entity_bundle(bundle)
    assert chunks, "expected entity chunks"
    kinds = {c.source.entity_type for c in chunks}
    assert kinds == {"department", "instructor", "program", "term"}


def test_entity_source_kind_matches_entity_type(entity_data_dir: Path) -> None:
    bundle = load_entity_bundle(entity_data_dir)
    for chunk in chunk_entity_bundle(bundle):
        assert chunk.source.kind == chunk.source.entity_type
        assert chunk.source.entity_id != ""
        assert chunk.source.document_id == chunk.source.entity_id


def test_dedupe_by_source_handles_entities(entity_data_dir: Path) -> None:
    from rag_core.index import dedupe_by_source

    bundle = load_entity_bundle(entity_data_dir)
    chunks = chunk_entity_bundle(bundle)
    # Duplicate one entity chunk to confirm dedupe uses (entity_type, entity_id).
    chunks.append(chunks[0])
    unique = dedupe_by_source(chunks, limit=len(chunks))
    assert len(unique) == len(chunks) - 1


def test_ragcore_ingests_entity_chunks(entity_data_dir: Path) -> None:
    class _StubEmbedder:
        dim = 8

        def embed_batch(self, texts):
            import numpy as np
            return [
                np.random.RandomState(hash(t) & 0xFFFFFFFF).randn(8).astype("float32").tolist()
                for t in texts
            ]

        def embed_query(self, t):
            return self.embed_batch([t])[0]

    class _StubGenerator:
        def generate(self, *args, **kwargs):
            return "stub"

        def stream(self, *args, **kwargs):
            return iter([])

        def generate_fallback(self, q):
            return "fallback"

    core = RagCore(
        data_dir=entity_data_dir,
        embedder=_StubEmbedder(),
        generator=_StubGenerator(),
    )
    entity_chunks = [c for c in core._index.chunks if c.source.entity_type]
    assert len(entity_chunks) > 0
    assert {c.source.entity_type for c in entity_chunks} == {
        "department",
        "instructor",
        "program",
        "term",
    }


def test_ragcore_entities_property_returns_bundle(entity_data_dir: Path) -> None:
    class _StubEmbedder:
        dim = 8

        def embed_batch(self, texts):
            import numpy as np
            return [
                np.random.RandomState(hash(t) & 0xFFFFFFFF).randn(8).astype("float32").tolist()
                for t in texts
            ]

        def embed_query(self, t):
            return self.embed_batch([t])[0]

    class _StubGenerator:
        def generate(self, *args, **kwargs):
            return "stub"

        def stream(self, *args, **kwargs):
            return iter([])

        def generate_fallback(self, q):
            return "fallback"

    core = RagCore(
        data_dir=entity_data_dir,
        embedder=_StubEmbedder(),
        generator=_StubGenerator(),
    )
    bundle = core.entities
    assert isinstance(bundle, type(load_entity_bundle(entity_data_dir)))
    assert len(bundle.instructors) > 0