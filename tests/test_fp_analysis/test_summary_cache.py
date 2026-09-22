"""Tests for SummaryCache persistent JSON cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_client.fp_analysis.function_summary import CachedFunctionData, FunctionSummary
from llm_client.fp_analysis.summary_cache import SummaryCache


@pytest.fixture
def cache(tmp_path: Path) -> SummaryCache:
    """Create a SummaryCache backed by a temp directory."""
    return SummaryCache(cache_root=tmp_path / "summary_cache")


def _make_summary(
    function_name: str = "my_func",
    signature: str = "void my_func(void)",
    file_path: str = "src/my.c",
    body_available: bool = False,
    analysis: str = "",
) -> FunctionSummary:
    return FunctionSummary(
        function_name=function_name,
        signature=signature,
        file_path=file_path,
        body_available=body_available,
        analysis=analysis,
    )


class TestSummaryCacheSetGet:
    def test_set_and_get(self, cache: SummaryCache):
        summary = _make_summary(
            function_name="my_alloc",
            analysis="Allocates memory",
        )
        cache.set("testproj", "my_alloc", "memory_leak", summary)

        retrieved = cache.get("testproj", "my_alloc", "memory_leak")
        assert retrieved is not None
        assert retrieved.function_name == "my_alloc"
        assert isinstance(retrieved, FunctionSummary)
        assert retrieved.analysis == "Allocates memory"

    def test_get_missing(self, cache: SummaryCache):
        assert cache.get("nonexistent", "nope", "memory_leak") is None

    def test_get_wrong_bug_type(self, cache: SummaryCache):
        summary = _make_summary(function_name="my_alloc")
        cache.set("testproj", "my_alloc", "memory_leak", summary)
        assert cache.get("testproj", "my_alloc", "null_pointer") is None


class TestSummaryCacheHas:
    def test_has_existing(self, cache: SummaryCache):
        summary = _make_summary(function_name="my_alloc")
        cache.set("testproj", "my_alloc", "memory_leak", summary)
        assert cache.has("testproj", "my_alloc", "memory_leak") is True

    def test_has_missing(self, cache: SummaryCache):
        assert cache.has("no", "such", "entry") is False

    def test_has_wrong_bug_type(self, cache: SummaryCache):
        summary = _make_summary(function_name="my_alloc")
        cache.set("testproj", "my_alloc", "memory_leak", summary)
        assert cache.has("testproj", "my_alloc", "null_pointer") is False


class TestSummaryCacheMerge:
    def test_merge_different_bug_types(self, cache: SummaryCache):
        s1 = _make_summary(
            function_name="my_func",
            analysis="leak analysis",
        )
        s2 = _make_summary(
            function_name="my_func",
            analysis="null analysis",
        )

        cache.set("testproj", "my_func", "memory_leak", s1)
        cache.set("testproj", "my_func", "null_pointer", s2)

        r1 = cache.get("testproj", "my_func", "memory_leak")
        assert r1 is not None
        assert r1.analysis == "leak analysis"

        r2 = cache.get("testproj", "my_func", "null_pointer")
        assert r2 is not None
        assert r2.analysis == "null analysis"

    def test_merge_then_overwrite_bug_type(self, cache: SummaryCache):
        s1 = _make_summary(function_name="my_func", analysis="v1")
        s2 = _make_summary(function_name="my_func", analysis="v2")

        cache.set("testproj", "my_func", "memory_leak", s1)
        cache.set("testproj", "my_func", "memory_leak", s2)

        r = cache.get("testproj", "my_func", "memory_leak")
        assert r is not None
        assert r.analysis == "v2"


class TestSummaryCacheClear:
    def test_clear_all(self, cache: SummaryCache):
        s1 = _make_summary(function_name="f1")
        s2 = _make_summary(function_name="f2")
        cache.set("proj_a", "f1", "memory_leak", s1)
        cache.set("proj_b", "f2", "memory_leak", s2)

        cache.clear()
        assert cache.get("proj_a", "f1", "memory_leak") is None
        assert cache.get("proj_b", "f2", "memory_leak") is None

    def test_clear_project(self, cache: SummaryCache):
        s1 = _make_summary(function_name="f1")
        cache.set("proj_a", "f1", "memory_leak", s1)
        cache.clear(project="proj_a")
        assert cache.get("proj_a", "f1", "memory_leak") is None

    def test_clear_function(self, cache: SummaryCache):
        s1 = _make_summary(function_name="f1")
        s2 = _make_summary(function_name="f2")
        cache.set("proj_a", "f1", "memory_leak", s1)
        cache.set("proj_a", "f2", "memory_leak", s2)

        cache.clear(project="proj_a", function_name="f1")
        assert cache.get("proj_a", "f1", "memory_leak") is None
        assert cache.get("proj_a", "f2", "memory_leak") is not None


class TestSummaryCacheGetOrGenerate:
    def test_cache_hit(self, cache: SummaryCache):
        summary = _make_summary(function_name="my_func", analysis="cached")
        cache.set("testproj", "my_func", "memory_leak", summary)

        generated = False

        def generate_fn():
            nonlocal generated
            generated = True
            return _make_summary(function_name="my_func", analysis="generated")

        result = cache.get_or_generate("testproj", "my_func", "memory_leak", generate_fn)
        assert result.analysis == "cached"
        assert generated is False  # Cache hit, no generation

    def test_cache_miss(self, cache: SummaryCache):
        generated = False

        def generate_fn():
            nonlocal generated
            generated = True
            return _make_summary(function_name="new_func", analysis="fresh")

        result = cache.get_or_generate("testproj", "new_func", "memory_leak", generate_fn)
        assert result.analysis == "fresh"
        assert generated is True  # Cache miss, generation called


class TestSummaryCachePersistence:
    def test_json_file_created(self, cache: SummaryCache):
        summary = _make_summary(function_name="my_func")
        cache.set("testproj", "my_func", "memory_leak", summary)

        cache_path = cache.get_cache_path("testproj", "my_func")
        assert cache_path.exists()

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["function_name"] == "my_func"
        assert data["project"] == "testproj"
        assert "memory_leak" in data["summaries"]

    def test_round_trip_with_merge(self, cache: SummaryCache):
        """Verify JSON file structure after setting multiple bug types."""
        s1 = _make_summary(function_name="my_func", analysis="leak analysis")
        s2 = _make_summary(function_name="my_func", analysis="null analysis")

        cache.set("testproj", "my_func", "memory_leak", s1)
        cache.set("testproj", "my_func", "null_pointer", s2)

        # Read the JSON file directly
        cache_path = cache.get_cache_path("testproj", "my_func")
        raw = json.loads(cache_path.read_text(encoding="utf-8"))

        assert len(raw["summaries"]) == 2
        assert "memory_leak" in raw["summaries"]
        assert "null_pointer" in raw["summaries"]
        assert raw["summaries"]["memory_leak"]["analysis"] == "leak analysis"
        assert raw["summaries"]["null_pointer"]["analysis"] == "null analysis"
