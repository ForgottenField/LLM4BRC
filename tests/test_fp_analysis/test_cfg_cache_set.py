"""Tests for the multi-translation-unit CFG cache view."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from llm_client.fp_analysis import cfg_cache_set as ccs
from llm_client.fp_analysis.cfg_cache_set import CFGCacheSet, find_fn_key

PROJECT = "faiss"


def _signature(short: str, arg: str = "int") -> str:
    return f"void Ns::{short}({arg})"


def _entry(signature: str, raw_dump: str = "") -> dict:
    return {
        "function_name": signature,
        "source_file": None,          # caches never populate this
        "blocks": {},
        "entry_block_id": "B1",
        "exit_block_id": "B0",
        "raw_dump": raw_dump,
    }


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the cache lookup at ``<tmp>/cfg_cache/<project>/``."""
    monkeypatch.setattr(ccs, "_REPO_ROOT", tmp_path)

    def fake_cache_path(source_file: str, project_name: str) -> Path:
        digest = hashlib.md5(str(Path(source_file).resolve()).encode()).hexdigest()
        return Path("cfg_cache") / project_name / f"{digest}.json"

    monkeypatch.setattr(ccs, "_cfg_cache_path", fake_cache_path)
    return tmp_path


def _write_cache(
    cache_root: Path, source_file: Path, entries: list[dict]
) -> None:
    digest = hashlib.md5(str(source_file.resolve()).encode()).hexdigest()
    target = cache_root / "cfg_cache" / PROJECT / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({e["function_name"]: e for e in entries}))


class TestCFGCacheSet:
    def _sources(self, tmp_path: Path) -> tuple[Path, Path]:
        src = tmp_path / "src"
        src.mkdir(exist_ok=True)
        a, b = src / "a.cpp", src / "b.cpp"
        a.write_text("// a\n")
        b.write_text("// b\n")
        return a, b

    def test_merges_several_translation_units(
        self, cache_root: Path, tmp_path: Path
    ) -> None:
        a, b = self._sources(tmp_path)
        _write_cache(cache_root, a, [_entry(_signature("only_in_a"))])
        _write_cache(cache_root, b, [_entry(_signature("only_in_b"))])

        cache_set = CFGCacheSet(PROJECT, source_root=str(tmp_path))
        assert cache_set.ensure(a) is True
        assert cache_set.ensure(b) is True

        merged = cache_set.as_dict()
        assert set(merged) == {_signature("only_in_a"), _signature("only_in_b")}
        assert cache_set.loaded_files == [str(a), str(b)]
        # The returned dict IS the live view — later loads are visible.
        assert cache_set.as_dict() is merged

    def test_raw_dump_is_dropped_at_load(
        self, cache_root: Path, tmp_path: Path
    ) -> None:
        a, _ = self._sources(tmp_path)
        _write_cache(cache_root, a, [_entry(_signature("big"), raw_dump="D" * 100000)])

        cache_set = CFGCacheSet(PROJECT, source_root=str(tmp_path))
        cache_set.ensure(a)
        assert cache_set.as_dict()[_signature("big")].raw_dump == ""

    def test_missing_cache_is_recorded_not_fatal(
        self, cache_root: Path, tmp_path: Path, caplog
    ) -> None:
        a, b = self._sources(tmp_path)
        _write_cache(cache_root, a, [_entry(_signature("present"))])
        header = tmp_path / "src" / "thing.h"
        header.write_text("// h\n")

        cache_set = CFGCacheSet(PROJECT, source_root=str(tmp_path))
        assert cache_set.ensure(a) is True
        assert cache_set.ensure(b) is False
        assert cache_set.ensure(header) is False
        # Lookups are memoized: a second probe does not re-report.
        assert cache_set.ensure(b) is False
        assert len(cache_set.missing_files) == 2

        with caplog.at_level(logging.WARNING, logger=ccs.__name__):
            cache_set.warn_missing()
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert str(b) in warnings[0]
        assert str(header) not in warnings[0]   # headers are expected to be absent

    def test_ensure_is_idempotent(self, cache_root: Path, tmp_path: Path) -> None:
        a, _ = self._sources(tmp_path)
        _write_cache(cache_root, a, [_entry(_signature("f"))])
        cache_set = CFGCacheSet(PROJECT, source_root=str(tmp_path))
        cache_set.ensure(a)
        cache_set.ensure(a)
        assert cache_set.loaded_files == [str(a)]

    def test_first_translation_unit_wins_on_key_collision(
        self, cache_root: Path, tmp_path: Path
    ) -> None:
        a, b = self._sources(tmp_path)
        shared = _signature("shared")
        _write_cache(cache_root, a, [_entry(shared, raw_dump="FROM_A")])
        _write_cache(cache_root, b, [_entry(shared, raw_dump="FROM_B")])

        cache_set = CFGCacheSet(PROJECT, source_root=str(tmp_path))
        cache_set.ensure(a)
        cache_set.ensure(b)
        assert cache_set.key_source_file(shared) == str(a)


class TestFindFnKey:
    """The merged cache can hold the same method from several TUs."""

    def test_prefers_the_requested_file(
        self, cache_root: Path, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        a, b = src / "a.cpp", src / "b.cpp"
        a.write_text("// a\n")
        b.write_text("// b\n")
        key_a, key_b = _signature("run", "int"), _signature("run", "double")
        _write_cache(cache_root, a, [_entry(key_a)])
        _write_cache(cache_root, b, [_entry(key_b)])

        cache_set = CFGCacheSet(PROJECT, source_root=str(tmp_path))
        cache_set.ensure(a)
        cache_set.ensure(b)

        assert cache_set.find_key("run", source_file=str(b)) == key_b
        assert cache_set.find_key("run", source_file=str(a)) == key_a
        # No file hint: deterministic first-loaded key.
        assert cache_set.find_key("run") == key_a

    def test_exact_method_component_beats_substring(self) -> None:
        cache = {
            _signature("setFromLocalAddress"): object(),
            _signature("setFromLocalAddr"): object(),
        }
        assert find_fn_key(cache, "setFromLocalAddr") == _signature("setFromLocalAddr")

    def test_pointer_return_type_does_not_hide_the_method_name(self) -> None:
        """``T *f(...)`` must resolve to ``f``, not to ``*f`` or to ``f_header``.

        The return type's ``*`` used to survive into the method component, so
        the exact-match stage found nothing and the substring fallback returned
        whichever same-prefixed function happened to be dumped first
        (``read_index`` → ``read_index_header``).
        """
        cache = {
            "static void read_index_header(IOReader *f)": object(),
            "faiss::Index *read_index(IOReader *f, int io_flags)": object(),
        }
        assert find_fn_key(cache, "read_index") == (
            "faiss::Index *read_index(IOReader *f, int io_flags)"
        )

    def test_template_arguments_do_not_become_the_method_name(self) -> None:
        """``heap_push<faiss::CMax<float, long>>`` is ``heap_push``, not ``long>>``."""
        cache = {
            "template<> inline void heap_push<faiss::CMax<float, long>>("
            "faiss::CMax<float, long> &heap, float v)": object(),
            "std::allocator_traits<std::allocator<long>>::allocate": object(),
        }
        assert find_fn_key(cache, "heap_push") == (
            "template<> inline void heap_push<faiss::CMax<float, long>>("
            "faiss::CMax<float, long> &heap, float v)"
        )

    def test_unknown_function(self) -> None:
        assert find_fn_key({}, "nope") is None
        assert find_fn_key({_signature("f"): object()}, "") is None


class TestMethodComponent:
    @pytest.mark.parametrize("name,expected", [
        ("void SocketAddress::setFromLocalAddr(int)", "setFromLocalAddr"),
        ("faiss::Index *read_index(IOReader *f, int io_flags)", "read_index"),
        ("template<> inline void heap_push<faiss::CMax<float, long>>(T &h)",
         "heap_push"),
        ("AlignedTableTightAlloc<float, 32> &operator=(const T &other)",
         "operator="),
        ("IndexScalarQuantizer::search", "search"),
        ("", ""),
    ])
    def test_reduces_signature_to_method_name(self, name, expected) -> None:
        assert ccs._method_component(name) == expected


class TestAddCacheFile:
    def test_explicit_cache_file_is_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "explicit.json"
        path.write_text(json.dumps({_signature("manual"): _entry(_signature("manual"))}))

        cache_set = CFGCacheSet(PROJECT)
        assert cache_set.add_cache_file(path) == 1
        assert _signature("manual") in cache_set.as_dict()

    def test_missing_cache_file(self, tmp_path: Path) -> None:
        cache_set = CFGCacheSet(PROJECT)
        assert cache_set.add_cache_file(tmp_path / "nope.json") == 0
        assert cache_set.as_dict() == {}
