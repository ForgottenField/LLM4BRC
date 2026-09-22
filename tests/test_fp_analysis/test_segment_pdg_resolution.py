"""Segment → PDG resolution by (file, line range) evidence.

A segment carries only a bare method name, and for events inside a macro
expansion not even that (the macro's name, or nothing).  Resolving such a name
against the SDG's function table is a first-match game that lands on a
same-named function from another class, another TU, or libstdc++ — which is how
``read_index`` used to extract the branches of ``read_index_header`` and
``search`` those of ``IndexPQ::search`` instead of the bug's
``MultiIndexQuantizer::search``.

These tests pin the resolution rules that replaced the name-only lookup:

* a name-resolved PDG is accepted only when it has nodes inside the segment's
  line range;
* otherwise the best covering PDG from the segment's own file wins;
* the PDG path is taken only when the PDG has a branch condition or a call in
  range — an empty PDG must fall through to the range-aware CFG fallback rather
  than silently reporting zero branches (``clone_IndexIVF``'s ``TRYCLONE``
  branches live only in the CFG).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_client.fp_analysis.cfg_branch_extractor import CFGBranchExtractor
from llm_client.fp_analysis.pdg_models import ControlDepEdge, FunctionPDG, PDGNode

FILE_A = "/proj/IndexPQ.cpp"
FILE_B = "/proj/io.cpp"


def _node(node_id: str, line: int, kind: str = "binary_op") -> PDGNode:
    return PDGNode(id=node_id, kind=kind, source_line=line, expression=f"x_{line}")


def _pdg(name: str, source_file: str, nodes: list[PDGNode], branches: bool) -> FunctionPDG:
    edges = []
    if branches:
        edges = [ControlDepEdge(source_id=n.id, target_id=n.id, kind="if_branch")
                 for n in nodes]
    return FunctionPDG(function_name=name, source_file=source_file,
                       nodes={n.id: n for n in nodes}, control_dep_edges=edges)


class FakeSDG:
    def __init__(self, pdgs: list[FunctionPDG]):
        self.pdgs = {p.function_name: p for p in pdgs}

    def get_pdg(self, name: str):
        if name in self.pdgs:
            return self.pdgs[name]
        for qname, pdg in self.pdgs.items():
            if qname.endswith("::" + name) or qname.split("::")[-1] == name:
                return pdg
        return None

    def build_file_index(self):
        index: dict[str, list[FunctionPDG]] = {}
        for pdg in self.pdgs.values():
            index.setdefault(pdg.source_file, []).append(pdg)
        return index


@pytest.fixture
def extractor_factory():
    def build(pdgs: list[FunctionPDG], cfg_cache: dict | None = None):
        return CFGBranchExtractor(cfg_cache or {}, FakeSDG(pdgs), None)

    return build


class TestResolveSegmentPdg:
    def test_name_inside_the_line_range_is_accepted(self, extractor_factory) -> None:
        pdg = _pdg("IndexPQ::search", FILE_A, [_node("a", 160)], branches=True)
        ex = extractor_factory([pdg])
        assert ex.resolve_segment_pdg("search", FILE_A, 152, 249) is pdg

    def test_same_named_function_elsewhere_is_rejected_by_lines(
        self, extractor_factory
    ) -> None:
        """``search`` must not resolve to a class whose ``search`` is elsewhere."""
        wrong = _pdg("IndexPQ::search", FILE_A, [_node("a", 160)], branches=True)
        right = _pdg("MultiIndexQuantizer::search", FILE_A, [_node("b", 880)], branches=True)
        ex = extractor_factory([wrong, right])

        resolved = ex.resolve_segment_pdg("search", FILE_A, 874, 921)

        assert resolved is right

    def test_file_local_fallback_beats_an_off_file_name_match(
        self, extractor_factory
    ) -> None:
        """A libstdc++ ``operator=`` must lose to the segment's own file.

        ``operator=`` first resolves to ``std::exception_ptr::operator=`` — the
        ``::name`` suffix rule of :meth:`SDG.get_pdg` finds it before the
        segment's own class — but that overload's nodes are nowhere near the
        segment's lines, so the file-local ``faiss`` one wins.
        """
        libstdcxx = _pdg("std::exception_ptr::operator=", "/usr/include/c++/9/exc.h",
                         [_node("a", 500)], branches=True)
        local = _pdg("faiss::AlignedTableTightAlloc::operator=", FILE_A,
                     [_node("b", 100)], branches=True)
        ex = extractor_factory([libstdcxx, local])

        assert ex.resolve_segment_pdg("operator=", FILE_A, 100, 101) is local

    def test_unresolvable_name_and_file_returns_none(self, extractor_factory) -> None:
        ex = extractor_factory([_pdg("frobnicate", FILE_A, [_node("a", 1)], branches=True)])
        assert ex.resolve_segment_pdg("own_fields", "/proj/IndexScalarQuantizer.cpp",
                                      52, 52) is None

    def test_no_file_hint_keeps_the_name_resolution(self, extractor_factory) -> None:
        """Without a file there is nothing to verify against — keep the name."""
        pdg = _pdg("Ns::frobnicate", FILE_A, [_node("a", 5)], branches=True)
        ex = extractor_factory([pdg])
        assert ex.resolve_segment_pdg("frobnicate", "", 1, 9) is pdg


class TestPdgYieldsGate:
    def test_pdg_with_branch_in_range_yields(self, extractor_factory) -> None:
        pdg = _pdg("Ns::frobnicate", FILE_A, [_node("a", 5)], branches=True)
        assert extractor_factory([pdg])._pdg_yields(pdg, 1, 9) is True

    def test_pdg_with_nothing_in_range_does_not_yield(self, extractor_factory) -> None:
        pdg = _pdg("Ns::frobnicate", FILE_A, [_node("a", 50)], branches=True)
        assert extractor_factory([pdg])._pdg_yields(pdg, 1, 9) is False

    def test_call_in_range_yields(self, extractor_factory) -> None:
        call = _node("c", 5, kind="call_expr")
        pdg = _pdg("Ns::frobnicate", FILE_A, [call], branches=False)
        pdg.nodes["c"].is_call = True
        assert extractor_factory([pdg])._pdg_yields(pdg, 1, 9) is True

    def _extract(self, ex, fn_name: str, fn_file: str, line: int):
        return ex.extract(
            segment_index=0, function_name=fn_name, function_file=fn_file,
            line_start=line, line_end=line,
            postcondition=SimpleNamespace(
                function_name="clone_IndexIVF", source_line=line,
                condition_expr="", branch_taken=True,
            ),
        )

    def test_empty_pdg_falls_back_to_the_cfg(self, extractor_factory) -> None:
        """An empty PDG must not swallow the CFG's branches.

        ``clone_IndexIVF``'s four ``TRYCLONE`` branches are macro expansions
        the PDG builder recorded as nodes but not as control-dependence edges —
        the PDG resolves, has nodes in range, and still yields nothing.  The
        CFG holds the block terminators, so it has to get a turn.
        """
        pdg = _pdg("faiss::Cloner::clone_IndexIVF", FILE_A, [_node("a", 72)],
                   branches=False)
        ex = extractor_factory([pdg])
        used = []
        ex._extract_via_pdg = lambda **kw: used.append("pdg")

        self._extract(ex, "faiss::Cloner::clone_IndexIVF", FILE_A, 72)

        assert used == [], "the empty PDG path must not be taken"

    def test_pdg_with_a_branch_in_range_takes_the_pdg_path(
        self, extractor_factory
    ) -> None:
        pdg = _pdg("faiss::Cloner::clone_IndexIVF", FILE_A, [_node("a", 72)],
                   branches=True)
        ex = extractor_factory([pdg])
        used = []
        ex._extract_via_pdg = lambda **kw: used.append("pdg")

        self._extract(ex, "faiss::Cloner::clone_IndexIVF", FILE_A, 72)

        assert used == ["pdg"]
