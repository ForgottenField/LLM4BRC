"""Tests for path_seed_extractor — AST-based PDG node matching for seed set
extraction from CSA bug report path events."""

from __future__ import annotations

import pytest

from llm_client.csa_analysis.models import BugPath, EventKind, PathEvent
from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    FunctionSummary,
    PDGNode,
    SDG,
)
from llm_client.fp_analysis.path_seed_extractor import (
    build_caller_map,
    build_path_annotations,
    extract_seeds_from_bug_path,
    find_pdg_nodes_for_event,
)


# ===========================================================================
# Test fixtures
# ===========================================================================


@pytest.fixture
def sample_sdg() -> SDG:
    """SDG with a simple function containing various statement types:
      line 5:  int x = get_value();        (decl, variable=x)
      line 7:  ptr = get_buffer();         (assign, variable=ptr)
      line 8:  if (ptr != nullptr)          (if_cond, no variable)
      line 9:    *ptr = 42;                 (assign via deref, no variable)
      line 11: result = compute(x, ptr);    (call_expr, variable=result,
                                             callee=compute, params=[x, ptr])
      line 12: return result;               (return)
    """
    fn_name = "MyClass::process"
    nodes = {
        "S_5_3": PDGNode(id="S_5_3", kind="decl", variable="x",
                          expression="int x = get_value()", type="int",
                          source_line=5, source_col=3, block_id=1),
        "S_7_5": PDGNode(id="S_7_5", kind="assign", variable="ptr",
                          expression="ptr = get_buffer()", type="int*",
                          source_line=7, source_col=5, block_id=2),
        "S_8_5": PDGNode(id="S_8_5", kind="if_cond", variable="",
                          expression="ptr != nullptr", type="bool",
                          source_line=8, source_col=5, block_id=3),
        "S_9_7": PDGNode(id="S_9_7", kind="assign", variable="",
                          expression="*ptr = 42", type="void",
                          source_line=9, source_col=7, block_id=4),
        "S_11_5": PDGNode(id="S_11_5", kind="call_expr", variable="result",
                           expression="compute(x, ptr)", type="int",
                           source_line=11, source_col=5, block_id=5,
                           is_call=True, callee="compute",
                           actual_params=["x", "ptr"]),
        "S_12_5": PDGNode(id="S_12_5", kind="return", variable="",
                           expression="return result", type="int",
                           source_line=12, source_col=5, block_id=5),
    }
    du = [
        DefUseEdge(def_node_id="S_5_3", use_node_id="S_11_5", variable="x"),
        DefUseEdge(def_node_id="S_7_5", use_node_id="S_8_5", variable="ptr"),
        DefUseEdge(def_node_id="S_11_5", use_node_id="S_12_5", variable="result"),
    ]
    cd = [
        ControlDepEdge(source_id="S_8_5", target_id="S_9_7", kind="if_branch"),
    ]
    pdg = FunctionPDG(
        function_name=fn_name,
        source_file="src/process.cc",
        nodes=nodes,
        def_use_edges=du,
        control_dep_edges=cd,
        summary=FunctionSummary(
            input_variables=[], output_variables=["result"],
        ),
    )
    return SDG(pdgs={fn_name: pdg})


# ===========================================================================
# Tests: PDG node matching (AST-based)
# ===========================================================================


class TestFindPDGNodesForEvent:
    """find_pdg_nodes_for_event() uses only PDG AST metadata for matching."""

    def test_match_assign_by_variable(self, sample_sdg):
        """Event describes 'ptr' assignment → matches PDG node with variable=ptr."""
        event = PathEvent(
            event_kind=EventKind.EVENT,
            file_path="src/process.cc",
            line_number=7,
            description="Assigning 'ptr' = get_buffer()",
        )
        results = find_pdg_nodes_for_event(event, sample_sdg)
        assert len(results) >= 1
        assert any(r[1] == "S_7_5" for r in results)

    def test_match_control_by_kind(self, sample_sdg):
        """Control event → matches if_cond node by kind."""
        event = PathEvent(
            event_kind=EventKind.CONTROL,
            file_path="src/process.cc",
            line_number=8,
            description="Taking true branch",
            branch_condition=True,
        )
        results = find_pdg_nodes_for_event(event, sample_sdg)
        assert len(results) >= 1
        assert any(r[1] == "S_8_5" for r in results)

    def test_match_call_by_callee(self, sample_sdg):
        """Calling event → matches call_expr node with matching callee."""
        event = PathEvent(
            event_kind=EventKind.EVENT,
            file_path="src/process.cc",
            line_number=11,
            description="Calling 'compute'",
        )
        results = find_pdg_nodes_for_event(event, sample_sdg)
        assert len(results) >= 1
        assert any(r[1] == "S_11_5" for r in results)

    def test_match_return_by_kind(self, sample_sdg):
        """Return event → matches return node."""
        event = PathEvent(
            event_kind=EventKind.EVENT,
            file_path="src/process.cc",
            line_number=12,
            description="Returning from 'process'",
        )
        results = find_pdg_nodes_for_event(event, sample_sdg)
        assert len(results) >= 1
        assert any(r[1] == "S_12_5" for r in results)

    def test_match_report_by_trigger_kind(self, sample_sdg):
        """Report event → matches trigger-like node kind at the line."""
        event = PathEvent(
            event_kind=EventKind.REPORT,
            file_path="src/process.cc",
            line_number=9,
            description="Dereference of null pointer",
        )
        results = find_pdg_nodes_for_event(event, sample_sdg)
        assert len(results) >= 1

    def test_no_match_different_file(self, sample_sdg):
        """Event in a file not in SDG → empty results."""
        event = PathEvent(
            event_kind=EventKind.EVENT,
            file_path="other/foo.cc",
            line_number=5,
            description="Assigning 'x' = 0",
        )
        results = find_pdg_nodes_for_event(event, sample_sdg)
        assert len(results) == 0

    def test_no_match_line_zero(self, sample_sdg):
        """Event with line_number=0 → empty results."""
        event = PathEvent(
            event_kind=EventKind.EVENT,
            file_path="src/process.cc",
            line_number=0,
            description="Synthetic event",
        )
        results = find_pdg_nodes_for_event(event, sample_sdg)
        assert len(results) == 0

    def test_conservative_fallback_all_candidates(self):
        """When no AST match found, all candidates at the line are returned."""
        nodes = {
            "S_1_1": PDGNode(id="S_1_1", kind="event", variable="",
                              expression="some_op()", type="void",
                              source_line=1, source_col=1, block_id=1),
            "S_1_2": PDGNode(id="S_1_2", kind="event", variable="",
                              expression="another_op()", type="void",
                              source_line=1, source_col=2, block_id=1),
        }
        pdg = FunctionPDG(function_name="f", source_file="t.cc", nodes=nodes)
        sdg = SDG(pdgs={"f": pdg})
        event = PathEvent(
            event_kind=EventKind.EVENT,
            file_path="t.cc", line_number=1,
            description="Generic event with no variable mention",
        )
        results = find_pdg_nodes_for_event(event, sdg)
        # Both nodes should be included conservatively
        assert len(results) == 2


# ===========================================================================
# Tests: Complete seed extraction from bug path
# ===========================================================================


class TestExtractSeedsFromBugPath:
    """extract_seeds_from_bug_path() builds correct seed sets."""

    def test_single_event_yields_seed(self, sample_sdg):
        """Single path event with matching PDG → at least one seed."""
        bug_path = BugPath(
            path_id="t1", bug_type="NullDereference", category="Logic error",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=7,
                          description="Assigning 'ptr'"),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Null dereference"),
            ],
        )
        seeds = extract_seeds_from_bug_path(bug_path, sample_sdg)
        assert len(seeds) >= 1

    def test_multiple_events_multiple_seeds(self, sample_sdg):
        """Multiple path events → multiple seeds across the path."""
        bug_path = BugPath(
            path_id="t2", bug_type="NullDereference", category="Logic error",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=7,
                          description="Assigning 'ptr'"),
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="src/process.cc", line_number=8,
                          description="Taking true branch",
                          branch_condition=True),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Dereference of null pointer"),
            ],
        )
        seeds = extract_seeds_from_bug_path(bug_path, sample_sdg)
        # Should have seeds covering the key path nodes
        node_ids = {s[1] for s in seeds}
        assert len(node_ids) >= 2

    def test_dedup_on_same_node(self, sample_sdg):
        """Same PDG node from two events → deduplicated in seed set."""
        bug_path = BugPath(
            path_id="t3", bug_type="NullDereference", category="Logic error",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=7,
                          description="Assigning 'ptr'"),
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=7,
                          description="'ptr' points to null"),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        seeds = extract_seeds_from_bug_path(bug_path, sample_sdg)
        # Both events at line 7 should map to S_7_5 once
        node_ids = [s[1] for s in seeds]
        assert node_ids.count("S_7_5") <= 1

    def test_event_not_in_pdg(self, sample_sdg):
        """Event at line not covered by PDG → skipped."""
        bug_path = BugPath(
            path_id="t4", bug_type="NullDereference", category="Logic error",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=999,
                          description="Non-existent line"),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=999,
                          description="Bug"),
            ],
        )
        seeds = extract_seeds_from_bug_path(
            bug_path, sample_sdg,
            fallback_trigger_fn="MyClass::process",
            fallback_trigger_line=12,
        )
        # Fallback creates a seed at S_12_5 (within tolerance of fallback_trigger_line=12)
        assert len(seeds) == 1
        assert seeds[0][2] == "fallback_near_line"

    def test_all_event_kinds(self):
        """All three event kinds (event, control, report) produce seeds."""
        nodes = {
            "S_1_5": PDGNode(id="S_1_5", kind="assign", variable="x",
                              expression="x = 1", type="int",
                              source_line=1, source_col=5, block_id=1),
            "S_2_5": PDGNode(id="S_2_5", kind="if_cond", variable="",
                              expression="x > 0", type="bool",
                              source_line=2, source_col=5, block_id=2),
            "S_3_5": PDGNode(id="S_3_5", kind="call_expr", variable="",
                              expression="foo(x)", type="void",
                              source_line=3, source_col=5, block_id=3,
                              is_call=True, callee="foo"),
        }
        pdg = FunctionPDG(function_name="test_fn", source_file="t.cc",
                          nodes=nodes)
        sdg = SDG(pdgs={"test_fn": pdg})

        bug_path = BugPath(
            path_id="t5", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="t.cc", line_number=1,
                          description="Assigning 'x'"),
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="t.cc", line_number=2,
                          description="Taking true branch"),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="t.cc", line_number=3,
                          description="Calling 'foo'"),
            ],
        )
        seeds = extract_seeds_from_bug_path(bug_path, sdg)
        assert len(seeds) > 0

    def test_empty_bug_path(self, sample_sdg):
        """Empty bug path → empty seeds."""
        bug_path = BugPath(
            path_id="t6", bug_type="Test", category="test",
            events=[],
        )
        seeds = extract_seeds_from_bug_path(bug_path, sample_sdg)
        assert len(seeds) == 0

    def test_event_seed_reason_tagging(self):
        """Each seed gets a reason tag matching its event kind."""
        nodes = {
            "S_1_5": PDGNode(id="S_1_5", kind="assign", variable="x",
                              expression="x = 1", type="int",
                              source_line=1, source_col=5, block_id=1),
            "S_2_5": PDGNode(id="S_2_5", kind="if_cond", variable="",
                              expression="x > 0", type="bool",
                              source_line=2, source_col=5, block_id=2),
        }
        pdg = FunctionPDG(function_name="f", source_file="t.cc", nodes=nodes)
        sdg = SDG(pdgs={"f": pdg})

        bug_path = BugPath(
            path_id="t7", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="t.cc", line_number=1,
                          description="Assigning 'x' = 5"),
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="t.cc", line_number=2,
                          description="Taking true branch"),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="t.cc", line_number=1,
                          description="Bug trigger"),
            ],
        )
        seeds = extract_seeds_from_bug_path(bug_path, sdg)
        reasons = {s[2] for s in seeds if s[1] == "S_2_5"}
        # The control event should have reason = "path_control"
        assert "path_control" in reasons


# ===========================================================================
# Tests: BTBS path annotations
# ===========================================================================


class TestBuildPathAnnotations:
    """build_path_annotations() extracts branch decisions from bug paths."""

    def test_control_event_with_branch_condition(self, sample_sdg):
        """CONTROL event with branch_condition=True → taken annotation."""
        bug_path = BugPath(
            path_id="btbs1", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="src/process.cc", line_number=8,
                          description="Taking true branch",
                          branch_condition=True),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sample_sdg)
        assert annotations.get(("MyClass::process", 8)) is True

    def test_control_event_not_taken(self, sample_sdg):
        """CONTROL event with branch_condition=False → NOT-taken annotation."""
        bug_path = BugPath(
            path_id="btbs2", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="src/process.cc", line_number=8,
                          description="Taking false branch",
                          branch_condition=False),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sample_sdg)
        assert annotations.get(("MyClass::process", 8)) is False

    def test_event_kind_ignored(self, sample_sdg):
        """EVENT kind (not CONTROL) events are ignored by build_path_annotations."""
        bug_path = BugPath(
            path_id="btbs3", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=8,
                          description="Assuming condition is false",
                          branch_condition=False),  # EVENT with bc — ignored
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sample_sdg)
        assert len(annotations) == 0  # EVENT kind, not CONTROL

    def test_control_without_branch_condition_ignored(self, sample_sdg):
        """CONTROL event without branch_condition → not in annotations."""
        bug_path = BugPath(
            path_id="btbs4", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="src/process.cc", line_number=8,
                          description="Loop condition is true"),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sample_sdg)
        assert len(annotations) == 0

    def test_non_predicate_line_ignored(self, sample_sdg):
        """CONTROL event at a line without a predicate-kind PDG node → ignored."""
        bug_path = BugPath(
            path_id="btbs5", bug_type="Test", category="test",
            events=[
                # L9 is an assign node, not a control predicate
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="src/process.cc", line_number=9,
                          description="Taking true branch",
                          branch_condition=True),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sample_sdg)
        assert len(annotations) == 0

    def test_multiple_annotations(self):
        """Multiple CONTROL events produce multiple annotations."""
        nodes = {
            "L3_binop": PDGNode(id="L3_binop", kind="binary_op", variable="",
                                 expression="a > b", type="bool",
                                 source_line=3, source_col=5, block_id=2),
            "L5_if": PDGNode(id="L5_if", kind="if_cond", variable="",
                              expression="ptr == nullptr", type="bool",
                              source_line=5, source_col=5, block_id=3),
        }
        pdg = FunctionPDG(function_name="multi", source_file="multi.cc",
                          nodes=nodes)
        sdg = SDG(pdgs={"multi": pdg})
        bug_path = BugPath(
            path_id="btbs6", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="multi.cc", line_number=3,
                          description="Taking true branch",
                          branch_condition=True),
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="multi.cc", line_number=5,
                          description="Taking false branch",
                          branch_condition=False),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="multi.cc", line_number=10,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sdg)
        assert len(annotations) == 2
        assert annotations.get(("multi", 3)) is True
        assert annotations.get(("multi", 5)) is False

    def test_line_zero_ignored(self, sample_sdg):
        """CONTROL event at line 0 → ignored."""
        bug_path = BugPath(
            path_id="btbs7", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.CONTROL,
                          file_path="src/process.cc", line_number=0,
                          description="Taking true branch",
                          branch_condition=True),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sample_sdg)
        assert len(annotations) == 0

    def test_no_control_events(self, sample_sdg):
        """Bug path with no CONTROL events → empty annotations."""
        bug_path = BugPath(
            path_id="btbs8", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=5,
                          description="Assigning 'x'"),
                PathEvent(event_kind=EventKind.EVENT,
                          file_path="src/process.cc", line_number=7,
                          description="Assigning 'ptr'"),
                PathEvent(event_kind=EventKind.REPORT,
                          file_path="src/process.cc", line_number=9,
                          description="Bug"),
            ],
        )
        annotations = build_path_annotations(bug_path, sample_sdg)
        assert len(annotations) == 0


# ===========================================================================
# Tests: BTBS caller map (callee→caller reverse propagation)
# ===========================================================================


class TestBuildCallerMap:
    """build_caller_map() extracts the call chain from bug path events."""

    @pytest.fixture
    def two_fn_sdg(self) -> SDG:
        """SDG with two functions: caller 'start' and callee 'helper'."""
        return SDG(pdgs={
            "MyClass::start": FunctionPDG(
                function_name="MyClass::start",
                source_file="src/start.cc",
            ),
            "helper": FunctionPDG(
                function_name="helper",
                source_file="src/helper.cc",
            ),
        })

    def test_calling_event_creates_edge(self, two_fn_sdg):
        """'Calling helper' → caller_map['helper'] = 'start'."""
        bug_path = BugPath(
            path_id="cm1", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT, file_path="a.cc",
                          line_number=10, description="Calling 'helper'"),
                PathEvent(event_kind=EventKind.REPORT, file_path="a.cc",
                          line_number=20, description="Bug"),
            ],
        )
        caller_map = build_caller_map(bug_path, two_fn_sdg, entry_fn="MyClass::start")
        assert "helper" in caller_map
        assert caller_map["helper"] == "MyClass::start"

    def test_no_calling_events(self, two_fn_sdg):
        """Bug path with no calling events → empty map."""
        bug_path = BugPath(
            path_id="cm2", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.REPORT, file_path="a.cc",
                          line_number=10, description="Bug"),
            ],
        )
        caller_map = build_caller_map(bug_path, two_fn_sdg, entry_fn="start")
        assert len(caller_map) == 0

    def test_return_event_pops_stack(self):
        """'Returning from helper' pops the stack; later calling events
        resume from the original caller."""
        sdg = SDG(pdgs={
            "MyClass::start": FunctionPDG(function_name="MyClass::start",
                                          source_file="src/start.cc"),
            "helper": FunctionPDG(function_name="helper", source_file="src/helper.cc"),
            "other_fn": FunctionPDG(function_name="other_fn",
                                    source_file="src/other.cc"),
        })
        bug_path = BugPath(
            path_id="cm3", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT, file_path="a.cc",
                          line_number=5, description="Calling 'helper'"),
                PathEvent(event_kind=EventKind.EVENT, file_path="a.cc",
                          line_number=6, description="Returning from 'helper'"),
                PathEvent(event_kind=EventKind.EVENT, file_path="a.cc",
                          line_number=7, description="Calling 'other_fn'"),
                PathEvent(event_kind=EventKind.REPORT, file_path="a.cc",
                          line_number=10, description="Bug"),
            ],
        )
        caller_map = build_caller_map(bug_path, sdg, entry_fn="MyClass::start")
        assert "helper" in caller_map
        assert caller_map["helper"] == "MyClass::start"
        assert "other_fn" in caller_map
        assert caller_map["other_fn"] == "MyClass::start"

    def test_entry_fn_none_fallback(self, two_fn_sdg):
        """When entry_fn is None, the first calling event has no caller."""
        bug_path = BugPath(
            path_id="cm4", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT, file_path="a.cc",
                          line_number=5, description="Calling 'helper'"),
                PathEvent(event_kind=EventKind.REPORT, file_path="a.cc",
                          line_number=10, description="Bug"),
            ],
        )
        caller_map = build_caller_map(bug_path, two_fn_sdg, entry_fn=None)
        assert len(caller_map) == 0

    def test_name_resolution_qualified(self):
        """Calling event uses unqualified name; SDG has qualified name."""
        sdg = SDG(pdgs={
            "Namespace::compute": FunctionPDG(
                function_name="Namespace::compute",
                source_file="src/comp.cc",
            ),
        })
        bug_path = BugPath(
            path_id="cm5", bug_type="Test", category="test",
            events=[
                PathEvent(event_kind=EventKind.EVENT, file_path="a.cc",
                          line_number=3, description="Calling 'compute'"),
                PathEvent(event_kind=EventKind.EVENT, file_path="a.cc",
                          line_number=6, description="Returning from 'compute'"),
                PathEvent(event_kind=EventKind.REPORT, file_path="a.cc",
                          line_number=10, description="Bug"),
            ],
        )
        caller_map = build_caller_map(bug_path, sdg, entry_fn="main")
        assert "Namespace::compute" in caller_map
        assert caller_map["Namespace::compute"] == "main"

    def test_empty_bug_path(self, two_fn_sdg):
        """Empty bug path → empty caller_map."""
        bug_path = BugPath(
            path_id="cm6", bug_type="Test", category="test",
            events=[],
        )
        caller_map = build_caller_map(bug_path, two_fn_sdg, entry_fn="start")
        assert len(caller_map) == 0
