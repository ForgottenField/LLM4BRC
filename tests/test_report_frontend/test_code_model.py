"""The code model: report paths → local files, with the strategy recorded.

This is the half of "a model of the report *and its related code*" that has no
equivalent in the old parser.  Two properties carry the weight:

* **The strategy is recorded.**  ``basename`` means "a file of that name exists
  somewhere in the tree", which is a very different claim from ``direct``, and
  the dependency builder refuses the weak one.  A resolver that returns a path
  without saying how it found it cannot support that check.
* **Ambiguity is surfaced, not decided silently.**  ``index_read.cpp`` exists
  under both ``faiss/impl/`` and ``faiss/invlists/`` in these corpora, and
  ``rglob(name)[0]`` — the code this replaces — returned whichever the
  filesystem yielded first.

The tests build a tree of a known shape so the *strategy* can be asserted;
asserting it against a real corpus would only ever say "something matched".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_client.report_frontend import code_model
from llm_client.report_frontend.code_model import (
    STRATEGY_BASENAME,
    STRATEGY_BUILD_PREFIX,
    STRATEGY_DIRECT,
    STRATEGY_EMBEDDED,
    STRATEGY_RELATIVE,
    STRATEGY_SUFFIX,
    STRATEGY_UNRESOLVED,
    STRONG_STRATEGIES,
    SourceCodeModel,
    clear_index_cache,
)
from llm_client.report_frontend.models import (
    AnalyzerKind,
    BugFinding,
    SourceLocation,
    StepKind,
    TraceStep,
)

BUILD_ROOT = "/home/me/projects/cpp-project/"


@pytest.fixture(autouse=True)
def _fresh_index():
    # The tree index is cached per root for the process; a stale entry from
    # another test's tmp_path would make these assertions meaningless.
    clear_index_cache()
    yield
    clear_index_cache()


@pytest.fixture
def model(source_tree: Path) -> SourceCodeModel:
    return SourceCodeModel(source_root=source_tree)


class TestResolutionLadder:
    def test_direct_absolute_path(self, source_tree: Path, model) -> None:
        target = source_tree / "faiss" / "utils" / "AlignedTable.h"
        hit = model.resolve(str(target))
        assert hit.resolved and hit.local == target.resolve()
        assert hit.strategy == STRATEGY_DIRECT

    def test_build_prefix_is_stripped(self, model) -> None:
        hit = model.resolve(
            BUILD_ROOT + "faiss/faiss/utils/AlignedTable.h"
        )
        assert hit.strategy == STRATEGY_BUILD_PREFIX
        assert hit.local.name == "AlignedTable.h"
        assert hit.local.parent.name == "utils"

    def test_relative_path_resolves_under_the_root(self, model) -> None:
        hit = model.resolve("faiss/utils/AlignedTable.h")
        assert hit.strategy == STRATEGY_RELATIVE
        assert hit.local.name == "AlignedTable.h"

    def test_uri_base_is_tried_for_relative_paths(self, source_tree: Path) -> None:
        # A SARIF artifact may be relative to ``%SRCROOT%`` rather than to the
        # analysing root; ``uri_bases`` is how the frontend passes that in.
        model = SourceCodeModel(
            source_root=source_tree, uri_bases=[str(source_tree)],
        )
        hit = model.resolve("faiss/utils/AlignedTable.h")
        assert hit.resolved
        assert hit.strategy in (STRATEGY_RELATIVE, STRATEGY_SUFFIX)

    def test_longest_suffix_pins_a_foreign_prefix(self, model) -> None:
        hit = model.resolve("/some/other/machine/checkout/faiss/utils/AlignedTable.h")
        assert hit.strategy == STRATEGY_SUFFIX
        assert hit.local.name == "AlignedTable.h"

    def test_basename_is_the_last_resort_and_says_so(self, model) -> None:
        hit = model.resolve("/nowhere/at/all/AlignedTable.h")
        assert hit.strategy == STRATEGY_BASENAME
        assert hit.local.name == "AlignedTable.h"

    def test_basename_match_reports_its_rivals(self, model) -> None:
        # Two files share this name in the fixture tree — the exact trap
        # ``rglob(name)[0]`` fell into.  The model picks one *and* records both.
        hit = model.resolve("/nowhere/at/all/index_read.cpp")
        assert hit.strategy == STRATEGY_BASENAME
        assert hit.is_ambiguous
        assert len(hit.candidates) >= 2

    def test_the_chosen_file_is_not_one_of_its_own_rivals(self, model) -> None:
        # ``candidates`` carries the pick first, so anything *reporting* the
        # ambiguity has to drop it — the message used to read
        # ``X → X (also: X, Y)``, which names the chosen file as its own rival.
        hit = model.resolve("/nowhere/at/all/index_read.cpp")
        assert hit.candidates[0] == hit.local
        assert hit.local not in hit.rivals
        assert len(hit.rivals) == len(hit.candidates) - 1
        assert str(hit.local) not in hit.to_dict()["ambiguous_with"]

    def test_an_unambiguous_resolution_has_no_rivals(self, model) -> None:
        hit = model.resolve("faiss/impl/index_read.cpp")
        assert hit.strategy != STRATEGY_BASENAME
        assert hit.rivals == ()
        assert not hit.is_ambiguous

    def test_a_suffix_match_is_preferred_over_an_ambiguous_basename(
        self, model
    ) -> None:
        # ``faiss/invlists/index_read.cpp`` names one of the two unambiguously;
        # falling back to basename here would be a coin flip.
        hit = model.resolve("/other/tree/faiss/invlists/index_read.cpp")
        assert hit.strategy == STRATEGY_SUFFIX
        assert "invlists" in str(hit.local)

    def test_unresolved_is_reported_not_guessed(self, model) -> None:
        hit = model.resolve("/nowhere/definitely_missing.cpp")
        assert not hit.resolved
        assert hit.strategy == STRATEGY_UNRESOLVED
        assert hit.local is None

    def test_empty_path(self, model) -> None:
        assert not model.resolve("").resolved

    def test_embedded_only_when_the_tree_has_nothing(
        self, source_tree: Path, tmp_path: Path
    ) -> None:
        model = SourceCodeModel(source_root=source_tree)
        model.add_embedded("/buildonly/header.h", {1: "// embedded"})
        hit = model.resolve("/buildonly/header.h")
        assert hit.strategy == STRATEGY_EMBEDDED
        # ``resolved`` means "there is a local file" — which is what the PDG,
        # the CFG cache and the dependency builder all need.  An embedded-only
        # file is *not* that, however readable and however useful the snippet
        # is; conflating the two is how a report analysed against a tree it does
        # not belong to stops being noticeable.
        assert not hit.resolved
        assert hit.strategy not in STRONG_STRATEGIES
        assert model.snippet("/buildonly/header.h", 1, radius=0) == {1: "// embedded"}

    def test_resolution_is_memoised(self, model) -> None:
        first = model.resolve(BUILD_ROOT + "faiss/faiss/utils/AlignedTable.h")
        second = model.resolve(BUILD_ROOT + "faiss/faiss/utils/AlignedTable.h")
        assert first is second

    def test_no_source_root(self) -> None:
        model = SourceCodeModel(source_root=None)
        assert not model.resolve("faiss/impl/index_read.cpp").resolved


class TestStrongStrategies:
    def test_basename_is_not_strong(self) -> None:
        assert STRATEGY_BASENAME not in STRONG_STRATEGIES
        assert STRATEGY_EMBEDDED not in STRONG_STRATEGIES
        assert STRATEGY_UNRESOLVED not in STRONG_STRATEGIES

    def test_positional_strategies_are_strong(self) -> None:
        for strategy in (STRATEGY_DIRECT, STRATEGY_BUILD_PREFIX,
                         STRATEGY_RELATIVE, STRATEGY_SUFFIX):
            assert strategy in STRONG_STRATEGIES


class TestReading:
    def test_snippet_reads_the_local_file(self, model, source_tree: Path) -> None:
        lines = model.snippet("faiss/utils/AlignedTable.h", 10, radius=2)
        assert sorted(lines) == [8, 9, 10, 11, 12]
        assert "aligned 10" in lines[10]

    def test_snippet_prefers_the_embedded_copy(self, model) -> None:
        model.add_embedded(
            "faiss/utils/AlignedTable.h", {10: "// from the report"}
        )
        lines = model.snippet("faiss/utils/AlignedTable.h", 10, radius=1)
        assert lines[10] == "// from the report"

    def test_snippet_clamps_at_the_file_boundaries(self, model) -> None:
        lines = model.snippet("faiss/utils/AlignedTable.h", 1, radius=30)
        assert min(lines) == 1
        assert max(lines) <= 70

    def test_snippet_is_empty_for_an_unresolvable_file(self, model) -> None:
        assert model.snippet("/nowhere/nope.cpp", 5) == {}

    def test_read_range(self, model) -> None:
        local = model.resolve("faiss/utils/AlignedTable.h").local
        assert sorted(model.read_range(local, 3, 5)) == [3, 4, 5]


class TestReferences:
    @pytest.fixture
    def finding(self, source_tree: Path) -> BugFinding:
        return BugFinding(
            finding_id="csa:index_read.cpp:12",
            analyzer=AnalyzerKind.CSA,
            rule_id="Memory leak",
            message="m",
            location=SourceLocation(
                file=str(source_tree / "faiss" / "impl" / "index_read.cpp"),
                line=12,
            ),
            steps=[
                TraceStep(index=1, kind=StepKind.CALL,
                          location=SourceLocation(
                              file=str(source_tree / "faiss" / "utils"
                                         / "AlignedTable.h"),
                              line=66),
                          message="Calling 'helper'", callee="helper"),
                TraceStep(index=2, kind=StepKind.REPORT,
                          location=SourceLocation(
                              file=str(source_tree / "faiss" / "impl"
                                         / "index_read.cpp"),
                              line=12),
                          message="m"),
            ],
        )

    def test_roles_are_labelled(self, model, finding: BugFinding) -> None:
        roles = {r.role for r in model.references(finding)}
        assert "bug_file" in roles
        # A CALL step's file is the *callee's* file — a different claim from
        # "the path passed through here", and the one the callee-source
        # extraction (F2) keys on.
        assert "call_target" in roles

    def test_references_are_deduplicated_by_path_and_role(
        self, model, finding: BugFinding
    ) -> None:
        refs = model.references(finding)
        keys = [(r.reported, r.role) for r in refs]
        assert len(keys) == len(set(keys))

    def test_summary_counts_and_names_the_gaps(
        self, model, finding: BugFinding
    ) -> None:
        summary = model.summary(finding)
        assert summary["reported_files"] >= 2
        assert summary["resolved_files"] >= 2
        assert summary["unresolved"] == []
        assert "references" in summary

    def test_unresolved_lists_only_the_missing(
        self, model, source_tree: Path
    ) -> None:
        finding = BugFinding(
            finding_id="x", analyzer=AnalyzerKind.CODEQL,
            location=SourceLocation(file="/gone/missing.cpp", line=1),
        )
        assert [r.reported for r in model.unresolved(finding)] == ["/gone/missing.cpp"]


class TestMaterialize:
    """The report's index → the code, read once, *into* the model.

    The pipeline's rule is that the model is the one reader of project source.
    That is only true if the code is actually in it, so ``load`` materialises
    what the report names — whole files, because a window is a guess about what
    the reader will want next.
    """

    @pytest.fixture
    def finding(self, source_tree: Path) -> BugFinding:
        """A finding naming three files: bug, one path event, one call target."""
        return BugFinding(
            finding_id="csa:materialize",
            analyzer=AnalyzerKind.CSA,
            rule_id="Memory leak",
            message="m",
            location=SourceLocation(
                file=str(source_tree / "faiss" / "impl" / "index_read.cpp"),
                line=12,
            ),
            steps=[
                TraceStep(index=1, kind=StepKind.CALL,
                          location=SourceLocation(
                              file=str(source_tree / "faiss" / "utils"
                                         / "AlignedTable.h"),
                              line=66),
                          message="Calling 'helper'", callee="helper"),
                TraceStep(index=2, kind=StepKind.EVENT,
                          location=SourceLocation(
                              file=str(source_tree / "faiss" / "invlists"
                                         / "index_read.cpp"),
                              line=4),
                          message="Assuming the condition is true"),
                TraceStep(index=3, kind=StepKind.REPORT,
                          location=SourceLocation(
                              file=str(source_tree / "faiss" / "impl"
                                         / "index_read.cpp"),
                              line=12),
                          message="m"),
            ],
        )

    def test_nothing_is_materialised_before_the_step(self, model) -> None:
        assert model.code_set is None

    def test_the_files_the_report_names_are_read_whole(
        self, model, finding: BugFinding, source_tree: Path
    ) -> None:
        info = model.materialize(finding)
        bug = source_tree / "faiss" / "impl" / "index_read.cpp"
        record = next(r for r in info["files"] if r["local"] == str(bug))
        assert record["source"] == "tree"
        assert record["strategy"] == STRATEGY_DIRECT
        # Whole file, not the bug line's window: every line is answerable, and
        # the byte count is the file's, so the inventory measures the real cost.
        assert record["lines"] == len(bug.read_text().splitlines())
        assert record["bytes"] == len(bug.read_bytes())
        assert len(model.lines(bug)) == record["lines"]

    def test_every_named_file_is_covered(
        self, model, finding: BugFinding, source_tree: Path
    ) -> None:
        info = model.materialize(finding)
        locals_ = {r["local"] for r in info["files"]}
        assert str(source_tree / "faiss" / "utils" / "AlignedTable.h") in locals_
        assert str(source_tree / "faiss" / "invlists" / "index_read.cpp") in locals_
        assert info["unresolved"] == []

    def test_one_record_per_reported_path(
        self, model, finding: BugFinding
    ) -> None:
        # Two events on the same file are one file to read; the record is the
        # index's unit (the reported path), not the step's.
        info = model.materialize(finding)
        reported = [r["reported"] for r in info["files"]]
        assert len(reported) == len(set(reported))
        assert info["reported_paths"] == len(reported)

    def test_a_path_only_the_report_carries_is_not_a_gap(self, model) -> None:
        # No local file, but the report embeds that file's source: readable
        # without the tree, so it is served rather than reported missing.
        model.add_embedded("/gone/remote.cc", {1: "// from the report"})
        finding = BugFinding(
            finding_id="x", analyzer=AnalyzerKind.CSA, rule_id="r", message="m",
            location=SourceLocation(file="/gone/remote.cc", line=1),
        )
        info = model.materialize(finding)
        assert info["unresolved"] == []
        assert info["files"][0]["source"] == STRATEGY_EMBEDDED
        assert info["files"][0]["local"] is None
        assert info["files"][0]["lines"] == 1

    def test_a_named_file_with_nothing_to_read_is_listed(self, model) -> None:
        finding = BugFinding(
            finding_id="x", analyzer=AnalyzerKind.CSA, rule_id="r", message="m",
            location=SourceLocation(file="/nowhere/definitely_missing.cc", line=3),
        )
        info = model.materialize(finding)
        assert info["unresolved"] == [{
            "reported": "/nowhere/definitely_missing.cc",
            "strategy": STRATEGY_UNRESOLVED,
        }]
        assert info["files"] == []

    def test_a_same_stem_sibling_is_read_and_a_missing_one_is_not_invented(
        self, model, finding: BugFinding, source_tree: Path
    ) -> None:
        sibling = source_tree / "faiss" / "impl" / "index_read.h"
        sibling.write_text("// declared here\n// and here\n// and here\n")
        info = model.materialize(finding)
        companions = {c["local"] for c in info["companions"]}
        assert str(sibling) in companions
        # ``X-inl.h`` is not beside the file, so it is not named: the suffix
        # exists only to look for a sibling, never to spell one into being.
        assert not any("index_read-inl.h" in c for c in companions)
        assert len(model.lines(sibling)) == 3

    def test_a_companion_read_is_not_a_lazy_read(
        self, model, finding: BugFinding, source_tree: Path
    ) -> None:
        (source_tree / "faiss" / "impl" / "index_read.h").write_text("// h\n")
        model.materialize(finding)
        model.lines(source_tree / "faiss" / "impl" / "index_read.h")
        assert model.lazy_reads == []

    def test_a_read_the_pre_load_missed_is_counted(
        self, model, finding: BugFinding, source_tree: Path
    ) -> None:
        model.materialize(finding)
        assert model.lazy_reads == []
        other = source_tree / "CMakeLists.txt"   # real, but not the report's
        model.lines(other)
        assert model.lazy_reads == [str(other)]
        model.lines(other)   # cached — the same file is not counted twice
        assert len(model.lazy_reads) == 1
        assert model.lazy_reads_truncated == 0

    def test_the_cap_counts_what_it_drops(
        self, model, finding: BugFinding, monkeypatch
    ) -> None:
        monkeypatch.setattr(code_model, "_MAX_CODE_SET_RECORDS", 1)
        info = model.materialize(finding)
        assert len(info["files"]) == 1
        assert info["files_truncated"] == 2   # 3 reported paths, 1 kept
        assert info["reported_paths"] == 3

    def test_the_inventory_is_visible_and_serialisable(
        self, model, finding: BugFinding
    ) -> None:
        info = model.materialize(finding)
        assert model.code_set == info
        assert json.loads(json.dumps(info))["bytes"] == info["bytes"] > 0


class TestDrift:
    """The report's own copy vs the local tree: recorded, deduped, capped.

    The report is authoritative (it is the revision the analyzer ran on, and its
    line numbers are what every stage means by "line N"), so a disagreement is a
    fact about the *checkout*.  Nothing here may influence a verdict — the
    record exists so a stale tree is visible instead of inferred.
    """

    def test_a_disagreement_is_recorded_with_both_copies(self, model) -> None:
        model.add_embedded("faiss/utils/AlignedTable.h", {10: "// from the report"})
        model.snippet("faiss/utils/AlignedTable.h", 10, radius=1)
        assert model.drift == [{
            "reported": "faiss/utils/AlignedTable.h",
            "line": 10,
            "embedded": "// from the report",
            "local": "// aligned 10",
            # How the local file was found is part of the record: a
            # disagreement against a name-only match may be a different file of
            # that name rather than a moved checkout.  ``relative`` is a strong
            # rung, so this one is a moved checkout.
            "strategy": STRATEGY_RELATIVE,
        }]

    def test_agreeing_lines_are_not_drift(self, model) -> None:
        # The embedded window also covers lines 9 and 11, which the tree has
        # exactly; only the edited line is a disagreement.
        model.add_embedded(
            "faiss/utils/AlignedTable.h",
            {9: "// aligned 9", 10: "// from the report", 11: "// aligned 11"},
        )
        model.snippet("faiss/utils/AlignedTable.h", 10, radius=1)
        assert [r["line"] for r in model.drift] == [10]

    def test_each_line_is_recorded_once(self, model) -> None:
        # ``snippet()`` is called once per rendered block and once per path step
        # in the simulation; without the dedupe the same line lands in the
        # result JSON hundreds of times.
        model.add_embedded("faiss/utils/AlignedTable.h", {10: "// from the report"})
        for _ in range(4):
            model.snippet("faiss/utils/AlignedTable.h", 10, radius=1)
        assert len(model.drift) == 1
        assert model.drift_truncated == 0

    def test_nothing_is_recorded_without_a_local_file(self, model) -> None:
        model.add_embedded("/elsewhere/gone.h", {10: "// from the report"})
        model.snippet("/elsewhere/gone.h", 10, radius=1)
        assert model.drift == []

    def test_the_cap_counts_what_it_drops(self, model, monkeypatch) -> None:
        monkeypatch.setattr(code_model, "_MAX_DRIFT_RECORDS", 3)
        model.add_embedded(
            "faiss/utils/AlignedTable.h",
            {n: f"// report {n}" for n in range(1, 11)},
        )
        model.snippet("faiss/utils/AlignedTable.h", 5, radius=4)
        assert len(model.drift) == 3
        assert model.drift_truncated == 6  # 9 compared lines - 3 kept
