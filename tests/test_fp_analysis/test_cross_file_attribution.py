"""Cross-file bug paths: events, nodes and segments keep their own file.

A CSA path routinely starts in a header and continues through the .cpp that
the report's bug file belongs to (or vice versa).  Attribution used to
collapse every such event onto ``bug_file`` — line numbers were matched
against the SDG without their file half, and a line like 72 matches dozens of
files — which turned multi-file paths into single-function linear segments.
"""

from __future__ import annotations

from pathlib import Path

from llm_client.fp_analysis.cfg_feasibility import (
    _resolve_local_path,
    build_bug_report_path,
    segment_path,
)
from fp_analysis.cfg_models import NodeType
from llm_client.fp_analysis.models import ParsedReport, ReportMetadata, code_model_of
from fp_analysis.pdg_models import FunctionPDG, PDGNode, SDG
from report_frontend.code_model import SourceCodeModel

# Build-machine paths (as they appear in a report); resolved locally by
# suffix-matching against the synthetic source root created by the fixture.
CLONE_CPP = "/build/faiss/clone_index.cpp"
OTHER_CPP = "/build/faiss/other.cpp"
ALIGNED_H = "/build/faiss/utils/AlignedTable.h"

CLONE_SRC = """\
int clone_IndexIVF(int* p) {
    if (p) {
        return *p;
    }
    return 0;
}
"""

OTHER_SRC = """\
// same method name as in clone_index.cpp, different file
int clone_IndexIVF(int* q) {
    if (q) {
        return *q;
    }
    return 0;
}
"""

HEADER_SRC = """\
void nbytes() {
    if (x) {
    }
}
"""


def _event(num, line, file_path, kind="event", branch=None, desc=""):
    return {
        "path_number": num,
        "event_kind": kind,
        "description": desc or f"event {num}",
        "line_number": line,
        "branch_condition": branch,
        "file_path": file_path,
    }


def _make_report(
    tmp_path: Path, events: list[dict], path_evidence: str = "trace",
) -> ParsedReport:
    """Materialize the synthetic sources and wrap *events* in a ParsedReport.

    ``path_evidence`` defaults to ``"trace"`` because these reports stand in
    for an analyzer that walked a path — a CSA report always does, and the
    tests below turn on the CONTROL boundaries that path contributed.  The
    model makes that an explicit statement rather than something read off the
    event count, so a hand-built report has to make it too.
    """
    (tmp_path / "faiss" / "utils").mkdir(parents=True, exist_ok=True)
    (tmp_path / "faiss" / "clone_index.cpp").write_text(CLONE_SRC)
    (tmp_path / "faiss" / "other.cpp").write_text(OTHER_SRC)
    (tmp_path / "faiss" / "utils" / "AlignedTable.h").write_text(HEADER_SRC)
    return ParsedReport(
        metadata=ReportMetadata(
            bug_type="Argument with 'nonnull' attribute passed null",
            bug_file=ALIGNED_H,
            bug_line=12,
            function_name="nbytes",
            path_evidence=path_evidence,
        ),
        path_events=events,
        source_snippets={},
        raw_html_path=str(tmp_path / "report.html"),
    )


class TestPathNodeAttribution:
    def test_each_node_keeps_its_own_file(self, tmp_path: Path) -> None:
        parsed = _make_report(tmp_path, [
            _event(1, 3, CLONE_CPP, kind="control", branch=True),
            _event(9999, 2, ALIGNED_H, desc="Null pointer"),
        ])
        path = build_bug_report_path(parsed, source_root=str(tmp_path))

        assert path.nodes[0].source_file.endswith("faiss/clone_index.cpp")
        assert path.nodes[1].source_file.endswith("utils/AlignedTable.h")
        # ... and each node is attributed to the function of ITS file.
        assert path.nodes[0].function_name == "clone_IndexIVF"
        assert path.nodes[1].function_name == "nbytes"

    def test_node_without_report_file_falls_back_to_bug_file(
        self, tmp_path: Path
    ) -> None:
        parsed = _make_report(tmp_path, [
            _event(9999, 2, "", desc="Null pointer"),
        ])
        path = build_bug_report_path(parsed, source_root=str(tmp_path))
        assert path.nodes[0].source_file.endswith("utils/AlignedTable.h")


class TestEventFileResolution:
    """The resolver that decides an event's file, and what it must refuse.

    Every attribution below it starts here: ``build_bug_report_path`` fills
    ``event_files[idx]`` from this call, and only falls back to the line-number
    guess — which picks whatever file shares the line — when it returns
    ``None``.  So the two failure modes are opposite halves of one contract:
    too narrow and a whole format's events lose their file, too eager and an
    event is confidently attributed to the wrong one.
    """

    def test_a_relative_report_path_is_placed(self, tmp_path: Path) -> None:
        # The spelling a SARIF (CodeQL/Infer/Cppcheck) ``uri`` always has: the
        # path is relative to the analysis root, so nothing can be trimmed off
        # its front.  The resolver used to walk suffixes only, tried
        # ``<root>/AlignedTable.h``, and gave up — the file is two levels down.
        _make_report(tmp_path, [])
        hit = _resolve_local_path("faiss/utils/AlignedTable.h", str(tmp_path))
        assert hit == (tmp_path / "faiss/utils/AlignedTable.h").resolve()

    def test_a_leading_dot_slash_is_placed(self, tmp_path: Path) -> None:
        _make_report(tmp_path, [])
        assert _resolve_local_path("./faiss/clone_index.cpp", str(tmp_path)) == (
            tmp_path / "faiss/clone_index.cpp").resolve()

    def test_a_build_machine_path_is_still_placed(self, tmp_path: Path) -> None:
        # The CSA spelling, and the behaviour that must survive: an absolute
        # build path whose tree root differs from ours.
        _make_report(tmp_path, [])
        hit = _resolve_local_path("/build/faiss/clone_index.cpp", str(tmp_path))
        assert hit == (tmp_path / "faiss/clone_index.cpp").resolve()

    def test_an_absolute_local_path_is_direct(self, tmp_path: Path) -> None:
        _make_report(tmp_path, [])
        target = tmp_path / "faiss/clone_index.cpp"
        assert _resolve_local_path(target, str(tmp_path)) == target.resolve()

    def test_a_name_only_match_is_refused(self, tmp_path: Path) -> None:
        # ``core.h`` exists in the tree, but that is not evidence that this
        # *build machine's* ``/usr/include/fmt/core.h`` is it — a different
        # directory, a different library.  Returning it would silently
        # attribute the event's function to that file; returning None sends it
        # through the line-index fallback, which is restricted to the files the
        # report itself mentions.
        (tmp_path / "lib/plugins/core").mkdir(parents=True)
        (tmp_path / "lib/plugins/core/core.h").write_text("// unrelated\n")
        assert _resolve_local_path("/usr/include/fmt/core.h", str(tmp_path)) is None

    def test_nothing_is_invented_without_a_source_root(self, tmp_path: Path) -> None:
        # No root means no ladder above ``direct``: a relative path is not
        # resolved against the process's working directory.
        _make_report(tmp_path, [])
        assert _resolve_local_path("faiss/clone_index.cpp", None) is None

    def test_the_events_file_reaches_the_path(self, tmp_path: Path) -> None:
        # End to end through the attribution the resolver feeds: a relative
        # event file must keep its own file *and* its own function, instead of
        # falling back to the bug file.
        parsed = _make_report(tmp_path, [
            _event(1, 3, "faiss/clone_index.cpp", kind="control", branch=True),
            _event(9999, 2, "faiss/utils/AlignedTable.h", desc="Null pointer"),
        ])
        path = build_bug_report_path(parsed, source_root=str(tmp_path))

        assert path.nodes[0].source_file.endswith("faiss/clone_index.cpp")
        assert path.nodes[0].function_name == "clone_IndexIVF"
        assert path.nodes[1].function_name == "nbytes"


class TestSegmentAttribution:
    def test_segment_carries_its_function_file(self, tmp_path: Path) -> None:
        parsed = _make_report(tmp_path, [
            _event(1, 3, CLONE_CPP, kind="control", branch=True,
                   desc="Taking true branch"),
            _event(9999, 2, ALIGNED_H, desc="Null pointer"),
        ])
        segments = segment_path(parsed, source_root=str(tmp_path))

        assert len(segments) == 2
        assert segments[0].function_name == "clone_IndexIVF"
        assert segments[0].function_file.endswith("faiss/clone_index.cpp")
        assert segments[1].function_name == "nbytes"
        assert segments[1].function_file.endswith("utils/AlignedTable.h")

    def test_same_function_name_in_two_files_keeps_ranges_apart(
        self, tmp_path: Path
    ) -> None:
        """A name-only key would resume the second file's function from the
        first file's line, because both are called ``clone_IndexIVF``."""
        parsed = _make_report(tmp_path, [
            _event(1, 3, CLONE_CPP, kind="control", branch=True),
            _event(2, 3, OTHER_CPP, kind="control", branch=False),
        ])
        segments = segment_path(parsed, source_root=str(tmp_path))

        assert [s.function_name for s in segments] == ["clone_IndexIVF"] * 2
        assert segments[0].function_file.endswith("clone_index.cpp")
        assert segments[1].function_file.endswith("other.cpp")
        # other.cpp's clone_IndexIVF starts at line 2; resuming from the other
        # file's line 3 would be the name-only-key bug.
        assert segments[1].line_start == 2

    def test_control_node_type_is_derived(self, tmp_path: Path) -> None:
        parsed = _make_report(tmp_path, [
            _event(1, 3, CLONE_CPP, kind="control", branch=True),
        ])
        path = build_bug_report_path(parsed, source_root=str(tmp_path))
        assert path.nodes[0].node_type == NodeType.CONTROL


class TestLocationOnlyReports:
    """A report that states *where* rather than *how* is segmented at the where.

    Most CodeQL results are this (127 of 176 in the real corpus): a location and
    a claim, no ``codeFlow``.  The finding still renders one step — the report
    step — so nothing downstream can tell it from a one-step path by counting;
    ``ReportMetadata.path_evidence`` is the analyzer's own statement, and the
    pipeline segments the reported location instead of inventing a path.
    """

    def test_the_reported_function_is_the_analysis_unit(self, tmp_path: Path) -> None:
        # No CONTROL/CALL steps at all: the reported location's function is the
        # one segment, which is what the path-selection stages then work on.
        parsed = _make_report(tmp_path, [
            _event(9999, 2, ALIGNED_H, desc="Null pointer"),
        ], path_evidence="location")
        segments = segment_path(parsed, source_root=str(tmp_path))

        assert len(segments) == 1
        assert segments[0].function_name == "nbytes"
        assert segments[0].function_file.endswith("utils/AlignedTable.h")

    def test_a_path_shaped_rendering_is_not_segmented_as_a_path(
        self, tmp_path: Path
    ) -> None:
        # The contradiction case: the model says "no path" while the events
        # carry boundaries.  The model is authoritative (it is what the analyzer
        # stated), so the boundaries do not become segments — and the mismatch
        # is a warning rather than a silent choice.
        parsed = _make_report(tmp_path, [
            _event(1, 3, CLONE_CPP, kind="control", branch=True),
            _event(9999, 2, ALIGNED_H, desc="Null pointer"),
        ], path_evidence="location")
        segments = segment_path(parsed, source_root=str(tmp_path))

        assert len(segments) == 1
        assert segments[0].function_name == "nbytes"

    def test_the_default_is_the_weaker_reading(self) -> None:
        # A report that does not say must not be handed a path.
        meta = ReportMetadata(bug_type="t", bug_file="f.cc", bug_line=1)
        assert meta.path_evidence == "location" and not meta.has_path


class TestTheSdgReachesTheCodeModel:
    """``build_bug_report_path`` is the one place holding the SDG.

    Where a function is defined is a fact the PDG records and a brace scan only
    guesses at, so the pipeline's "give me this function's body" must be
    answered from the PDG.  The index used to be built here and then used here
    only — every other consumer, the code model included, kept guessing.
    """

    @staticmethod
    def _sdg(function: str, file: str, lines: list[int]) -> SDG:
        return SDG(pdgs={function: FunctionPDG(
            function_name=function, source_file=file,
            nodes={
                f"S_{n}_1": PDGNode(id=f"S_{n}_1", source_line=n) for n in lines
            },
        )})

    def test_the_model_answers_from_the_pdg_index(self, tmp_path: Path) -> None:
        parsed = _make_report(tmp_path, [
            _event(9999, 2, ALIGNED_H, desc="Null pointer"),
        ])
        model = SourceCodeModel(source_root=tmp_path)
        parsed.code_model = model
        # The PDG says ``nbytes`` spans lines 3-4 of the header; the brace scan
        # would say 2-5 (its signature through its closing brace).  A body that
        # starts where the PDG's first node is can only have come from the SDG.
        build_bug_report_path(
            parsed,
            sdg=self._sdg("nbytes", ALIGNED_H, [3, 4]),
            source_root=str(tmp_path),
        )

        hit = model.function_body("nbytes")
        assert hit is not None
        local, start, end, text = hit
        assert local == tmp_path / "faiss" / "utils" / "AlignedTable.h"
        assert (start, end) == (3, 4)
        assert text == "\n".join(HEADER_SRC.splitlines()[2:4])

    def test_without_the_sdg_it_is_still_a_brace_scan(self, tmp_path: Path) -> None:
        # The same lookup with no PDG artifact — bitcoin, or a project whose
        # graph has not been built — must still answer, from the tree.
        parsed = _make_report(tmp_path, [
            _event(9999, 2, ALIGNED_H, desc="Null pointer"),
        ])
        model = SourceCodeModel(source_root=tmp_path)
        parsed.code_model = model
        build_bug_report_path(parsed, source_root=str(tmp_path))

        hit = model.function_body("nbytes")
        # Signature line through closing brace, as the brace scan has it.
        assert hit is not None and hit[1:3] == (1, 4)

    def test_the_shared_model_refuses_a_reports_index(self, tmp_path: Path) -> None:
        # A hand-built report falls back to the process-wide model.  One
        # report's SDG must not become the next report's answer, so the write
        # is refused (loudly) rather than kept.
        parsed = _make_report(tmp_path, [
            _event(9999, 2, ALIGNED_H, desc="Null pointer"),
        ])
        build_bug_report_path(
            parsed,
            sdg=self._sdg("nbytes", ALIGNED_H, [3, 4]),
            source_root=str(tmp_path),
        )

        shared = code_model_of(parsed, str(tmp_path))
        hit = shared.function_body("nbytes")
        # What the tree says (1-4), not what this report's SDG said (3-4).
        assert hit is not None and hit[1:3] == (1, 4)
