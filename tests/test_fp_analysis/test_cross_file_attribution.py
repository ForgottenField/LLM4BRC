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
    build_bug_report_path,
    segment_path,
)
from llm_client.fp_analysis.cfg_models import NodeType
from llm_client.fp_analysis.models import ParsedReport, ReportMetadata

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


def _make_report(tmp_path: Path, events: list[dict]) -> ParsedReport:
    """Materialize the synthetic sources and wrap *events* in a ParsedReport."""
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
