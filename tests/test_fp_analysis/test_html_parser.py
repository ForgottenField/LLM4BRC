"""Tests for the FP HTML report parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_client.fp_analysis.html_parser import FPReportParser
from llm_client.fp_analysis.models import ParsedReport


class TestFPReportParser:
    """Test parsing of CSA HTML reports."""

    def setup_method(self) -> None:
        self.parser = FPReportParser()

    def test_parse_metadata(self, memory_test_report: str) -> None:
        """Test that metadata is correctly extracted."""
        parsed = self.parser.parse(memory_test_report)
        assert parsed.metadata.bug_type == "Dereference of null pointer"
        assert parsed.metadata.function_name == "Destroy"
        assert parsed.metadata.bug_line == 407
        assert parsed.metadata.path_length == 25

    def test_parse_path_events(self, memory_test_report: str) -> None:
        """Test that path events are extracted and ordered."""
        parsed = self.parser.parse(memory_test_report)
        assert len(parsed.path_events) > 0
        # First event should be #1
        assert parsed.path_events[0]["path_number"] == 1
        # Last event should be the report
        assert parsed.path_events[-1]["event_kind"] == "report"
        # Events should be in order
        numbers = [e["path_number"] for e in parsed.path_events]
        assert numbers == sorted(numbers)

    def test_parse_path_events_content(self, memory_test_report: str) -> None:
        """Test that path event descriptions are meaningful."""
        parsed = self.parser.parse(memory_test_report)
        # Control events should have descriptions
        control_events = [e for e in parsed.path_events if e["event_kind"] == "control"]
        assert len(control_events) > 0
        for evt in control_events:
            assert len(evt["description"]) > 5

    def test_parse_source_lines(self, memory_test_report: str) -> None:
        """Test that source code lines are extracted."""
        parsed = self.parser.parse(memory_test_report)
        assert len(parsed.source_snippets) > 0
        # Should include lines around the bug
        assert parsed.metadata.bug_line in parsed.source_snippets

    def test_parse_to_bug_path(self, memory_test_report: str) -> None:
        """Test conversion to BugPath."""
        bp = self.parser.parse_to_bug_path(memory_test_report)
        assert bp.bug_type == "Dereference of null pointer"
        assert len(bp.events) > 0
        # Last event should be report type
        assert bp.events[-1].event_kind.value == "report"

    def test_missing_file(self) -> None:
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/report.html")

    def test_resolve_source_path(self) -> None:
        """Test source file path resolution."""
        from llm_client.fp_analysis.html_parser import resolve_report_source_file

        # Non-existent path with known prefix
        result = resolve_report_source_file(
            "/tmp/fbcode_builder/installed/include/gtest/gtest-matchers.h",
            project_root="project/folly",
        )
        # The file may or may not exist — just check it doesn't crash
        assert result is None or isinstance(result, Path)

    def test_html_clean_text(self) -> None:
        """Test HTML entity cleaning."""
        cleaned = self.parser._clean_html_text(
            '1 &#x2190; Calling &lt;T&gt; &amp; foo &#x2192;'
        )
        assert '&#x2190;' not in cleaned
        assert '&#x2192;' not in cleaned
        assert '&lt;' not in cleaned
        assert '&amp;' not in cleaned
        assert '<T>' in cleaned or 'T>' in cleaned
        # Leading number should be stripped
        assert not cleaned.startswith("1 ")

    def test_socket_address_report(self, socket_address_report: str) -> None:
        """Test parsing of a report where the bug is in folly's own code."""
        parsed = self.parser.parse(socket_address_report)
        assert parsed.metadata.bug_line == 660
        assert parsed.metadata.bug_type == "Dereference of null pointer"
        assert "SocketAddress" in parsed.metadata.bug_file
        assert len(parsed.path_events) > 0


# ---------------------------------------------------------------------------
# Cross-file attribution
# ---------------------------------------------------------------------------


FAISS_REPORT = Path(
    "/home/yanghengqin/csa_reports/reports/faiss/sf-off"
    "/report-clone_index.cpp-operator=-2-1.html"
)


def _build_report(
    sections: list[tuple[str, list[str]]],
    events: list[tuple[int, int, str]],
    bug_file: str,
    bug_line: int = 1,
    quoted_class: bool = False,
) -> str:
    """Render a minimal CSA-shaped report.

    Args:
        sections: ``[(file_path, [code line, ...]), ...]`` — one per file.
        events: ``[(path_number, section_index, description), ...]``; the event
            is placed immediately after the section's LAST code line.
        bug_file: value of the ``BUGFILE`` metadata comment.
        quoted_class: emit ``class="FileName"`` instead of ``class=FileName``.
    """
    cls_attr = 'class="FileName"' if quoted_class else "class=FileName"
    parts = [
        "<html><body>",
        f"<!-- BUGTYPE Test bug -->",
        f"<!-- BUGFILE {bug_file} -->",
        f"<!-- BUGLINE {bug_line} -->",
    ]
    by_section: dict[int, list[tuple[int, str]]] = {}
    for num, sec_idx, desc in events:
        by_section.setdefault(sec_idx, []).append((num, desc))

    for sec_idx, (file_path, code_lines) in enumerate(sections):
        parts.append(f"<h4 {cls_attr}>{file_path}</h4>")
        parts.append(f'<table class="code" data-fileid="{sec_idx + 1}">')
        last = len(code_lines)
        for i, text in enumerate(code_lines, start=1):
            parts.append(
                f'<tr class="codeline" data-linenumber="{i}">'
                f'<td class="num" id="LN{i}">{i}</td>'
                f'<td class="line">{text}</td></tr>'
            )
            if i == last:
                for num, desc in by_section.get(sec_idx, []):
                    div_id = "EndPath" if num == 9999 else f"Path{num}"
                    kind = "msgEvent" if num != 9999 else "msgEvent"
                    parts.append(
                        '<tr><td class="num"></td><td class="line">'
                        f'<div id="{div_id}" class="msg {kind}" '
                        'style="margin-left:5ex"><table class="msgT"><tr>'
                        f'<td valign="top"><div class="PathIndex">{num}</div></td>'
                        f"<td>{desc}</td><td></td></tr></table></div>"
                        "</td></tr>"
                    )
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


class TestCrossFileAttribution:
    """Events must carry the file of the report section they appear in."""

    def test_events_use_their_own_section_file(self, tmp_path: Path) -> None:
        html = _build_report(
            sections=[
                ("/build/mine/a.cpp", ["line one", "line two"]),
                ("/build/dep/b.h", ["header one"]),
            ],
            events=[(1, 0, "Assuming 'p' is non-null"), (2, 1, "Null pointer")],
            bug_file="/build/dep/b.h",
        )
        path = tmp_path / "report.html"
        path.write_text(html)
        parsed = FPReportParser().parse(path)

        assert [e["file_path"] for e in parsed.path_events] == [
            "/build/mine/a.cpp",
            "/build/dep/b.h",
        ]
        # Lines come from the event's own section, not a global counter.
        assert [e["line_number"] for e in parsed.path_events] == [2, 1]

    def test_quoted_filename_class(self, tmp_path: Path) -> None:
        """``class="FileName"`` (quoted) must index the same as unquoted."""
        html = _build_report(
            sections=[("/build/mine/a.cpp", ["x"]), ("/build/dep/b.h", ["y"])],
            events=[(1, 1, "Null pointer")],
            bug_file="/build/dep/b.h",
            quoted_class=True,
        )
        path = tmp_path / "report.html"
        path.write_text(html)
        parsed = FPReportParser().parse(path)
        assert parsed.path_events[0]["file_path"] == "/build/dep/b.h"

    def test_single_section_report_falls_back_to_bug_file(
        self, tmp_path: Path
    ) -> None:
        """A report without a FileName header keeps the pre-index behaviour."""
        html = _build_report(
            sections=[("", ["x"])],
            events=[(1, 0, "Null pointer")],
            bug_file="/build/only/one.cpp",
        )
        # Drop the header entirely, as single-file reports have none.
        html = html.replace("<h4 class=FileName></h4>", "")
        path = tmp_path / "report.html"
        path.write_text(html)
        parsed = FPReportParser().parse(path)
        assert parsed.path_events[0]["file_path"] == "/build/only/one.cpp"

    def test_codeline_anchor_beats_legacy_window(self, tmp_path: Path) -> None:
        """A code line longer than the legacy 2000-char lookback is still found.

        The legacy scan searched backwards for a ``<td class=num>`` anchor;
        a very long source line pushes that anchor out of its window and the
        line came back as 0.
        """
        html = _build_report(
            sections=[("/build/a.cpp", ["z" * 3000, "short"])],
            events=[(1, 0, "Null pointer")],
            bug_file="/build/a.cpp",
        )
        path = tmp_path / "report.html"
        path.write_text(html)
        parsed = FPReportParser().parse(path)
        assert parsed.path_events[0]["line_number"] == 2

    def test_memory_test_events_follow_their_sections(
        self, memory_test_report: str
    ) -> None:
        """The 3-section MemoryTest fixture attributes by section, not by bug file."""
        parsed = FPReportParser().parse(memory_test_report)
        bug_name = Path(parsed.metadata.bug_file).name
        assert bug_name == "gtest-matchers.h"

        files = {Path(e["file_path"]).name for e in parsed.path_events}
        assert "MemoryTest.cpp" in files, "events in MemoryTest.cpp lost their file"
        assert "gmock-matchers.h" in files
        # The bug file is still reachable for the events that live there.
        assert len(files) > 1

    @pytest.mark.skipif(
        not FAISS_REPORT.exists(), reason="faiss corpus not present"
    )
    def test_clone_index_cross_file_split(self) -> None:
        """clone_index's TRYCLONE events belong to clone_index.cpp, not the bug file."""
        parsed = FPReportParser().parse(FAISS_REPORT)
        by_number = {e["path_number"]: e for e in parsed.path_events}

        for num in range(1, 7):
            assert by_number[num]["file_path"].endswith("faiss/clone_index.cpp")
            assert by_number[num]["line_number"] == 72
        for num in (7, 8, 9, 9999):
            assert by_number[num]["file_path"].endswith("utils/AlignedTable.h")
        assert by_number[9999]["line_number"] == 101
