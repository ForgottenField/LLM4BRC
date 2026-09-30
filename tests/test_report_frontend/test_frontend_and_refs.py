"""`ReportRef` addressing and the frontend facade.

A CSA report is one file and one finding, so "which report" and "which finding"
were the same question for as long as CSA was the only input.  A SARIF export
makes them different, and the pipeline's unit of work is the *finding*: it
writes one result JSON, generates one POC, and names both after the report.

``ReportRef`` is what carries that distinction without a second kind of path
threaded through 2,600 lines of CLI — it is an :class:`os.PathLike`, so the
dozen places that do ``Path(report_path)``, ``report_path.stem`` or
``report_path.name`` keep working, and ``key`` is what names the outputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_client.report_frontend import (
    LoadedReport,
    ReportFrontend,
    ReportRef,
    load_report,
)
from llm_client.report_frontend.errors import ReportNotFoundError


class TestReportRefParsing:
    def test_bare_path_has_no_selector(self) -> None:
        ref = ReportRef.parse("/tmp/report-a.html")
        assert ref.path == Path("/tmp/report-a.html")
        assert ref.selector == ""

    def test_selector_is_split_off_the_right(self) -> None:
        ref = ReportRef.parse("/tmp/x.sarif::cpp/rule@a.cpp:12")
        assert ref.path == Path("/tmp/x.sarif")
        assert ref.selector == "cpp/rule@a.cpp:12"

    def test_a_path_containing_a_colon_is_not_mangled(self) -> None:
        # ``::`` cannot appear in a POSIX filename component, but a single
        # colon can (and does, in Windows-style spellings in some reports).
        ref = ReportRef.parse("/tmp/C:/weird/x.sarif")
        assert ref.path == Path("/tmp/C:/weird/x.sarif")
        assert ref.selector == ""

    def test_parse_is_idempotent(self, sarif_path: Path) -> None:
        ref = ReportRef.parse(sarif_path)
        assert ReportRef.parse(ref) is ref

    def test_selector_is_stripped(self) -> None:
        assert ReportRef.parse("/a.sarif::  x  ").selector == "x"


class TestReportRefPathLikeness:
    @pytest.fixture
    def ref(self, sarif_path: Path) -> ReportRef:
        return ReportRef(sarif_path, "cpp/rule@a.cpp:12")

    def test_os_pathlike(self, ref: ReportRef) -> None:
        import os

        assert os.fspath(ref) == str(ref.path)
        assert Path(ref) == ref.path

    def test_path_attributes_match_the_file(self, ref: ReportRef) -> None:
        # These are what the CLI and the tools already call; a ref that did not
        # answer them would force a rewrite of every call site.
        assert ref.name == ref.path.name
        assert ref.stem == ref.path.stem
        assert ref.suffix == ref.path.suffix
        assert ref.parent == ref.path.parent
        assert ref.exists()

    def test_key_is_the_stem_without_a_selector(self, ref: ReportRef) -> None:
        plain = ReportRef(ref.path)
        assert plain.key == ref.path.stem

    def test_key_appends_a_sanitised_selector(self, ref: ReportRef) -> None:
        assert ref.key.startswith(ref.path.stem + "__")
        assert "/" not in ref.key and "@" not in ref.key and "::" not in ref.key
        assert ref.key.endswith(".json") is False

    def test_key_is_bounded(self, ref: ReportRef) -> None:
        long_ref = ReportRef(ref.path, "x" * 500)
        assert len(long_ref.key) <= len(ref.path.stem) + 2 + 80

    def test_key_is_filesystem_safe_for_every_corpus_id(
        self, sarif_path: Path
    ) -> None:
        for selector in ("cpp/rule@a.cpp:12", "a::b", "weird name?", "..."):
            key = ReportRef(sarif_path, selector).key
            assert key and not any(c in key for c in '/\\:*?"<>|')

    def test_with_selector_keeps_the_path(self, ref: ReportRef) -> None:
        other = ref.with_selector("other")
        assert other.path == ref.path and other.selector == "other"


class TestExpansion:
    def test_a_single_finding_document_is_itself(self, csa_report: Path) -> None:
        ref = ReportRef(csa_report)
        assert ReportFrontend().expand([ref]) == [ref]

    def test_a_multi_finding_document_becomes_one_ref_per_finding(
        self, sarif_path: Path
    ) -> None:
        refs = ReportFrontend().expand([ReportRef(sarif_path)])
        assert len(refs) == 2
        assert all(r.path == sarif_path for r in refs)
        assert len({r.selector for r in refs}) == 2
        assert all(r.selector for r in refs)

    def test_explicit_selectors_are_respected(self, sarif_path: Path) -> None:
        ref = ReportRef(sarif_path, "cpp/leap-year@AlignedTable.h:66")
        assert ReportFrontend().expand([ref]) == [ref]

    def test_an_unreadable_document_is_passed_through_for_the_pipeline_to_report(
        self, tmp_path: Path
    ) -> None:
        # Expansion must not swallow the error: the per-report ERROR row with
        # the real message is more useful than a silently shorter batch.
        bad = tmp_path / "broken.sarif"
        bad.write_text("{not json")
        refs = ReportFrontend().expand([ReportRef(bad)])
        assert refs == [ReportRef(bad)]

    def test_expansion_keys_are_unique(self, sarif_path: Path) -> None:
        keys = [r.key for r in ReportFrontend().expand([ReportRef(sarif_path)])]
        assert len(set(keys)) == len(keys)


class TestLoading:
    def test_finding_selected_by_selector(self, sarif_path: Path) -> None:
        frontend = ReportFrontend()
        first = frontend.load(ReportRef(sarif_path, "cpp/leap-year@AlignedTable.h:66"))
        assert first.finding.rule_id == "cpp/leap-year"
        assert first.sibling_findings == 2

    def test_missing_report_raises_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ReportNotFoundError):
            ReportFrontend().load(tmp_path / "nope.html")

    def test_warns_about_paths_it_cannot_place(
        self, sarif_path: Path, tmp_path: Path
    ) -> None:
        # The source root is a *different* tree — every path in the report is
        # unresolvable, and saying so early is the point (the pipeline has been
        # silently run against the wrong project twice).
        empty = tmp_path / "empty-project"
        empty.mkdir()
        loaded = ReportFrontend(source_root=empty).load(
            ReportRef(sarif_path, "cpp/leap-year@AlignedTable.h:66")
        )
        assert loaded.warnings
        assert "AlignedTable.h" in loaded.warnings[0]

    def test_no_warning_when_everything_resolves(
        self, csa_report: Path
    ) -> None:
        loaded = ReportFrontend().load(csa_report)
        # This fixture's bug file is a build-machine path that does not exist
        # here, so there *is* a warning — the assertion is on the API, not on
        # the fixture's luck.
        assert isinstance(loaded.unresolved_files(), list)

    def test_summary_is_serialisable(self, sarif_path: Path) -> None:
        import json

        loaded = ReportFrontend().load(
            ReportRef(sarif_path, "cpp/leap-year@AlignedTable.h:66")
        )
        payload = json.dumps(loaded.summary(), default=str)
        assert "sarif" in payload

    def test_load_report_convenience(self, csa_report: Path) -> None:
        loaded = load_report(csa_report)
        assert isinstance(loaded, LoadedReport)
        assert loaded.ref.path == csa_report

    def test_a_selector_is_required_for_a_multi_finding_file(
        self, sarif_path: Path
    ) -> None:
        from llm_client.report_frontend.errors import UnsupportedReportFormat

        with pytest.raises(UnsupportedReportFormat):
            ReportFrontend().load(sarif_path)

    def test_iter_findings(self, sarif_path: Path) -> None:
        pairs = list(ReportFrontend().iter_findings(sarif_path))
        assert len(pairs) == 2
        assert all(isinstance(fid, str) and fid for fid, _ in pairs)

    def test_detect_format(self, sarif_path: Path, csa_report: Path) -> None:
        frontend = ReportFrontend()
        assert frontend.detect_format(sarif_path).value == "sarif"
        assert frontend.detect_format(csa_report).value == "csa-html"

    def test_pinned_format_overrides_detection(self, tmp_path: Path) -> None:
        # No extension and nothing a sniffer recognises: only the pin can read
        # it, and that is the documented escape hatch.
        from llm_client.report_frontend.errors import UnsupportedReportFormat

        path = tmp_path / "mystery"
        path.write_text("nothing here looks like a report\n")
        with pytest.raises(UnsupportedReportFormat):
            ReportFrontend().detect_format(path)
        assert ReportFrontend(report_format="csa-html").detect_format(path).value == (
            "csa-html"
        )
