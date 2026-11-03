"""The CLI seam: work items, output names, source-root inference, entry function.

Stage 1 of ``run_path_selection`` is where a report becomes a *unit of work*, and
that is where the formats refactor is riskiest — everything here used to assume
"one report file = one work item", and a SARIF export breaks that assumption in
four separate places:

* the report identity in the result JSON and the batch summary,
* the output filename (sixty findings must not write to one file),
* the source-root inference (``<project>/reports/{TP,FP}/x`` is a CSA layout),
* the entry function (a SARIF finding names none, and the CFG-cache last resort
  and the slice's fallback mask both want one).

Each is checked here without an LLM call, because a misinvocation must cost
seconds, not tens of minutes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_pipeline as rps  # noqa: E402
from pipeline import runner as pipeline_runner

from report_frontend import ReportRef  # noqa: E402
from fp_analysis.pdg_models import FunctionPDG, PDGNode, SDG  # noqa: E402


@pytest.fixture
def pdg(tmp_path: Path) -> Path:
    """A hand-made PDG artifact — the documented ``--pdg`` escape hatch.

    It carries no ``metadata`` block, which ``_check_pdg_artifact_version``
    warns about and accepts; a *real* artifact's declared version is guarded
    separately (see :class:`TestPdgVersionGuard`).
    """
    path = tmp_path / "pdg.json"
    path.write_text('{"functions": {}}')
    return path


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        project=None, projects_dir=str(REPO_ROOT / "does-not-exist"),
        source_root=None, pdg=None, cfg_cache=None, report=None,
        scan_dir=None, report_format=None, output=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestWorkItemIdentity:
    def test_a_plain_path_keeps_the_file_name(self, csa_report: Path) -> None:
        # Every existing CSA result JSON must read exactly as it did.
        ref = ReportRef(csa_report)
        assert rps.report_fields(ref)["report"] == csa_report.name
        assert rps.report_fields(ref)["finding"] == ""

    def test_a_selector_switches_to_the_key(self, sarif_path: Path) -> None:
        ref = ReportRef(sarif_path, "cpp/rule@a.cpp:12")
        fields = rps.report_fields(ref)
        assert fields["report"] == ref.key
        assert fields["report"] == ref.key != sarif_path.name
        assert fields["report_file"] == sarif_path.name
        assert fields["finding"] == "cpp/rule@a.cpp:12"

    def test_findings_of_one_file_get_distinct_names(self, sarif_path: Path) -> None:
        names = {
            rps.report_fields(ReportRef(sarif_path, f"rule@a.cpp:{n}"))["report"]
            for n in (1, 2, 3)
        }
        assert len(names) == 3

    def test_output_path_is_namespaced_by_the_key(self, sarif_path: Path) -> None:
        out = rps.report_out_path(_args(), ReportRef(sarif_path, "r@a.cpp:12"))
        assert out.name.startswith("path_selection_")
        assert out.name.endswith(".json")
        assert sarif_path.stem in out.name and "r@a.cpp:12" not in out.name

    def test_output_dir_mode_for_batch(self, tmp_path: Path, sarif_path: Path) -> None:
        args = _args(output=str(tmp_path), project="faiss")
        out = rps.report_out_path(args, ReportRef(sarif_path, "r@a.cpp:12"))
        assert out.parent == tmp_path

    def test_report_key_of_a_bare_path(self) -> None:
        assert rps.report_key(Path("/tmp/report-a.html")) == "report-a"


class TestReportCollection:
    def test_sarif_counts_as_a_report_file(self, tmp_path: Path) -> None:
        assert rps._is_report_file(tmp_path / "results.sarif")
        assert rps._is_report_file(tmp_path / "results.sarif.json")
        assert not rps._is_report_file(tmp_path / "index.html")
        assert not rps._is_report_file(tmp_path / "other.html")
        assert rps._is_report_file(tmp_path / "report-x.html")

    def test_scan_dir_expands_a_sarif_into_one_ref_per_finding(
        self, tmp_path: Path, sarif_path: Path
    ) -> None:
        # Its own directory: the fixture's copy must not be scanned too, or the
        # count says more about the fixture than about the expansion.
        scan = tmp_path / "scan"
        scan.mkdir()
        shutil.copy(sarif_path, scan / "results.sarif")
        refs = rps._collect_reports(_args(scan_dir=str(scan)))
        assert len(refs) == 2
        assert all(isinstance(r, ReportRef) for r in refs)
        assert {r.path for r in refs} == {(scan / "results.sarif").resolve()}
        assert len({r.key for r in refs}) == 2

    def test_scan_dir_finds_csa_reports(
        self, tmp_path: Path, csa_report: Path
    ) -> None:
        scan = tmp_path / "scan"
        scan.mkdir()
        shutil.copy(csa_report, scan / "report-x.html")
        refs = rps._collect_reports(_args(scan_dir=str(scan)))
        # A CSA report is one finding and needs no selector — the ref must stay
        # bare so its result JSON and output name are byte-identical to today's.
        assert refs == [ReportRef((scan / "report-x.html").resolve())]

    def test_single_report_with_a_selector_is_one_work_item(
        self, sarif_path: Path
    ) -> None:
        spec = f"{sarif_path}::cpp/leap-year@AlignedTable.h:66"
        refs = rps._collect_reports(_args(report=spec))
        assert len(refs) == 1
        assert refs[0].selector == "cpp/leap-year@AlignedTable.h:66"

    def test_missing_scan_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            rps._collect_reports(_args(scan_dir=str(tmp_path / "nope")))


class TestTheSelectorReachesStageOne:
    """Collecting the right work item is not enough — stage 1 must receive it.

    ``process_report`` rebinds its argument to ``report_ref.path`` for the
    banner, and then handed *that* to ``parse_report``.  The selection was
    therefore lost on the last step of the way: the frontend got a
    selector-less ref for a multi-finding document and correctly refused it
    ("holds 62 findings; select one"), so `--report '<doc>::<finding>'` and
    every ``--scan-dir`` expansion of a SARIF died before stage 1 — no SARIF
    finding could be analysed at all.  Reported on the real CodeQL corpus.
    """

    def test_parse_report_receives_the_ref_not_the_bare_path(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        ref = ReportRef(
            tmp_path / "results.sarif", "cpp/leap-year@AlignedTable.h:66"
        )
        # The seam lives in the pipeline runner now, not in the script: the two
        # names are looked up in `runner`'s own globals, so patching `rps` would
        # bind a second, unread name.  The *assertion* below is what this test
        # has always been about and is unchanged.
        monkeypatch.setattr(
            pipeline_runner, "_resolve_paths",
            lambda args, rp: (ref, None, tmp_path / "pdg.json", tmp_path),
        )
        seen: dict = {}

        class _Stop(Exception):
            """Ends `process_report` at stage 1 — nothing below is under test."""

        def _fake_parse(report, source_root, report_format=None):
            seen["report"] = report
            raise _Stop

        monkeypatch.setattr(pipeline_runner, "parse_report", _fake_parse)
        with pytest.raises(_Stop):
            rps.process_report(_args(report=str(ref)), ref)

        got = seen["report"]
        assert isinstance(got, ReportRef), (
            "stage 1 got a bare path, so the finding selector is gone"
        )
        assert got.selector == "cpp/leap-year@AlignedTable.h:66"
        assert got.path == ref.path


class TestSourceRootInference:
    def test_the_report_layout_is_tried_first(self, tmp_path: Path,
                                              csa_report: Path) -> None:
        # <project>/reports/FP/report-x.html — the layout of every CSA corpus.
        root = tmp_path / "faiss"
        reports = root / "reports" / "FP"
        reports.mkdir(parents=True)
        (root / "CMakeLists.txt").write_text("project(faiss)\n")
        ref = ReportRef(reports / "report-x.html")
        found, _ = rps._infer_source_root(_args(), ref)
        assert found == root.resolve()

    def test_a_bare_sarif_falls_back_to_projects_dir(self, tmp_path: Path) -> None:
        # ``faiss.sarif`` copied anywhere: the only honest guess is that a
        # project of that name lives under --projects-dir.
        projects = tmp_path / "projects"
        (projects / "faiss").mkdir(parents=True)
        (projects / "faiss" / "Makefile").write_text("all:\n")
        sarif = tmp_path / "faiss.sarif"
        sarif.write_text("{}")
        found, _ = rps._infer_source_root(
            _args(projects_dir=str(projects)), ReportRef(sarif)
        )
        assert found == (projects / "faiss").resolve()

    def test_nothing_matching_returns_none_and_says_what_it_tried(
        self, tmp_path: Path
    ) -> None:
        sarif = tmp_path / "mystery.sarif"
        sarif.write_text('{"version": "2.1.0", "runs": []}')
        found, tried = rps._infer_source_root(
            _args(projects_dir=str(tmp_path / "empty")), ReportRef(sarif)
        )
        assert found is None
        assert tried  # the caller puts this in the error message

    def test_the_error_names_every_candidate(self, tmp_path: Path) -> None:
        sarif = tmp_path / "mystery.sarif"
        sarif.write_text('{"version": "2.1.0", "runs": []}')
        args = _args(projects_dir=str(tmp_path / "empty"))
        with pytest.raises(ValueError) as exc:
            rps._resolve_paths(args, ReportRef(sarif))
        message = str(exc.value)
        assert "--source-root" in message
        assert "cannot infer" in message

    def test_never_substitutes_another_project(self, tmp_path: Path) -> None:
        # The defect this guards: an unroutable report used to be analysed
        # against aria2 — a plausible-looking wrong run instead of an error.
        sarif = tmp_path / "mystery.sarif"
        sarif.write_text('{"version": "2.1.0", "runs": []}')
        with pytest.raises(ValueError):
            rps._resolve_paths(
                _args(projects_dir=str(tmp_path / "empty")), ReportRef(sarif)
            )

    def test_explicit_project_wins(self, tmp_path: Path, sarif_path: Path,
                                   pdg: Path) -> None:
        root = tmp_path / "faiss"
        root.mkdir()
        (root / "CMakeLists.txt").write_text("project(faiss)\n")
        _, _, pdg_path, source_root = rps._resolve_paths(
            _args(project=str(root), pdg=str(pdg)), ReportRef(sarif_path),
        )
        assert source_root == root.resolve()
        assert pdg_path == pdg.resolve()

    def test_missing_pdg_raises_rather_than_using_another(self, tmp_path: Path,
                                                          sarif_path: Path) -> None:
        # A project name with no ``pdg_<name>.json`` beside the code: the run
        # must stop here, not pick up a *different* project's graph — one that
        # parses fine and analyses the wrong call graph end to end.
        root = tmp_path / "nosuchproject"
        root.mkdir()
        (root / "CMakeLists.txt").write_text("project(nosuchproject)\n")
        with pytest.raises(ValueError) as exc:
            rps._resolve_paths(_args(project=str(root)), ReportRef(sarif_path))
        assert "pdg_nosuchproject.json" in str(exc.value)
        assert "--pdg" in str(exc.value)

    def test_lite_first_allows_a_missing_exact_pdg(self, tmp_path: Path,
                                                   sarif_path: Path) -> None:
        root = tmp_path / "nosuchproject"
        root.mkdir()
        (root / "CMakeLists.txt").write_text("project(nosuchproject)\n")

        _, _, pdg_path, source_root = rps._resolve_paths(
            _args(project=str(root)),
            ReportRef(sarif_path),
            require_pdg=False,
        )

        assert pdg_path is None
        assert source_root == root.resolve()


class TestPdgVersionGuard:
    """A stale artifact parses fine; only its declaration says it is stale."""

    def _resolve(self, tmp_path: Path, sarif_path: Path, pdg_body: str):
        root = tmp_path / "faiss"
        root.mkdir(exist_ok=True)
        (root / "CMakeLists.txt").write_text("project(faiss)\n")
        pdg = tmp_path / "pdg.json"
        pdg.write_text(pdg_body)
        return rps._resolve_paths(
            _args(project=str(root), pdg=str(pdg)), ReportRef(sarif_path)
        )

    def test_a_stale_artifact_is_refused(self, tmp_path: Path,
                                         sarif_path: Path) -> None:
        from fp_analysis.pdg_models import PDG_ARTIFACT_VERSION

        with pytest.raises(ValueError) as exc:
            self._resolve(
                tmp_path, sarif_path,
                json.dumps({"metadata": {"version": "1.0"}}),
            )
        message = str(exc.value)
        assert "1.0" in message and PDG_ARTIFACT_VERSION in message
        assert "build_project_deps.py" in message  # the fix, not just the fault

    def test_the_current_version_passes(self, tmp_path: Path,
                                        sarif_path: Path) -> None:
        from fp_analysis.pdg_models import PDG_ARTIFACT_VERSION

        ref, _, pdg_path, _ = self._resolve(
            tmp_path, sarif_path,
            json.dumps({"metadata": {"version": PDG_ARTIFACT_VERSION}}),
        )
        assert pdg_path.is_file()

    def test_no_metadata_is_a_warning_not_a_refusal(self, tmp_path: Path,
                                                    sarif_path: Path) -> None:
        # A hand-made fixture has nothing to compare; refusing it would make
        # ``--pdg`` unusable for tests and one-off runs.
        self._resolve(tmp_path, sarif_path, '{"functions": {}}')


class TestResolvePathsKeepsTheSelector:
    def test_the_selector_survives_resolution(self, tmp_path: Path,
                                              sarif_path: Path, pdg: Path) -> None:
        root = tmp_path / "faiss"
        root.mkdir()
        (root / "CMakeLists.txt").write_text("project(faiss)\n")
        ref, _, _, _ = rps._resolve_paths(
            _args(project=str(root), pdg=str(pdg)),
            ReportRef(sarif_path, "cpp/leap-year@AlignedTable.h:66"),
        )
        assert isinstance(ref, ReportRef)
        assert ref.selector == "cpp/leap-year@AlignedTable.h:66"
        assert ref.path.is_absolute()

    def test_a_missing_report_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            rps._resolve_paths(_args(), ReportRef(tmp_path / "nope.sarif"))


class TestEntryFunctionInference:
    """A SARIF finding names no function; the PDG is asked, and only then."""

    @pytest.fixture
    def sdg(self) -> SDG:
        def fn(name: str, file: str, lines: list[int]) -> FunctionPDG:
            return FunctionPDG(
                function_name=name, source_file=file,
                nodes={
                    f"S_{line}_1": PDGNode(id=f"S_{line}_1", source_line=line)
                    for line in lines
                },
            )

        return SDG(pdgs={
            "outer": fn("outer", "faiss/impl/index_read.cpp", [10, 40]),
            "inner": fn("inner", "faiss/impl/index_read.cpp", [12, 20]),
            "other_file": fn("other_file", "faiss/utils/other.cpp", [12, 20]),
        })

    @staticmethod
    def _parsed(function: str, file: str, line: int):
        from fp_analysis.models import ParsedReport, ReportMetadata

        return ParsedReport(
            metadata=ReportMetadata(
                bug_type="cpp/x", bug_category="", bug_file=file, bug_line=line,
                bug_column=0, bug_description="", function_name=function,
                path_length=0,
            ),
            path_events=[],
            source_snippets={},
            raw_html_path="",
        )

    def test_the_analyzers_own_answer_is_never_second_guessed(self, sdg: SDG) -> None:
        parsed = self._parsed("stated_by_report", "faiss/impl/index_read.cpp", 15)
        assert rps.infer_entry_function(sdg, parsed) == "stated_by_report"

    def test_the_tightest_containing_range_wins(self, sdg: SDG) -> None:
        parsed = self._parsed("", "faiss/impl/index_read.cpp", 15)
        # ``outer`` (10-40) and ``inner`` (12-20) both contain line 15; the
        # inner one is the function the line is actually in.
        assert rps.infer_entry_function(sdg, parsed) == "inner"

    def test_the_file_must_match(self, sdg: SDG) -> None:
        parsed = self._parsed("", "faiss/utils/other.cpp", 15)
        assert rps.infer_entry_function(sdg, parsed) == "other_file"

    def test_no_containing_function_is_reported_empty(self, sdg: SDG) -> None:
        parsed = self._parsed("", "faiss/impl/index_read.cpp", 999)
        assert rps.infer_entry_function(sdg, parsed) == ""

    def test_no_location_is_reported_empty(self, sdg: SDG) -> None:
        assert rps.infer_entry_function(sdg, self._parsed("", "", 0)) == ""


class TestListingFlags:
    """``--list-formats`` / ``--list-findings`` run without an API key."""

    def test_list_formats(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--list-formats"])
        with pytest.raises(SystemExit) as exc:
            rps.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "csa-html" in out and "sarif" in out

    def test_list_findings(self, capsys, monkeypatch, sarif_path: Path) -> None:
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--list-findings", "--report", str(sarif_path),
        ])
        with pytest.raises(SystemExit) as exc:
            rps.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "2 finding(s)" in out
        assert "cpp/uncontrolled-allocation-size@index_read.cpp:12" in out
        assert "cpp/leap-year@AlignedTable.h:66" in out

    def test_list_findings_needs_a_report(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--list-findings",
        ])
        with pytest.raises(SystemExit) as exc:
            rps.main()
        assert exc.value.code == 1
        assert "--report" in capsys.readouterr().err
