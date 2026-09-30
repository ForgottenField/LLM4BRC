"""``build_project_deps.collect_seeds`` — the reports decide which TUs get built.

This is the one consumer of the report frontend that runs *before* the pipeline
and leaves a durable artifact behind: ``pdg_<project>.json``.  A seed it gets
wrong is not a bad verdict, it is a graph that is missing the report's own
function — and nothing downstream can tell that from a function the analyzer
never had.  So the two things worth pinning here are:

* **who a report resolves to** — the bug host TU, the event TUs, the callees the
  closure walks, and whether the report's embedded source still lines up with the
  local checkout (the alignment check that refuses a drifted tree);
* **what happens when the report cannot be placed** — reported as a gap, never
  substituted, and graded: a *bug file* placed by name alone is refused (it is
  the report's own starting position), while an ambiguous *event file* is kept
  (it only widens the build).

It lives in this package because the frontend fixtures it leans on do, and
because the seeds are the frontend's largest real consumer outside the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import build_project_deps as bpd  # noqa: E402


def _sarif(path: Path, *, uri: str, line: int, flow: list[tuple[str, int]] = ()) -> Path:
    """A one-result SARIF document pointing at *uri* (no ``uriBaseId``).

    Written here rather than reusing the package fixture so the *path spelling*
    is the test's variable — a SARIF's ``uri`` is what the resolution ladder is
    given, and every rung is defined by its shape.
    """
    result: dict = {
        "ruleId": "cpp/example",
        "message": {"text": "flagged"},
        "locations": [{"physicalLocation": {
            "artifactLocation": {"uri": uri},
            "region": {"startLine": line},
        }}],
    }
    if flow:
        result["codeFlows"] = [{"threadFlows": [{"locations": [
            {"location": {"physicalLocation": {
                "artifactLocation": {"uri": u},
                "region": {"startLine": n},
            }}} for u, n in flow
        ]}]}]
    doc = {"version": "2.1.0", "runs": [{
        "tool": {"driver": {"name": "CodeQL", "version": "2.16.1", "rules": []}},
        "results": [result],
    }]}
    path.write_text(json.dumps(doc, indent=2))
    return path


class TestWhatAReportSeeds:
    def test_a_sarif_finding_seeds_its_host_and_its_flow(
        self, tmp_path: Path, source_tree: Path
    ) -> None:
        report = _sarif(
            tmp_path / "faiss.sarif",
            uri="faiss/impl/index_read.cpp", line=12,
            flow=[("faiss/impl/index_read.cpp", 5), ("faiss/impl/index_read.cpp", 9)],
        )
        (seeds,) = bpd.collect_seeds([report], source_tree)
        assert seeds.problems == []
        assert seeds.bug_file.endswith("faiss/impl/index_read.cpp")
        assert seeds.bug_line == 12
        assert seeds.local_file == source_tree / "faiss/impl/index_read.cpp"
        assert seeds.event_files == {source_tree / "faiss/impl/index_read.cpp"}

    def test_a_csa_report_seeds_the_same_way(
        self, csa_report: Path, tmp_path: Path
    ) -> None:
        # The corpus path must not have changed: same shape of answer, from a
        # report whose file is not on this machine.
        root = tmp_path / "MemoryTest"
        root.mkdir()
        (root / "CMakeLists.txt").write_text("project(memtest)\n")
        (seeds,) = bpd.collect_seeds([csa_report], root)
        assert seeds.bug_line > 0
        assert seeds.finding == ""          # one finding in the document
        assert seeds.label == csa_report.name
        assert seeds.local_file is None     # build-machine path, not a local file
        assert "unresolved bug file" in "; ".join(seeds.problems)

    def test_callees_are_collected_for_the_closure(
        self, csa_report: Path, tmp_path: Path
    ) -> None:
        # ``Calling 'X'`` steps drive ``close_over_callees`` — the second phase
        # of the seed set, and the only place the reports widen the build.
        root = tmp_path / "MemoryTest"
        root.mkdir()
        (seeds,) = bpd.collect_seeds([csa_report], root)
        assert seeds.callees


class TestOneFindingPerSeed:
    def test_a_multi_finding_document_yields_one_seed_each(
        self, sarif_path: Path, source_tree: Path
    ) -> None:
        seeds = bpd.collect_seeds([sarif_path], source_tree)
        assert len(seeds) == 2
        assert len({s.finding for s in seeds}) == 2
        assert all(s.finding for s in seeds)

    def test_the_label_distinguishes_them(
        self, sarif_path: Path, source_tree: Path
    ) -> None:
        seeds = bpd.collect_seeds([sarif_path], source_tree)
        labels = {s.label for s in seeds}
        assert len(labels) == 2
        assert all(sarif_path.name in label for label in labels)
        assert all("::" in label for label in labels)

    def test_each_finding_keeps_its_own_host(
        self, sarif_path: Path, source_tree: Path
    ) -> None:
        seeds = bpd.collect_seeds([sarif_path], source_tree)
        hosts = {s.bug_file for s in seeds}
        assert len(hosts) == 2

    def test_an_unreadable_report_is_a_gap_not_a_crash(
        self, tmp_path: Path, source_tree: Path
    ) -> None:
        # One bad file in a corpus must not stop the build; it must show up in
        # the gap list with the reason, which is what ``_finish`` writes out.
        bad = tmp_path / "broken.sarif"
        bad.write_text("{not json")
        (seeds,) = bpd.collect_seeds([bad], source_tree)
        assert seeds.problems and "unreadable report" in seeds.problems[0]


class TestPlacingTheHostIsNotGuessed:
    """The bug file is the report's own position — a name-only match is refused."""

    def test_an_ambiguous_basename_is_a_gap_that_names_the_rivals(
        self, tmp_path: Path, source_tree: Path
    ) -> None:
        # ``index_read.cpp`` exists under both ``faiss/impl`` and
        # ``faiss/invlists``; picking by name alone would build the wrong TU and
        # the artifact would silently lack the report's function.
        report = _sarif(tmp_path / "x.sarif", uri="index_read.cpp", line=12)
        (seeds,) = bpd.collect_seeds([report], source_tree)
        assert seeds.problems, "an ambiguous host must be refused"
        problem = seeds.problems[0]
        assert "basename" in problem
        assert "impl/index_read.cpp" in problem and "invlists/index_read.cpp" in problem

    def test_a_unique_basename_is_accepted(
        self, tmp_path: Path, source_tree: Path
    ) -> None:
        # One file of that name in the tree: name-only, but there is nothing to
        # confuse it with, so the report is usable.
        report = _sarif(tmp_path / "x.sarif", uri="AlignedTable.h", line=66)
        (seeds,) = bpd.collect_seeds([report], source_tree)
        assert seeds.problems == []
        assert seeds.local_file == source_tree / "faiss/utils/AlignedTable.h"

    def test_an_ambiguous_event_file_only_widens_the_build(
        self, tmp_path: Path, source_tree: Path
    ) -> None:
        # The same ambiguity on a *crossed* file is not fatal: the position the
        # analysis starts from is the host, and an extra TU costs build time
        # rather than correctness.
        report = _sarif(
            tmp_path / "x.sarif", uri="faiss/impl/index_read.cpp", line=12,
            flow=[("index_read.cpp", 7)],
        )
        (seeds,) = bpd.collect_seeds([report], source_tree)
        assert seeds.problems == []
        assert (source_tree / "faiss/impl/index_read.cpp") in seeds.event_files


class TestHostTranslationUnit:
    def test_a_source_host_is_its_own_tu(
        self, tmp_path: Path, source_tree: Path
    ) -> None:
        report = _sarif(tmp_path / "x.sarif", uri="faiss/impl/index_read.cpp", line=12)
        (seeds,) = bpd.collect_seeds([report], source_tree)
        assert bpd.host_tu_for(seeds, source_tree) == (
            source_tree / "faiss/impl/index_read.cpp"
        )

    def test_a_csa_name_still_yields_its_dot_cc(
        self, source_tree: Path
    ) -> None:
        # ``report-<file>-<fn>-<n>-<n>.html`` names the TU even when the report
        # itself could not be placed — the naming convention of the whole corpus,
        # and unchanged by the frontend rewrite.
        (source_tree / "faiss" / "utils" / "fourcc.cpp").write_text("//\n")
        rs = bpd.ReportSeeds(report=Path("report-fourcc.cpp-fourcc-3-1.html"))
        assert bpd.host_tu_for(rs, source_tree) == (
            source_tree / "faiss" / "utils" / "fourcc.cpp"
        )


class TestSourceResolutionStrategy:
    """The tool resolves through the frontend's ladder — not one of its own.

    A second ladder is how the same reported path comes to mean two different
    files in the build and in the analysis; the tool keeps only the *judgement*
    (what a name-only match is worth), never the rungs.
    """

    def test_a_unique_basename_is_reported_as_such(self, source_tree: Path) -> None:
        hit = bpd.resolve_reported("AlignedTable.h", source_tree)
        assert hit.local == source_tree / "faiss/utils/AlignedTable.h"
        assert hit.strategy == "basename" and not hit.is_ambiguous

    def test_a_shared_basename_records_its_rivals(self, source_tree: Path) -> None:
        hit = bpd.resolve_reported("index_read.cpp", source_tree)
        assert hit.is_ambiguous
        assert hit.local in {source_tree / "faiss/impl/index_read.cpp",
                             source_tree / "faiss/invlists/index_read.cpp"}
        assert hit.local not in hit.rivals

    def test_a_relative_path_is_a_strong_match(self, source_tree: Path) -> None:
        # The rung the tool used to lack: a SARIF/Cppcheck ``uri`` is relative to
        # the tree, and resolving it by basename instead is a coin flip.
        hit = bpd.resolve_reported("faiss/impl/index_read.cpp", source_tree)
        assert hit.local == source_tree / "faiss/impl/index_read.cpp"
        assert hit.strategy == "relative"
        assert hit.is_strong and not hit.is_ambiguous

    def test_a_suffix_that_pins_the_position_is_strong(self, source_tree: Path) -> None:
        hit = bpd.resolve_reported("/somewhere/else/faiss/invlists/index_read.cpp",
                                   source_tree)
        assert hit.local == source_tree / "faiss/invlists/index_read.cpp"
        assert hit.strategy == "suffix" and hit.is_strong

    def test_a_build_prefix_is_stripped(self, source_tree: Path) -> None:
        hit = bpd.resolve_reported(
            "/home/me/projects/cpp-project/faiss/faiss/impl/index_read.cpp",
            source_tree,
        )
        assert hit.local == source_tree / "faiss/impl/index_read.cpp"
        assert hit.strategy == "build-prefix"

    def test_an_absolute_path_that_exists_is_direct(self, source_tree: Path) -> None:
        target = source_tree / "faiss/impl/index_read.cpp"
        hit = bpd.resolve_reported(str(target), source_tree)
        assert hit.local == target.resolve()
        assert hit.strategy == "direct"

    def test_nothing_found_says_so(self, source_tree: Path) -> None:
        assert not bpd.resolve_reported("nope/missing.cpp", source_tree).resolved
        assert not bpd.resolve_reported("", source_tree).resolved

    def test_resolve_source_is_the_path_only_form(self, source_tree: Path) -> None:
        # ``host_tu_for`` and the alignment check call the two-value form.
        assert bpd.resolve_source(
            "faiss/impl/index_read.cpp", source_tree
        ) == source_tree / "faiss/impl/index_read.cpp"
