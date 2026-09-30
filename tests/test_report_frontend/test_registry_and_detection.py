"""Registry, format detection, and finding selection.

Detection is the one place where a wrong answer is *silent*: analyse a SARIF
file with the CSA adapter and the failure looks like a missing-metadata error a
long way downstream, or worse, parses into a plausible-looking empty path.  The
tests below pin the three defences — suffix/content sniffing, the loud
ambiguity error, and the explicit ``--report-format`` override.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_client.report_frontend.errors import (
    ReportFrontendError,
    UnsupportedReportFormat,
)
from llm_client.report_frontend.registry import (
    detect_adapter,
    finding_for,
    get_adapter,
    load_document,
    rank_adapters,
    registered_adapters,
)
from llm_client.report_frontend.models import AnalyzerKind, ReportFormat


class TestRegistry:
    def test_adapters_are_registered(self) -> None:
        names = set(registered_adapters())
        assert {"csa-html", "sarif"} <= names

    def test_get_adapter_is_case_insensitive(self) -> None:
        assert get_adapter("SARIF").name == "sarif"

    def test_unknown_format_names_the_known_ones(self) -> None:
        with pytest.raises(UnsupportedReportFormat) as exc:
            get_adapter("coverity")
        assert "sarif" in str(exc.value)


class TestDetection:
    def test_csa_html_is_detected(self, csa_report: Path) -> None:
        assert detect_adapter(csa_report).name == "csa-html"

    def test_sarif_is_detected(self, sarif_path: Path) -> None:
        assert detect_adapter(sarif_path).name == "sarif"

    def test_infer_sarif_is_detected_by_content_not_producer(
        self, infer_sarif_path: Path
    ) -> None:
        # The container is SARIF whoever wrote it; the *analyzer* is refined per
        # run inside the adapter.
        assert detect_adapter(infer_sarif_path).name == "sarif"

    def test_extensionless_sarif_still_detected(self, tmp_path: Path,
                                                 sarif_path: Path) -> None:
        # CodeQL's default export is ``results.sarif`` but users copy it around;
        # content sniffing is what keeps detection from depending on the name.
        copy = tmp_path / "results"
        copy.write_text(sarif_path.read_text())
        assert detect_adapter(copy).name == "sarif"

    def test_detection_is_ranked(self, sarif_path: Path) -> None:
        ranked = rank_adapters(sarif_path)
        assert ranked and ranked[0][1].name == "sarif"

    def test_unrecognised_document_raises_with_the_fix(self, tmp_path: Path) -> None:
        junk = tmp_path / "notes.txt"
        junk.write_text("this is not a report\n")
        with pytest.raises(UnsupportedReportFormat) as exc:
            detect_adapter(junk)
        assert "--report-format" in str(exc.value)

    def test_override_bypasses_detection(self, sarif_path: Path) -> None:
        # The escape hatch for a document nothing recognises or two things do.
        assert detect_adapter(sarif_path, override="sarif").name == "sarif"


class TestAmbiguityIsLoud:
    """Two adapters claiming one document must never be resolved by import order."""

    def test_equal_scores_raise(self, tmp_path: Path, monkeypatch) -> None:
        from llm_client.report_frontend import registry

        class Rival:
            name = "rival"
            format = ReportFormat.UNKNOWN

            def sniff(self, path, head):
                return 1.0  # exactly what the CSA adapter scores on this header

        target = tmp_path / "report-x.html"
        target.write_text("<!-- BUGTYPE: x -->\n")
        monkeypatch.setitem(registry._ADAPTERS, "rival", Rival())
        with pytest.raises(UnsupportedReportFormat):
            detect_adapter(target)

    def test_a_weaker_rival_does_not_block_detection(self, tmp_path: Path,
                                                     monkeypatch) -> None:
        from llm_client.report_frontend import registry

        class Weaker:
            name = "weaker"
            format = ReportFormat.UNKNOWN

            def sniff(self, path, head):
                return 0.5

        target = tmp_path / "report-x.html"
        target.write_text("<!-- BUGTYPE: x -->\n")
        monkeypatch.setitem(registry._ADAPTERS, "weaker", Weaker())
        assert detect_adapter(target).name == "csa-html"

    def test_ambiguity_message_names_both(self, tmp_path: Path, monkeypatch) -> None:
        from llm_client.report_frontend import registry

        class Rival:
            name = "rival"
            format = ReportFormat.UNKNOWN

            def sniff(self, path, head):
                return 1.0

        target = tmp_path / "report-x.html"
        target.write_text("<!-- BUGTYPE: x -->\n")
        monkeypatch.setitem(registry._ADAPTERS, "rival", Rival())
        with pytest.raises(UnsupportedReportFormat) as exc:
            detect_adapter(target)
        message = str(exc.value)
        assert "ambiguous" in message
        assert "rival" in message and "csa-html" in message
        assert "--report-format" in message

    def test_override_resolves_the_ambiguity(self, tmp_path: Path,
                                             monkeypatch) -> None:
        from llm_client.report_frontend import registry

        class Rival:
            name = "rival"
            format = ReportFormat.UNKNOWN

            def sniff(self, path, head):
                return 1.0

        target = tmp_path / "report-x.html"
        target.write_text("<!-- BUGTYPE: x -->\n")
        monkeypatch.setitem(registry._ADAPTERS, "rival", Rival())
        assert detect_adapter(target, override="csa-html").name == "csa-html"


class TestLoadDocument:
    def test_loads_and_labels_the_document(self, sarif_path: Path) -> None:
        document = load_document(sarif_path)
        assert document.format is ReportFormat.SARIF
        assert document.analyzer is AnalyzerKind.CODEQL
        assert len(document.findings) == 2
        assert document.tool_version == "2.16.1"

    def test_infer_run_is_labelled_infer(self, infer_sarif_path: Path) -> None:
        document = load_document(infer_sarif_path)
        assert document.analyzer is AnalyzerKind.INFER


class TestFindingFor:
    def test_single_finding_needs_no_selector(self, csa_report: Path) -> None:
        document = load_document(csa_report)
        assert finding_for(document) is document.findings[0]

    def test_multi_finding_without_selector_raises_with_the_fix(
        self, sarif_path: Path
    ) -> None:
        document = load_document(sarif_path)
        with pytest.raises(UnsupportedReportFormat) as exc:
            finding_for(document)
        assert "--list-findings" in str(exc.value) or "::<id>" in str(exc.value)

    def test_selector_picks_one(self, sarif_path: Path) -> None:
        document = load_document(sarif_path)
        first = document.findings[0]
        assert finding_for(document, first.finding_id) is first

    def test_bad_selector_lists_the_real_ids(self, sarif_path: Path) -> None:
        document = load_document(sarif_path)
        with pytest.raises(ReportFrontendError) as exc:
            finding_for(document, "does-not-exist")
        assert document.findings[0].finding_id in str(exc.value)
