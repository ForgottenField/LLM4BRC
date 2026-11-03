"""A report that states only a *location* ends in UNKNOWN, never in TP.

Stage 9 can always *construct* a feasible path — that is what path selection
does — so for a location-only report the pipeline would hand its own path to
POC verification.  The step-by-step conformance audit is what normally keeps
that honest, and it compares against the **reported** path; with no reported
path only the report step is left, so every POC passes it vacuously.  The
verdict is therefore UNKNOWN: `triggered` would otherwise stand on a path the
pipeline chose, which is the "POC manufactures its own precondition" failure
mode at report scale.

The fact itself is set by the adapter and carried by the bridge
(``tests/test_report_frontend/`` and ``TestLocationOnlyReports`` in
``test_cross_file_attribution.py`` cover that half); this file covers what the
pipeline does with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_path_selection as rps  # noqa: E402


class TestNoPathVerdict:
    def test_a_reported_path_verifies_as_usual(self) -> None:
        # CSA always prints the path it walked; `None` means "carry on to
        # `_verify_path`".
        assert rps._no_path_verdict("trace") is None

    def test_a_location_only_report_gets_a_step_and_a_reason(self) -> None:
        result = rps._no_path_verdict("location")
        assert result is not None
        step, reason = result

        # The step names the stage and the keyword the summary tooling greps —
        # deliberately NOT `遍历超限` (that phrase means the exploration budget
        # ran out, which is not what happened here).
        assert "10" in step
        assert rps._NO_PATH_STEP_KEYWORD in step
        assert "遍历超限" not in step

        # The reason has to say what makes this UNKNOWN rather than TP: there
        # is no reported path for the POC conformance audit to check against.
        assert "UNKNOWN" in reason
        assert "对照" in reason
        assert "path_evidence=location" in reason

    @pytest.mark.parametrize("weaker", ["", "unknown", "LOCATION_ONLY"])
    def test_anything_but_a_trace_is_treated_as_no_path(self, weaker: str) -> None:
        # Fail-safe direction: an unset or unrecognised value must not license a
        # TP.  The adapters always set the field; a caller that forgets it gets
        # the conservative reading, same as `ReportMetadata.has_path`.
        assert rps._no_path_verdict(weaker) is not None

    def test_the_model_and_the_policy_agree(self) -> None:
        # The policy reads the same string the pipeline carries, and
        # `ReportMetadata.has_path` is the same test on the model side — the two
        # must not drift.
        from fp_analysis.models import ReportMetadata

        trace = ReportMetadata(bug_type="t", bug_file="f.cc", bug_line=1,
                               path_evidence="trace")
        loc = ReportMetadata(bug_type="t", bug_file="f.cc", bug_line=1,
                             path_evidence="location")
        default = ReportMetadata(bug_type="t", bug_file="f.cc", bug_line=1)

        assert rps._no_path_verdict(trace.path_evidence) is None
        assert trace.has_path
        for meta in (loc, default):
            assert rps._no_path_verdict(meta.path_evidence) is not None
            assert not meta.has_path
