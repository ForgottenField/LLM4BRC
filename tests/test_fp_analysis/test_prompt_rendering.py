"""Prompt rendering guards.

A segment's candidate is one concrete branch combination over that segment's
branches, so its decision list grows with the segment's branch count.  Once
segments resolve to their real enclosing function (rather than to a same-named
function elsewhere), big ones reach tens of thousands of branches per segment —
rendering every decision overflowed the model context (faiss ``read_index``
requested 1.2M tokens against a 1M limit).  The display is therefore capped.
"""

from __future__ import annotations

from types import SimpleNamespace

from llm_client.fp_analysis.path_analyzer import (
    _MAX_RENDERED_DECISIONS, _MAX_RENDERED_STATE_CHARS, FPAnalyzer,
)


def _decision(line: int) -> SimpleNamespace:
    return SimpleNamespace(branch_taken=True, condition_expr=f"x > {line}",
                           node_id=f"n{line}", source_line=line)


def _metrics() -> SimpleNamespace:
    return SimpleNamespace(S=1, C=0, V=1)


class TestFormatCandidates:
    def test_short_candidate_is_rendered_in_full(self) -> None:
        candidates = [([_decision(1), _decision(2)], _metrics())]

        text = FPAnalyzer._format_candidates(candidates)

        assert "x > 1 → TRUE" in text
        assert "x > 2 → TRUE" in text
        assert "more branch decisions" not in text

    def test_long_candidate_is_truncated_with_a_count(self) -> None:
        decisions = [_decision(i) for i in range(5000)]

        text = FPAnalyzer._format_candidates([(decisions, _metrics())])

        rendered = text.count("→ TRUE")
        assert rendered == _MAX_RENDERED_DECISIONS
        assert "… (+4960 more branch decisions in this candidate — 5000 total)" in text

    def test_truncation_keeps_every_candidate_and_its_metrics(self) -> None:
        candidates = [([_decision(i) for i in range(100)], _metrics()),
                      ([_decision(i) for i in range(100)], _metrics())]

        text = FPAnalyzer._format_candidates(candidates)

        assert "**Candidate #1** (S=1, C=0, V=1)" in text
        assert "**Candidate #2** (S=1, C=0, V=1)" in text

    def test_linear_path_still_reads_as_linear(self) -> None:
        text = FPAnalyzer._format_candidates([([], _metrics())])

        assert "(no branch decisions — linear path)" in text


class TestClipState:
    def test_short_state_is_untouched(self) -> None:
        assert FPAnalyzer._clip_state("ptr == nullptr") == "ptr == nullptr"

    def test_empty_state(self) -> None:
        assert FPAnalyzer._clip_state(None) == ""
        assert FPAnalyzer._clip_state("") == ""

    def test_long_state_is_truncated_with_its_size(self) -> None:
        """A segment inside a large function returns ~1 MB of state."""
        state = "x" * 500_000

        clipped = FPAnalyzer._clip_state(state)

        assert clipped.startswith("x" * 100)
        assert clipped.count("x") == _MAX_RENDERED_STATE_CHARS
        assert f"{len(state)} characters total" in clipped

