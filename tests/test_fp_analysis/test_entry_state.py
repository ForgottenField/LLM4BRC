"""Tests for the deterministic fabricated-entry-state check (stage 6).

The check joins three independently-sourced facts:

* the *report* — its first claim is ``Assuming pointer value is null``, and it
  records taking the TRUE branch of a check on that pointer before the deref;
* the *bug file* — the flagged function carries an always-compiled fatal check
  (``ABSL_RAW_CHECK``/``CHECK``/``FAISS_THROW_IF_NOT``) on that parameter whose
  failure arm does not return;
* the *signature* — the dereferenced variable really is a pointer parameter of
  that function.

Together they make the reported path unrealizable: the check's failure arm
terminates, so control cannot continue to the deref with a null argument.

Every test builds the report events and the source file directly (no report
HTML, no PDG, no CFG cache), so the guards below are each pinned by a test that
the check stays silent — they are what keeps the conclusion sound:

    guard family (assert/DCHECK/FAISS_ASSERT excluded) · opposite polarity ·
    CHECK_EQ · non-pointer parameter · guard after the deref · guard in another
    function · entry assumption not first · non-null bug type · branch not taken
    · multi-line call · unreadable source · call-site count is corroboration
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_client.fp_analysis import entry_state as es

BUG_TYPE = "Dereference of null pointer"
DEREF_LINE = 3


# ─── builders ────────────────────────────────────────────────────────────────

def _write_source(tmp_path: Path, body: str, rel: str = "src/numbers.cc") -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _metadata(path: Path, *, function: str = "SimpleAtob",
              bug_file: str | None = None, bug_type: str = BUG_TYPE,
              bug_line: int = DEREF_LINE) -> SimpleNamespace:
    return SimpleNamespace(
        # A CSA build path (``/build/…``) that only resolves by suffix under
        # source_root — the shape the real reports carry.
        bug_file=bug_file or f"/build/out/{path.relative_to(path.parents[1])}",
        function_name=function,
        bug_type=bug_type,
        bug_line=bug_line,
    )


def _report(*, entry: str = "Assuming pointer value is null",
            var: str = "out", deref_line: int = DEREF_LINE,
            guard_line: int = 2, branch_at_guard: bool = True,
            entry_index: int = 0) -> SimpleNamespace:
    events = [
        {"path_number": 1, "line_number": 1, "description": entry},
        {"path_number": 2, "line_number": guard_line,
         "description": "Looping back to the head of the loop"},
        {"path_number": 3, "line_number": deref_line,
         "description": f"Dereference of null pointer (loaded from "
                        f"variable '{var}')", "event_kind": "report"},
    ]
    if branch_at_guard:
        events.insert(1, {
            "path_number": 2, "line_number": guard_line,
            "description": "Taking true branch", "branch_condition": True,
        })
    if entry_index == 1:
        events.insert(0, {"path_number": 1, "line_number": 1,
                          "description": "Error Start"})
    for i, ev in enumerate(events, 1):
        ev.setdefault("path_number", i)
    return SimpleNamespace(path_events=events, metadata=None)


def _parse(tmp_path: Path, body: str, *, report: SimpleNamespace | None = None,
           source_root: Path | None = None, **md_kwargs):
    path = _write_source(tmp_path, body)
    rep = report or _report()
    rep.metadata = _metadata(path, **md_kwargs)
    return rep, (tmp_path if source_root is None else source_root)


# The positive shape: absl's ``SimpleAtob``, whose ``ABSL_RAW_CHECK`` on the
# output parameter is unconditional and whose FATAL handler calls ``abort()``.
GOOD_SOURCE = """\
bool SimpleAtob(const char* str, int64_t* out) {
  ABSL_RAW_CHECK(out != nullptr, "Output pointer must not be null");
  *out = ParseTime(str);
  return true;
}
"""


# ─── the positive case ───────────────────────────────────────────────────────

class TestFires:
    def test_absl_raw_check_null_param(self, tmp_path):
        parsed, root = _parse(tmp_path, GOOD_SOURCE)
        f = es.detect_fabricated_entry_state(parsed, root)
        assert f is not None
        assert f.parameter == "out"
        assert f.guard_macro == "ABSL_RAW_CHECK"
        assert f.guard_line == 2
        assert f.deref_line == DEREF_LINE
        assert f.function == "SimpleAtob"
        assert f.kind == "fatal_guard_continued"

    def test_reason_names_the_check_and_the_parameter(self, tmp_path):
        parsed, root = _parse(tmp_path, GOOD_SOURCE)
        reason = es.detect_fabricated_entry_state(parsed, root).reason()
        assert "out" in reason
        assert "ABSL_RAW_CHECK" in reason
        assert "abort()" in reason          # the semantics, i.e. why it terminates
        assert "L2" in reason and "L3" in reason
        assert "不可实现" in reason

    def test_zero_call_sites_is_corroboration_in_the_reason(self, tmp_path):
        parsed, root = _parse(tmp_path, GOOD_SOURCE)
        f = es.detect_fabricated_entry_state(parsed, root)
        assert f.call_sites == 0
        assert "0 个真实调用点" in f.reason()

    def test_real_call_sites_are_reported_without_changing_the_finding(
            self, tmp_path):
        _write_source(tmp_path, GOOD_SOURCE)
        _write_source(tmp_path, "int main() { int64_t v; return SimpleAtob(\"1\", &v); }",
                      rel="main.cc")
        parsed, root = _parse(tmp_path, GOOD_SOURCE)
        f = es.detect_fabricated_entry_state(parsed, root)
        assert f is not None
        assert f.call_sites == 1
        assert "main.cc" in f.call_site_samples[0]
        assert "不依赖调用点数量" in f.reason()

    def test_unavailable_scan_is_not_a_zero_count(self, tmp_path):
        """A tree that cannot be scanned must read as unknown, never as 0."""
        path = _write_source(tmp_path, GOOD_SOURCE)
        parsed = _report()
        parsed.metadata = _metadata(path, bug_file=str(path))
        f = es.detect_fabricated_entry_state(parsed, None)
        assert f is not None
        assert f.call_sites is None
        assert "调用点扫描不可用" in f.reason()

    def test_faiss_throw_if_not_family(self, tmp_path):
        src = ("void read_index(Index* index, const char* fname) {\n"
               "  FAISS_THROW_IF_NOT(index != nullptr);\n"
               "  index->load(fname);\n"
               "}\n")
        rep = _report(var="index", guard_line=2, deref_line=3)
        parsed, root = _parse(
            tmp_path, src, report=rep, function="read_index", bug_line=3,
        )
        f = es.detect_fabricated_entry_state(parsed, root)
        assert f is not None and f.guard_macro == "FAISS_THROW_IF_NOT"

    def test_entry_assumption_may_follow_the_error_start_event(self, tmp_path):
        rep = _report(entry_index=1)
        parsed, root = _parse(tmp_path, GOOD_SOURCE, report=rep)
        assert es.detect_fabricated_entry_state(parsed, root) is not None


# ─── soundness guards: each of these must stay silent ────────────────────────

class TestSilent:
    def test_ndebug_family_is_excluded(self, tmp_path):
        """``assert``/``DCHECK``/``FAISS_ASSERT`` compile out under NDEBUG, so a
        null legitimately survives them — concluding FP from one is unsound."""
        for macro, body in (
            ("assert", "assert(out != nullptr);"),
            ("DCHECK", "DCHECK(out != nullptr);"),
            ("FAISS_ASSERT", "FAISS_ASSERT(out != nullptr);"),
        ):
            src = (f"bool SimpleAtob(const char* str, int64_t* out) {{\n"
                   f"  {body}\n  *out = 1;\n  return true;\n}}\n")
            parsed, root = _parse(tmp_path / macro, src)
            assert es.detect_fabricated_entry_state(parsed, root) is None, macro

    def test_two_operand_check_forms_are_skipped(self, tmp_path):
        """``CHECK_EQ``/``ABSL_CHECK_NE`` take two operands: "requires non-null"
        is not expressible from the first argument, so they are left alone."""
        for call in ('CHECK_EQ(out, nullptr);', 'ABSL_CHECK_NE(out, ptr);',
                     'CHECK_NOTNULL(out);'):
            src = (f"bool SimpleAtob(const char* str, int64_t* out) {{\n"
                   f"  {call}\n  *out = 1;\n  return true;\n}}\n")
            parsed, root = _parse(tmp_path / call[:9], src)
            assert es.detect_fabricated_entry_state(parsed, root) is None, call

    def test_opposite_polarity_is_silent(self, tmp_path):
        src = ("bool SimpleAtob(const char* str, int64_t* out) {\n"
               "  CHECK(out == nullptr);\n  *out = 1;\n  return true;\n}\n")
        parsed, root = _parse(tmp_path, src)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_compound_condition_is_silent(self, tmp_path):
        src = ("bool SimpleAtob(const char* str, int64_t* out) {\n"
               "  CHECK(out != nullptr && str != nullptr);\n"
               "  *out = 1;\n  return true;\n}\n")
        parsed, root = _parse(tmp_path, src)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_check_on_a_different_variable_is_silent(self, tmp_path):
        src = ("bool SimpleAtob(const char* str, int64_t* out) {\n"
               "  CHECK(str != nullptr);\n  *out = 1;\n  return true;\n}\n")
        parsed, root = _parse(tmp_path, src)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_value_parameter_is_silent(self, tmp_path):
        """A non-pointer parameter cannot be a null *dereference* target."""
        src = ("int SimpleAtob(const char* str, int64_t out) {\n"
               "  CHECK(out != 0);\n  return out;\n}\n")
        rep = _report(var="out")
        parsed, root = _parse(tmp_path, src, report=rep, bug_line=3)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_guard_after_the_deref_is_silent(self, tmp_path):
        src = ("bool SimpleAtob(const char* str, int64_t* out) {\n"
               "  *out = 1;\n"
               "  CHECK(out != nullptr);\n  return true;\n}\n")
        parsed, root = _parse(tmp_path, src, bug_line=2)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_deref_outside_the_function_is_silent(self, tmp_path):
        src = ("bool SimpleAtob(const char* str, int64_t* out) {\n"
               "  CHECK(out != nullptr);\n  return true;\n}\n"
               "int other(int64_t* out) {\n  return *out;\n}\n")
        parsed, root = _parse(tmp_path, src, bug_line=5)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_branch_not_recorded_as_taken_is_silent(self, tmp_path):
        """The report must itself record entering the fatal check's failure arm."""
        rep = _report(branch_at_guard=False)
        parsed, root = _parse(tmp_path, GOOD_SOURCE, report=rep)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_entry_assumption_must_be_the_reports_first_claim(self, tmp_path):
        rep = _report()
        for ev in rep.path_events[:2]:
            ev["description"] = "Taking true branch"  # no entry assumption
        parsed, root = _parse(tmp_path, GOOD_SOURCE, report=rep)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_non_null_bug_type_is_silent(self, tmp_path):
        parsed, root = _parse(tmp_path, GOOD_SOURCE,
                              bug_type="Memory leak")
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_report_variable_comes_from_the_report_event(self, tmp_path):
        """A report whose deref names a *member* yields no finding."""
        rep = _report(var="field")
        parsed, root = _parse(tmp_path, GOOD_SOURCE, report=rep)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_unreadable_source_is_silent(self, tmp_path):
        parsed, root = _parse(tmp_path, GOOD_SOURCE,
                              bug_file="/build/out/does/not/exist.cc")
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_multi_line_call_is_silent(self, tmp_path):
        """A macro call that does not close on its own line cannot be judged."""
        src = ("bool SimpleAtob(const char* str, int64_t* out) {\n"
               "  CHECK(out != nullptr,\n        \"long message\");\n"
               "  *out = 1;\n  return true;\n}\n")
        parsed, root = _parse(tmp_path, src)
        assert es.detect_fabricated_entry_state(parsed, root) is None

    def test_too_few_events_is_silent(self, tmp_path):
        path = _write_source(tmp_path, GOOD_SOURCE)
        parsed = SimpleNamespace(
            path_events=[{"path_number": 1, "line_number": 1,
                          "description": "Assuming pointer value is null"}],
            metadata=_metadata(path, bug_file=str(path)),
        )
        assert es.detect_fabricated_entry_state(parsed, None) is None

    def test_comment_lines_never_contribute_a_guard(self, tmp_path):
        src = ("bool SimpleAtob(const char* str, int64_t* out) {\n"
               "  // CHECK(out != nullptr);\n"
               "  *out = 1;\n  return true;\n}\n")
        parsed, root = _parse(tmp_path, src)
        assert es.detect_fabricated_entry_state(parsed, root) is None


# ─── lexical helpers ────────────────────────────────────────────────────────

class TestRequiresNonNull:
    @pytest.mark.parametrize("cond", [
        "out != nullptr", " nullptr != out ", "out != NULL", "NULL != out",
        "out != 0", "0 != out", "!!out", "out", "((out != nullptr))",
        "out!=nullptr",
    ])
    def test_accepts(self, cond):
        assert es._requires_non_null(cond, "out") is True

    @pytest.mark.parametrize("cond", [
        "out == nullptr", "nullptr == out", "!out", "!(out != nullptr)",
        "out && x", "out != nullptr && str != nullptr", "", "other != nullptr",
        "out != other", "out->next != nullptr", "output != nullptr",
    ])
    def test_rejects(self, cond):
        assert es._requires_non_null(cond, "out") is False

    def test_prefix_var_is_not_a_match(self):
        """``output != nullptr`` must not satisfy a check on ``out`` (the last
        case of ``test_rejects`` above, kept named for the reason)."""
        assert es._requires_non_null("output != nullptr", "out") is False


class TestCallSiteClassify:
    def test_comment_mention_is_not_a_call(self):
        line = "// SimpleAtob() is a helper"
        idx = line.index("SimpleAtob")
        assert es._classify_occurrence(line, idx) == "call"
        # …and the scanner skips such lines before ever classifying them.
        assert line.lstrip().startswith(es._COMMENT_PREFIXES)

    def test_bare_statement_call_is_a_call(self):
        line = "  SimpleAtob(str, &out);"
        assert es._classify_occurrence(line, line.index("SimpleAtob")) == "call"

    def test_definition_and_declaration(self):
        assert es._classify_occurrence(
            "bool SimpleAtob(const char* s, int64_t* o) {",
            len("bool ")) == "definition"
        assert es._classify_occurrence(
            "bool SimpleAtob(const char* s, int64_t* o);", len("bool ")) == "declaration"

    def test_expression_position(self):
        assert es._classify_occurrence(
            "return SimpleAtob(s, o);", len("return ")) == "call"
        assert es._classify_occurrence(
            "if (SimpleAtob(s, o)) {", len("if (")) == "call"


class TestFunctionRange:
    def test_brace_on_next_line(self):
        lines = ["bool f(int* p)", "{", "  *p = 1;", "}"]

        assert es._find_function_range(lines, "f") == (1, 4, "int* p")

    def test_forward_declaration_is_skipped_for_the_definition(self):
        lines = ["bool f(int* p);", "bool f(int* p) {", "  return true;", "}"]
        assert es._find_function_range(lines, "f") == (2, 4, "int* p")

    def test_no_definition(self):
        assert es._find_function_range(["int x = 1;"], "f") is None
