"""Tests for the simplified on-demand function summary models."""

from __future__ import annotations

from llm_client.fp_analysis.function_summary import (
    CachedFunctionData,
    FunctionSummary,
    ReturnValueConstraint,
    normalize_bug_type,
)


class TestFunctionSummary:
    def test_minimal(self):
        s = FunctionSummary(
            function_name="my_func",
            signature="void my_func(int x)",
        )
        assert s.function_name == "my_func"
        assert s.signature == "void my_func(int x)"
        assert s.file_path == ""
        assert s.body_available is False
        assert s.is_best_guess is False
        assert s.is_library is False
        assert s.preconditions == []
        assert s.analysis == ""
        assert s.return_value_constraints is None

    def test_with_all_fields(self):
        rvc = ReturnValueConstraint(
            return_value_semantics="0 = success, -1 = error",
            success_conditions=["arg > 0"],
            failure_conditions=["arg <= 0"],
            failure_nature="deterministic",
        )
        s = FunctionSummary(
            function_name="check_value",
            signature="int check_value(int arg)",
            file_path="src/check.c",
            body_available=True,
            body_snippet="int check_value(int arg) { return arg > 0 ? 0 : -1; }",
            is_best_guess=False,
            is_library=True,
            preconditions=["arg != NULL"],
            analysis="Validates input argument",
            return_value_constraints=rvc,
        )
        assert s.is_library is True
        assert s.body_available is True
        assert s.return_value_constraints is not None
        assert s.return_value_constraints.return_value_semantics == "0 = success, -1 = error"

    def test_format_for_llm(self):
        rvc = ReturnValueConstraint(
            return_value_semantics="0 = success",
            success_conditions=["init_done == true"],
            failure_conditions=["init_done == false"],
            failure_nature="deterministic",
        )
        s = FunctionSummary(
            function_name="do_work",
            signature="int do_work(int flag)",
            body_available=True,
            is_library=False,
            preconditions=["flag > 0"],
            analysis="Performs work when flag is set",
            return_value_constraints=rvc,
        )
        result = s.format_for_llm()
        assert "do_work" in result
        assert "(body available)" in result
        assert "Preconditions" in result
        assert "flag > 0" in result
        assert "Return Value Constraints" in result
        assert "Deterministic" in result
        assert "Performs work" in result

    def test_format_for_llm_best_guess(self):
        s = FunctionSummary(
            function_name="external_func",
            signature="int external_func(void* p)",
            file_path="lib/ext.c",
            body_available=False,
            is_best_guess=True,
            analysis="Best guess based on library docs",
        )
        result = s.format_for_llm()
        assert "BEST GUESS" in result

    def test_format_for_llm_library_flag(self):
        s = FunctionSummary(
            function_name="lib_func",
            signature="void lib_func()",
            is_library=True,
            analysis="External library function",
        )
        result = s.format_for_llm()
        assert "[LIBRARY]" in result


class TestReturnValueConstraint:
    def test_minimal(self):
        rvc = ReturnValueConstraint(
            return_value_semantics="0 = ok",
        )
        assert rvc.return_value_semantics == "0 = ok"
        assert rvc.failure_nature == "deterministic"

    def test_failure_conditions(self):
        rvc = ReturnValueConstraint(
            return_value_semantics="0 = success, -1 = error",
            success_conditions=["queue not full"],
            failure_conditions=["queue is full"],
            failure_nature="deterministic",
        )
        text = rvc.format_for_llm()
        assert "deterministic" in text.lower()
        assert "queue is full" in text

    def test_non_deterministic(self):
        rvc = ReturnValueConstraint(
            return_value_semantics="ptr or NULL",
            success_conditions=[],
            failure_conditions=[],
            failure_nature="non_deterministic",
        )
        text = rvc.format_for_llm()
        assert "Non-deterministic" in text


class TestCachedFunctionData:
    def test_minimal(self):
        data = CachedFunctionData(
            function_name="my_func", project="test_project"
        )
        assert data.function_name == "my_func"
        assert data.project == "test_project"
        assert data.summaries == {}
        assert data.is_library is False

    def test_with_library_flag(self):
        data = CachedFunctionData(
            function_name="lib_func",
            project="test_project",
            is_library=True,
        )
        assert data.is_library is True

    def test_with_summaries(self):
        data = CachedFunctionData(
            function_name="my_func",
            project="test_project",
            file_path="src/my.c",
            signature="void my_func(void)",
            summaries={
                "memory_leak": {"analysis": "test analysis"},
            },
        )
        assert "memory_leak" in data.summaries
        assert data.summaries["memory_leak"]["analysis"] == "test analysis"


class TestNormalizeBugType:
    def test_already_canonical(self):
        assert normalize_bug_type("memory_leak") == "memory_leak"

    def test_space_separated(self):
        assert normalize_bug_type("Memory leak") == "memory_leak"

    def test_dash_separated(self):
        assert normalize_bug_type("memory-leak") == "memory_leak"

    def test_case_insensitive(self):
        assert normalize_bug_type("MEMORY LEAK") == "memory_leak"

    def test_mixed_spacing(self):
        assert normalize_bug_type("  Memory   leak  ") == "memory_leak"

    def test_full_csa_name(self):
        result = normalize_bug_type(
            "Argument with Nonnull Attribute passed Null"
        )
        assert result == "argument_with_nonnull_attribute_passed_null"

    def test_null_pointer_variants(self):
        assert normalize_bug_type("null pointer") == "dereference_of_null_pointer"
        assert normalize_bug_type("NULL_POINTER") == "dereference_of_null_pointer"
        assert normalize_bug_type("Null") == "dereference_of_null_pointer"

    def test_unknown_type_underscored(self):
        result = normalize_bug_type("Some weird bug type")
        assert result == "some_weird_bug_type"

    def test_dots_replaced(self):
        result = normalize_bug_type("some.bug.type")
        assert result == "some_bug_type"
