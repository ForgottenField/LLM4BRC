"""Global report data-constraint extraction and solving."""

from fp_analysis.constraint_model import (
    build_global_constraint_bundle,
    solve_immediate_conflicts,
    validate_solution,
)
from fp_analysis.models import ParsedReport, ReportMetadata


def _report(tmp_path):
    caller = tmp_path / "FormatterTest.cpp"
    caller.write_text(
        "void test() {\n"
        "  auto out = sformat(\"{:5d}\", value);\n"
        "}\n",
        encoding="utf-8",
    )
    parser = tmp_path / "Format.cpp"
    parser.write_text(
        "void parse(const char* p) {\n"
        "  if (*p == ',') {\n"
        "    thousandsSeparator = true;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "FormatArg.h").write_text(
        "FormatArg() : thousandsSeparator(false) {}\n",
        encoding="utf-8",
    )
    return ParsedReport(
        metadata=ReportMetadata(
            bug_type="assigned value is garbage or undefined",
            bug_file=str(parser),
            bug_line=3,
            function_name="parse",
            path_evidence="trace",
        ),
        source_snippets={},
        raw_html_path=str(tmp_path / "report.html"),
        path_events=[
            {
                "path_number": 1,
                "event_kind": "event",
                "description": "Calling 'sformat'",
                "line_number": 2,
                "file_path": str(caller),
                "is_calling_event": True,
            },
            {
                "path_number": 2,
                "event_kind": "event",
                "description": "Assuming field 'thousandsSeparator' is true",
                "line_number": 3,
                "file_path": str(parser),
            },
        ],
    )


def test_boolean_assumption_is_extracted(tmp_path):
    bundle = build_global_constraint_bundle(_report(tmp_path), tmp_path)

    report_fact = next(
        fact for fact in bundle.facts
        if fact.subject == "thousandsSeparator"
    )
    assert report_fact.operator == "=="
    assert report_fact.value == "true"
    assert report_fact.provenance == "report_explicit"


def test_call_literal_and_source_effects_are_unified(tmp_path):
    bundle = build_global_constraint_bundle(_report(tmp_path), tmp_path)

    literal = next(
        fact for fact in bundle.facts
        if fact.provenance == "source_literal"
    )
    assert literal.value == "{:5d}"
    effects = {
        (effect.value, effect.condition)
        for effect in bundle.effects
        if effect.subject == "thousandsSeparator"
    }
    assert ("false", "unconditional") in effects
    assert ("true", "if (*p == ',') {") in effects


def test_guarded_literal_flag_detects_unsat(tmp_path):
    solution = solve_immediate_conflicts(
        build_global_constraint_bundle(_report(tmp_path), tmp_path)
    )

    assert solution["status"] == "unsat"
    conflict = solution["contradictions"][0]
    assert conflict["subject"] == "thousandsSeparator"
    assert {"true", "false"} == set(conflict["values"])
    assert len(conflict["basis_ids"]) == 4


def test_llm_unsat_requires_report_and_source_evidence(tmp_path):
    bundle = build_global_constraint_bundle(_report(tmp_path), tmp_path)
    report_fact = next(
        fact for fact in bundle.facts if fact.provenance == "report_explicit"
    )
    literal = next(
        fact for fact in bundle.facts if fact.provenance == "source_literal"
    )
    effect_ids = [effect.effect_id for effect in bundle.effects]

    accepted = validate_solution({
        "status": "unsat",
        "contradictions": [{
            "subject": "thousandsSeparator",
            "values": ["true", "false"],
            "basis_ids": [
                report_fact.fact_id, literal.fact_id, *effect_ids,
            ],
            "reason": "comma-only flag conflicts with the supplied format literal",
        }],
    }, bundle)
    rejected = validate_solution({
        "status": "unsat",
        "contradictions": [{
            "subject": "thousandsSeparator",
            "values": ["true", "false"],
            "basis_ids": ["F1", "DOES_NOT_EXIST"],
            "reason": "invented evidence",
        }],
    }, bundle)

    assert accepted["status"] == "unsat"
    assert len(accepted["contradictions"]) == 1
    assert rejected["status"] == "unknown"
    assert rejected["contradictions"] == []
