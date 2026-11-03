"""Tests for the independent Tree-sitter AST/PDG-Lite backend."""

from __future__ import annotations

import json
from types import SimpleNamespace

from pipeline.lite_graph import build_lite_graph
from tree_sitter_backend import CompileDatabase, TreeSitterBackend


SOURCE = """\
#include "dep.h"
static int helper(int value) { return value + 1; }
int target(int input) {
  if (input > 0) {
    int adjusted = helper(input);
    return adjusted;
  }
  return input;
}
"""


def _write_compile_db(tmp_path, source, *extra_args):
    (tmp_path / "dep.h").write_text("")
    database = tmp_path / "compile_commands.json"
    database.write_text(json.dumps([{
        "directory": str(tmp_path),
        "file": str(source),
        "arguments": [
            "g++", "-std=c++17", "-I", str(tmp_path), *extra_args,
            "-c", str(source), "-o", str(tmp_path / "sample.o"),
        ],
    }]))
    return database


def test_builds_functions_calls_branches_and_data_edges(tmp_path) -> None:
    source = tmp_path / "sample.cpp"
    source.write_text(SOURCE)

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(g for g in index.graphs.values() if g.function == "target")

    assert any(node.kind == "control" and "input > 0" in node.text for node in graph.nodes)
    assert any(edge.kind == "control" for edge in graph.edges)
    assert any(edge.kind == "data" for edge in graph.edges)
    assert [call.callee for call in index.calls] == ["helper"]
    assert index.includes[str(source)] == ["dep.h"]


def test_renders_compact_evidence(tmp_path) -> None:
    source = tmp_path / "sample.cpp"
    source.write_text(SOURCE)

    index = TreeSitterBackend().build_index([source], tmp_path)
    evidence = index.render_evidence({"target"})

    assert "Tree-sitter AST / PDG-Lite" in evidence
    assert "target" in evidence
    assert "helper(input)" in evidence
    assert "reaching-definition dependencies" in evidence


def test_parse_errors_do_not_abort_indexing(tmp_path) -> None:
    source = tmp_path / "broken.cpp"
    source.write_text("int broken( {")

    index = TreeSitterBackend().build_index([source], tmp_path)

    assert index.files[str(source)]["has_error"] is True
    assert index.errors[0]["kind"] == "error"
    assert index.errors[0]["line"] == "1"


def test_indexes_c_sources_with_the_c_grammar(tmp_path) -> None:
    source = tmp_path / "sample.c"
    source.write_text(
        "static int bump(int value) { return value + 1; }\n"
        "int read_value(int *value) {\n"
        "  if (!value) return 0;\n"
        "  return bump(*value);\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)

    assert next(iter(index.files.values()))["language"] == "c"
    assert {fn.name for fn in index.functions.values()} == {"bump", "read_value"}
    assert any(call.callee == "bump" for call in index.calls)


def test_compile_database_selects_the_configured_preprocessor_branch(tmp_path) -> None:
    source = tmp_path / "configured.cpp"
    source.write_text(
        "#define RESULT() 1\n"
        "#ifdef USE_FIRST\n"
        "int selected() { return RESULT(); }\n"
        "#else\n"
        "int selected() { return 2; }\n"
        "#endif\n"
    )
    database = _write_compile_db(tmp_path, source, "-DUSE_FIRST")

    index = TreeSitterBackend().build_index(
        [source], tmp_path, compile_commands=database,
    )

    graph = next(iter(index.graphs.values()))
    assert any("return 1;" in node.text for node in graph.nodes)
    assert not any("return 2;" in node.text for node in graph.nodes)
    assert index.preprocessor_mode == "configured"
    assert graph.start_line == 3
    assert not index.errors


def test_compile_database_accepts_standard_command_strings(tmp_path) -> None:
    source = tmp_path / "command.cpp"
    source.write_text("int value() { return 7; }\n")
    database = tmp_path / "compile_commands.json"
    database.write_text(json.dumps([{
        "directory": str(tmp_path),
        "file": "command.cpp",
        "command": (
            f"g++ -std=c++17 -I{tmp_path} -c command.cpp "
            f"-o {tmp_path / 'command.o'}"
        ),
    }]))

    loaded = CompileDatabase.load(database)
    prepared = loaded.preprocess(source)
    index = TreeSitterBackend().build_index(
        [source], tmp_path, compile_commands=database,
    )

    assert "int value()" in prepared.text
    assert next(iter(index.functions.values())).name == "value"


def test_if_else_builds_branch_join_and_reaching_definitions(tmp_path) -> None:
    source = tmp_path / "branch.cpp"
    source.write_text(
        "int choose(int flag) {\n"
        "  int x = 1;\n"
        "  if (flag) { x = 2; } else { x = 3; }\n"
        "  return x;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))
    by_id = {node.node_id: node for node in graph.nodes}
    return_node = next(node for node in graph.nodes if node.text == "return x;")
    reaching = {
        by_id[edge.source].text
        for edge in graph.edges
        if edge.kind == "data"
        and edge.target == return_node.node_id
        and edge.variable == "x"
    }

    assert reaching == {"x = 2;", "x = 3;"}
    true_node = next(node for node in graph.nodes if node.text == "x = 2;")
    false_node = next(node for node in graph.nodes if node.text == "x = 3;")
    assert not any(
        edge.source_block == true_node.block_id
        and edge.target_block == false_node.block_id
        for edge in graph.cfg_edges
    )
    assert any(edge.kind == "true" for edge in graph.cfg_edges)
    assert any(edge.kind == "false" for edge in graph.cfg_edges)
    assert any(edge.kind == "join" for edge in graph.cfg_edges)
    control_targets = {
        edge.target for edge in graph.edges if edge.kind == "control"
    }
    assert return_node.node_id not in control_targets


def test_loop_cfg_and_reaching_definitions_reach_the_return(tmp_path) -> None:
    source = tmp_path / "loop.cpp"
    source.write_text(
        "int countdown(int n) {\n"
        "  int x = 0;\n"
        "  while (n > 0) { x = x + 1; n = n - 1; }\n"
        "  return x;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))
    by_id = {node.node_id: node for node in graph.nodes}
    return_node = next(node for node in graph.nodes if node.text == "return x;")
    reaching = {
        by_id[edge.source].text
        for edge in graph.edges
        if edge.kind == "data"
        and edge.target == return_node.node_id
        and edge.variable == "x"
    }

    assert reaching == {"int x = 0;", "x = x + 1;"}
    assert any(edge.kind == "loop_back" for edge in graph.cfg_edges)


def test_switch_break_and_fallthrough_update_reaching_definitions(tmp_path) -> None:
    source = tmp_path / "switch.cpp"
    source.write_text(
        "int select_value(int code) {\n"
        "  int value = 0;\n"
        "  switch (code) {\n"
        "  case 1: value = 1; break;\n"
        "  case 2: value = 2;\n"
        "  default: value = 3;\n"
        "  }\n"
        "  return value;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))
    by_id = {node.node_id: node for node in graph.nodes}
    return_node = next(node for node in graph.nodes if node.text == "return value;")
    reaching = {
        by_id[edge.source].text
        for edge in graph.edges
        if edge.kind == "data"
        and edge.target == return_node.node_id
        and edge.variable == "value"
    }

    assert reaching == {"value = 1;", "value = 3;"}
    assert any(edge.kind == "break" for edge in graph.cfg_edges)
    assert any(edge.kind == "switch_fallthrough" for edge in graph.cfg_edges)


def test_compound_assignment_is_both_a_definition_and_use(tmp_path) -> None:
    source = tmp_path / "compound.cpp"
    source.write_text(
        "int increment(int value) {\n"
        "  value += 2;\n"
        "  return value;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))
    assignment = next(node for node in graph.nodes if node.text == "value += 2;")

    assert assignment.defines == ("value",)
    assert assignment.uses == ("value",)


def test_single_statement_if_and_for_update_are_part_of_the_cfg(tmp_path) -> None:
    source = tmp_path / "control.cpp"
    source.write_text(
        "int adjust(int flag, int limit) {\n"
        "  int value = 0;\n"
        "  if (flag) value = 1;\n"
        "  for (int i = 0; i < limit; ++i) value += i;\n"
        "  return value;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))

    assert any(node.text == "value = 1;" for node in graph.nodes)
    update = next(node for node in graph.nodes if node.text == "++i")
    assert update.defines == ("i",)
    assert update.uses == ("i",)
    assert any(edge.kind == "loop_update" for edge in graph.cfg_edges)
    assert any(
        edge.source_block == update.block_id
        and edge.kind == "loop_back"
        for edge in graph.cfg_edges
    )


def test_code_after_terminating_if_branches_is_unreachable(tmp_path) -> None:
    source = tmp_path / "unreachable.cpp"
    source.write_text(
        "int choose(int flag) {\n"
        "  if (flag) return 1;\n"
        "  else return 2;\n"
        "  return 3;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))
    unreachable = next(node for node in graph.nodes if node.text == "return 3;")

    assert not any(
        edge.target_block == unreachable.block_id
        for edge in graph.cfg_edges
    )


def test_single_statement_do_while_keeps_its_body(tmp_path) -> None:
    source = tmp_path / "do_while.cpp"
    source.write_text(
        "int advance(int value) {\n"
        "  do value += 1; while (value < 3);\n"
        "  return value;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))

    assert any(node.text == "value += 1;" for node in graph.nodes)
    assert any(edge.kind == "loop_test" for edge in graph.cfg_edges)
    assert any(
        edge.kind == "true" and edge.label == "loop"
        for edge in graph.cfg_edges
    )


def test_range_for_defines_its_loop_variable(tmp_path) -> None:
    source = tmp_path / "range_for.cpp"
    source.write_text(
        "int sum_values(int *values) {\n"
        "  int total = 0;\n"
        "  for (int value : values) total += value;\n"
        "  return total;\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)
    graph = next(iter(index.graphs.values()))
    control = next(node for node in graph.nodes if node.kind == "control")

    assert "value" in control.defines
    assert "values" in control.uses
    assert "return total" not in control.text


def test_language_casts_are_not_reported_as_function_calls(tmp_path) -> None:
    source = tmp_path / "cast.cpp"
    source.write_text(
        "int *convert(void *value) {\n"
        "  return reinterpret_cast<int *>(value);\n"
        "}\n"
    )

    index = TreeSitterBackend().build_index([source], tmp_path)

    assert not any(
        call.callee.startswith("reinterpret_cast")
        for call in index.calls
    )


def test_pipeline_adapter_builds_advisory_evidence(tmp_path) -> None:
    source = tmp_path / "sample.cpp"
    source.write_text(SOURCE)
    database = _write_compile_db(tmp_path, source)
    resolution = SimpleNamespace(local=source)
    reference = SimpleNamespace(resolution=resolution)
    finding = SimpleNamespace(
        location=SimpleNamespace(file=str(source), line=3),
        function="target",
        steps=[],
    )
    loaded = SimpleNamespace(
        finding=finding,
        code=SimpleNamespace(references=lambda _finding: [reference]),
    )

    evidence = build_lite_graph(loaded, tmp_path)

    assert evidence.enabled
    assert evidence.error == ""
    assert evidence.summary["parsed_functions"] == 2
    assert evidence.available
    assert evidence.summary["compile_commands"] == str(database.resolve())
    assert evidence.summary["preprocessor_mode"] == "configured"
    assert evidence.sdg.get_pdg("target") is not None
    assert evidence.to_dict()["precision"] == "syntax"
    assert "Tree-sitter AST / PDG-Lite" in evidence.prompt_text


def test_pipeline_requires_an_entry_for_every_report_source(tmp_path) -> None:
    source = tmp_path / "sample.cpp"
    source.write_text("int target(int input) { return input; }\n")
    other = tmp_path / "other.cpp"
    other.write_text("int other() { return 0; }\n")
    _write_compile_db(tmp_path, other)
    resolution = SimpleNamespace(local=source)
    reference = SimpleNamespace(resolution=resolution)
    finding = SimpleNamespace(
        location=SimpleNamespace(file=str(source), line=1),
        function="target",
        steps=[],
    )
    loaded = SimpleNamespace(
        finding=finding,
        code=SimpleNamespace(references=lambda _finding: [reference]),
    )

    evidence = build_lite_graph(loaded, tmp_path)

    assert not evidence.available
    assert "no compile database entry" in evidence.error
