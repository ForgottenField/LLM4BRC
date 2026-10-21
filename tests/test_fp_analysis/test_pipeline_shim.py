"""The compatibility block in ``run_path_selection.py`` is a contract, not dead code.

The implementation moved into :mod:`llm_client.fp_analysis.pipeline`, but the
harness tools (``tools/{replay_steps1to6,abc_compare,probe_report_consistency}.py``)
and the CLI tests import their names **from the script**, and did so before the
move.  Keeping the script's re-export block is what made the move touch no import
site — and that is exactly what makes it dangerous: a block that nothing in the
repository itself imports looks like leftovers, and "cleaning it up" would break
three evaluation tools with no test failing.

Two assertions keep it honest:

* every alias is the **same object** as the pipeline module's attribute, so the
  block cannot rot into a second, divergent definition;
* the alias set **covers** what ``tools/`` and ``tests/`` actually pull out of
  the script, derived from their source rather than from a hand-kept list — a
  list would be the same kind of artefact this test exists to distrust.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_path_selection as rps  # noqa: E402


#: Every name the block re-exports, and the module that really defines it.
#: ``run_path_selection`` itself is the *owner* of ``main`` and of the three
#: imports at the top of the block (``CFGCacheSet`` and friends are imported
#: there directly, not re-exported through the pipeline).
ALIASES: dict[str, str] = {
    "ReportFrontend": "llm_client.report_frontend",
    "ReportRef": "llm_client.report_frontend",
    "describe_adapters_table": "llm_client.report_frontend.adapters.base",
    "CFGCacheSet": "llm_client.fp_analysis.cfg_cache_set",
    "_resolve_source_file_for_cfg": "llm_client.fp_analysis.cfg_feasibility",
    "build_dense_path_annotations": "llm_client.fp_analysis.path_seed_extractor",
    "DEFAULT_PROJECTS_DIR": "llm_client.fp_analysis.pipeline.layout",
    "code_model_report": "llm_client.fp_analysis.pipeline.artifacts",
    "report_fields": "llm_client.fp_analysis.pipeline.artifacts",
    "report_key": "llm_client.fp_analysis.pipeline.artifacts",
    "report_out_path": "llm_client.fp_analysis.pipeline.artifacts",
    "_collect_reports": "llm_client.fp_analysis.pipeline.batch",
    "_is_report_file": "llm_client.fp_analysis.pipeline.batch",
    "run_batch": "llm_client.fp_analysis.pipeline.batch",
    "_check_pdg_artifact_version": "llm_client.fp_analysis.pipeline.project_paths",
    "_infer_source_root": "llm_client.fp_analysis.pipeline.project_paths",
    "_resolve_paths": "llm_client.fp_analysis.pipeline.project_paths",
    "load_cfg_cache": "llm_client.fp_analysis.pipeline.project_paths",
    "load_sdg": "llm_client.fp_analysis.pipeline.project_paths",
    "process_report": "llm_client.fp_analysis.pipeline.runner",
    "_NO_PATH_STEP_KEYWORD": "llm_client.fp_analysis.pipeline.selection",
    "_no_path_verdict": "llm_client.fp_analysis.pipeline.selection",
    "_infer_entry_function": "llm_client.fp_analysis.pipeline.stages",
    "build_slice_mask": "llm_client.fp_analysis.pipeline.stages",
    "parse_report": "llm_client.fp_analysis.pipeline.stages",
    "run_branch_filter": "llm_client.fp_analysis.pipeline.stages",
    "segment_path": "llm_client.fp_analysis.pipeline.stages",
}


class TestTheAliasesResolveToTheRealDefinitions:
    @pytest.mark.parametrize("name,owner", sorted(ALIASES.items()))
    def test_identity(self, name: str, owner: str) -> None:
        module = importlib.import_module(owner)
        assert getattr(rps, name, None) is not None, (
            f"run_path_selection.{name} is gone — the harness tools import it "
            f"from there"
        )
        assert getattr(rps, name) is getattr(module, name), (
            f"run_path_selection.{name} is a second definition, not the one in "
            f"{owner}"
        )

    def test_the_block_is_complete_for_the_tools(self) -> None:
        """The external names, derived from the consumers' own source."""
        wanted = set()
        for path in sorted((REPO_ROOT / "tools").glob("*.py")) + sorted(
            (REPO_ROOT / "tests").rglob("test_*.py")
        ):
            wanted |= _names_imported_from_the_script(path)
        missing = sorted(n for n in wanted if not hasattr(rps, n))
        assert not missing, (
            f"{missing} are imported from run_path_selection but the "
            f"compatibility block no longer provides them"
        )
        # …and nothing obviously unrelated is being re-exported.
        assert wanted <= set(ALIASES) | {"main"}


def _names_imported_from_the_script(path: Path) -> set[str]:
    """``from run_path_selection import X`` and ``<alias>.X`` for its module alias."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # pragma: no cover — a broken tool would fail elsewhere
        return set()
    aliases: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "run_path_selection":
                    aliases.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "run_path_selection":
            for a in node.names:
                names.add(a.asname or a.name)
    if not aliases:
        return names
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            names.add(node.attr)
    return names


class TestTheMonkeypatchSeamsAreWhereTheTestsSay:
    """A seam is a *name lookup*, and it moves with the function that makes it.

    ``test_cli_integration`` and ``test_process_report_characterization`` patch
    module attributes; a patch on a module that does not read the name binds a
    second, unread name and the real call goes through.  These two assertions
    state where the call sites are, so a future move fails here — with a message
    naming the seam — rather than by quietly calling DeepSeek in CI.
    """

    def test_the_selector_seam_is_in_the_runner(self) -> None:
        from pipeline import runner

        source = (REPO_ROOT / "src/llm_client/fp_analysis/pipeline/runner.py").read_text()
        # The call sites, not the import: `_resolve_paths`/`parse_report` are
        # bare names inside `process_report`, so they resolve in `runner`.
        assert "parse_report(" in source
        assert hasattr(runner, "_resolve_paths") and hasattr(runner, "parse_report")

    def test_the_attempt_loop_seams_are_in_exploration(self) -> None:
        from pipeline import runner
        from pipeline import exploration

        assert hasattr(exploration, "run_path_selection")
        assert hasattr(exploration, "_verify_path")
        # …and have NOT been left behind as dead re-exports in `runner`, which
        # would invite patching the wrong module.
        assert not hasattr(runner, "run_path_selection")
        assert not hasattr(runner, "_verify_path")
