"""Smoke tests for the canonical package layout and import boundaries."""

from __future__ import annotations

import importlib
import json


def test_canonical_packages_import() -> None:
    for module_name in (
        "csa_analysis",
        "fp_analysis",
        "llm_client",
        "pipeline",
        "report_frontend",
    ):
        assert importlib.import_module(module_name)


def test_repository_root_is_resolved_from_actual_layout() -> None:
    from pipeline.layout import REPO_ROOT

    assert (REPO_ROOT / "run_path_selection.py").is_file()
    assert (REPO_ROOT / "src" / "pipeline" / "layout.py").is_file()


def test_pipeline_submodules_do_not_depend_on_package_initialization_order() -> None:
    import pipeline.batch  # noqa: F401
    import pipeline.exploration  # noqa: F401
    import pipeline.runner  # noqa: F401
    import pipeline.selection  # noqa: F401


def test_verdict_vocabulary_serializes_as_existing_strings() -> None:
    from pipeline.outcomes import Verdict

    assert Verdict.coerce("fp") is Verdict.FP
    assert json.loads(json.dumps({"verdict": Verdict.UNKNOWN})) == {
        "verdict": "UNKNOWN",
    }
