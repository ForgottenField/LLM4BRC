"""The pipeline runner: the ten stages as one program.

``fp_analysis/`` holds the *engine* — extractors, checkers, identifiers, models
— one flat module per capability.  This subpackage holds the *application* that
drives them: it knows the CLI's arguments, the console's output, the result
JSON's shape and where the repository keeps ``pdg_<project>.json``,
``cfg_cache/<project>/`` and ``output/``.  That is a different altitude, and
mixing the two made "which of these 37 modules is the entry point?" a question
you had to answer by reading them.

The layers, and why the boundary is where it is:

``layout``
    The repository-root arithmetic and the shared logger — both singletons
    (see that module's docstring for why the logger is deliberately not
    ``__name__``).
``models``
    The dataclasses the stage functions hand each other.
``artifacts``
    Result identity, output path, the ``code_model`` block, the one writer.
``project_paths``
    Turning a CLI invocation into a resolved project (stage 1's assembly half).
``stages``
    The canonical stage numbering and the deterministic drivers, stages 1-5.
``selection``
    Stage 9's selection mechanism and stage 10's verification, plus the
    verdict prose.
``exploration``
    The attempt loop and the post-loop aggregation — the only code that can
    decide a report-level FP without a per-attempt conclusion.
``runner``
    ``process_report`` — one work item, ten stages, one verdict — and the
    ``_stageN_*`` adapters it reads as ten calls.  Each adapter's body is the
    old function's corresponding line range, moved verbatim; ``_stage6_gate``
    and ``_stage7_gate`` are the two that may return the early FP result
    themselves, the way the old function's two early ``return``s did.
``batch``
    Work-item collection and the batch driver.

**One-way dependency: the package never imports the entry script.**  It could
not do so usefully anyway — ``run_path_selection.py`` lives at the repository
root, outside ``src/``, so it is not importable under ``PYTHONPATH=src``.  The
script imports *this*, and keeps a compatibility block of re-exports so the
harness tools and the CLI tests that import from ``run_path_selection`` keep
working (see the script's own comment, and
``tests/test_fp_analysis/test_pipeline_shim.py``, which holds that block to the
consumers' own source).

The pipeline used to live entirely in that script, ``process_report`` alone
being 1123 lines; this package is that code, moved — and then split, so the
1123 lines now read as the ten stages ``docs/pipeline-stages.md`` describes.
Two properties of the move are load-bearing and are stated where the code is:
the stage bodies are verbatim (see ``runner``'s module docstring on ``t0``), and
the attempt loop aliases its inputs to the state carriers and writes them back
once rather than reading attributes (see ``exploration``'s).
"""

from __future__ import annotations

from llm_client.fp_analysis.pipeline.artifacts import (
    code_model_report,
    report_fields,
    report_key,
    report_out_path,
)
from llm_client.fp_analysis.pipeline.batch import _collect_reports, run_batch
from llm_client.fp_analysis.pipeline.runner import process_report
from llm_client.fp_analysis.pipeline.stages import stage_label, stage_step

__all__ = [
    "_collect_reports",
    "code_model_report",
    "process_report",
    "report_fields",
    "report_key",
    "report_out_path",
    "run_batch",
    "stage_label",
    "stage_step",
]
