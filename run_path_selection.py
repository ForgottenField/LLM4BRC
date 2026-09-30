#!/usr/bin/env python3
"""End-to-end path selection pipeline — the CLI entry point.

Runs the complete deterministic pipeline (PDG slicing, CFG branch extraction,
conflict checking) followed by the segment-by-segment path selection algorithm.
Everything runs by default — there are no mode switches to pass:

- Path selection is conflict-driven reselection (the only mechanism; the legacy
  retry path is gone).
- After a TP verdict the POC is generated, LLM-audited and simulated, and the
  feedback loop can settle the verdict as FP or UNKNOWN.

**This file is only the runner.**  The implementation lives in
``llm_client.fp_analysis.pipeline`` (see that package's ``__init__`` for the
layer map); this module parses the arguments, loads ``.env``, configures
logging, resolves every work item up front so a misinvocation fails in seconds,
and hands the list to the batch driver.  The compatibility block below
re-exports the names the harness tools (``tools/``) and the CLI tests import
from here — those imports are unchanged by the move, on purpose.

Usage:
    python3 run_path_selection.py --report <path/to/report.html>
    python3 run_path_selection.py --source-root ~/csa_reports/project/faiss --report <path>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Load .env for DEEPSEEK_API_KEY
_HERE = Path(__file__).resolve().parent
_env_path = _HERE / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("path_selection")

HERE = _HERE

# ─── Compatibility re-exports ────────────────────────────────────────────────────
# The implementation lives in llm_client.fp_analysis.pipeline.*.  These names are
# re-bound here because the harness tools (tools/abc_compare.py,
# tools/replay_steps1to6.py, tools/probe_report_consistency.py) and the CLI tests
# (tests/test_report_frontend/test_cli_integration.py,
# tests/test_fp_analysis/test_location_only_verdict.py) import them from this
# module.  Keeping the block means the move changed no import site.  A test
# asserts each alias is the *same object* as the pipeline module's attribute, so
# it cannot rot into a second, divergent definition.
#
# Note the three genuinely-owned imports just below (CFGCacheSet,
# _resolve_source_file_for_cfg, build_dense_path_annotations): the tools got
# those from this module's import block before the refactor, so they are kept
# pointing at their real home rather than re-exported through the pipeline.
from llm_client.fp_analysis.cfg_cache_set import CFGCacheSet  # noqa: F401
from llm_client.fp_analysis.cfg_feasibility import _resolve_source_file_for_cfg  # noqa: F401
from llm_client.fp_analysis.path_seed_extractor import build_dense_path_annotations  # noqa: F401
from llm_client.report_frontend import ReportFrontend, ReportRef  # noqa: F401
from llm_client.report_frontend.adapters.base import describe_adapters_table

from llm_client.fp_analysis.pipeline.artifacts import (  # noqa: F401
    code_model_report,
    report_fields,
    report_key,
    report_out_path,
)
from llm_client.fp_analysis.pipeline.batch import (
    _collect_reports,
    _is_report_file,
    run_batch,
)
from llm_client.fp_analysis.pipeline.layout import DEFAULT_PROJECTS_DIR  # noqa: F401
from llm_client.fp_analysis.pipeline.project_paths import (  # noqa: F401
    _check_pdg_artifact_version,
    _infer_source_root,
    _resolve_paths,
    load_cfg_cache,
    load_sdg,
)
from llm_client.fp_analysis.pipeline.runner import process_report  # noqa: F401
from llm_client.fp_analysis.pipeline.selection import (  # noqa: F401
    _NO_PATH_STEP_KEYWORD,
    _no_path_verdict,
)
from llm_client.fp_analysis.pipeline.stages import (  # noqa: F401
    _infer_entry_function,
    build_slice_mask,
    parse_report,
    run_branch_filter,
    segment_path,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end path selection pipeline (segmentation + branch filter + path "
            "selection). Single --report, or batch-scan a project's reports dir with "
            "--project / --projects-dir, or an arbitrary directory with --scan-dir."
        ),
    )
    parser.add_argument("--report", default=None,
                        help="Path to a single CSA report HTML file")
    parser.add_argument("--scan-dir", default=None,
                        help="Batch mode: recursively process all report-*.html under this dir")
    parser.add_argument("--project", default=None,
                        help="Project name or source-root path. Without --report, scans "
                             "<projects-dir>/<project>/reports/ (TP, FP).")
    parser.add_argument("--projects-dir", default=str(DEFAULT_PROJECTS_DIR),
                        help=f"Base dir containing project source trees "
                             f"(default: {DEFAULT_PROJECTS_DIR})")
    parser.add_argument("--source-root", default=None,
                        help="Explicit source root (overrides --project inference)")
    parser.add_argument("--pdg", default=None, help="Path to PDG JSON file")
    parser.add_argument("--cfg-cache", default=None, help="Path to CFG cache JSON file")
    parser.add_argument("--max-path-attempts", type=int, default=3,
                        help="Path-traversal threshold: max candidate paths to explore "
                             "during POC verification before returning UNKNOWN "
                             "(default: 3)")
    parser.add_argument("--path-space", action="store_true",
                        help="Print the full per-segment feasible-path-space ASCII "
                             "tree (collection always runs and is serialized to JSON)")
    parser.add_argument("--output", default=None,
                        help="Results JSON path: the file itself for a single "
                             "--report run, or the DIRECTORY holding one "
                             "path_selection_<report>.json per report in batch "
                             "mode (default: output/path_selection_<report>.json)")
    parser.add_argument("--batch-summary", default=None,
                        help="Batch aggregate JSON path (default: "
                             "output/batch_summary.json).  Separate from --output "
                             "so the aggregate cannot overwrite a report's result.")
    parser.add_argument("--max-reports", type=int, default=None,
                        help="Batch mode: process at most this many reports (no cap by default)")
    parser.add_argument("--report-format", default=None,
                        help="Pin the report adapter (e.g. csa-html, sarif) instead "
                             "of detecting it from the file's content. Use when the "
                             "detector reports an ambiguous or unknown format. "
                             "See --list-formats.")
    parser.add_argument("--list-formats", action="store_true",
                        help="Print the registered report adapters and exit.")
    parser.add_argument("--list-findings", action="store_true",
                        help="Print every finding in --report (id, rule, location) "
                             "and exit — the selectors --report accepts as "
                             "'<path>::<finding id>'.")
    args = parser.parse_args()

    if args.list_formats:
        print(describe_adapters_table())
        sys.exit(0)

    if args.list_findings:
        if not args.report:
            print("❌ --list-findings needs --report <path>", file=sys.stderr)
            sys.exit(1)
        ref = ReportRef.parse(args.report)
        frontend = ReportFrontend(report_format=args.report_format)
        document = frontend.load_document(ref)
        print(f"{document.path.name}: {len(document.findings)} finding(s) "
              f"[{document.format.value} / {document.analyzer.value}]")
        for finding_id, finding in frontend.iter_findings(ref):
            print(f"  {finding_id}")
            print(f"      {finding.rule_id or '(no rule)'} · {finding.severity.value}"
                  f" · {finding.location.describe()}")
            if finding.message:
                print(f"      {finding.message.splitlines()[0][:160]}")
        sys.exit(0)

    # Check API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not set in .env")
        sys.exit(1)
    logger.info("DEEPSEEK_API_KEY: %s...", api_key[:8])

    reports = _collect_reports(args)
    if args.max_reports:
        reports = reports[: args.max_reports]
    if not reports:
        print("No reports found to process.", file=sys.stderr)
        sys.exit(1)

    # Fail fast: resolve every report's source root / PDG *before* the first LLM
    # call.  Otherwise a misinvocation is either silently analysed against the
    # wrong project (the removed fallback) or reported as a per-report ERROR row
    # only after tens of minutes of unrelated work.
    for rp in reports:
        try:
            _, _, pdg_path, source_root = _resolve_paths(args, rp)
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ {rp.key}: {e}", file=sys.stderr)
            sys.exit(2)
        logger.info("Resolved %s → source_root=%s | pdg=%s",
                    rp.key, source_root, pdg_path.name)

    run_batch(args, reports)


if __name__ == "__main__":
    main()
