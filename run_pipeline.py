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
``pipeline`` (see that package's ``__init__`` for the
layer map); this module parses the arguments, loads API credentials, configures
logging, resolves every work item up front so a misinvocation fails in seconds,
and hands the list to the batch driver.  The compatibility block below
re-exports the names the harness tools (``tools/``) and the CLI tests import
from here — those imports are unchanged by the move, on purpose.

Usage:
    python3 run_pipeline.py --report <path/to/report.html>
    python3 run_pipeline.py --source-root ~/csa_reports/project/faiss --report <path>
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

_HERE = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger("path_selection")

HERE = _HERE

# ─── Compatibility re-exports ────────────────────────────────────────────────────
# The implementation lives in pipeline.*.  These names are
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
from fp_analysis.cfg_cache_set import CFGCacheSet  # noqa: F401
from fp_analysis.cfg_feasibility import resolve_source_file_for_cfg  # noqa: F401
from fp_analysis.path_seed_extractor import build_dense_path_annotations  # noqa: F401
from llm_client.config import PROVIDER_SPECS, provider_spec
from llm_client.credentials import DEFAULT_API_KEY_FILE, load_api_key
from report_frontend import ReportFrontend, ReportRef  # noqa: F401
from report_frontend.adapters.base import describe_adapters_table

from pipeline import (  # noqa: F401
    code_model_report,
    report_fields,
    report_key,
    report_out_path,
)
from pipeline import (
    collect_reports,
    is_report_file,
    run_batch,
)
from pipeline import DEFAULT_PROJECTS_DIR  # noqa: F401
from pipeline import (  # noqa: F401
    check_pdg_artifact_version,
    infer_source_root,
    resolve_paths,
    load_cfg_cache,
    load_sdg,
)
from pipeline import process_report  # noqa: F401
from pipeline import (  # noqa: F401
    _NO_PATH_STEP_KEYWORD,
    _no_path_verdict,
)
from pipeline import (  # noqa: F401
    infer_entry_function,
    build_slice_mask,
    parse_report,
    run_branch_filter,
    segment_path,
)
from pipeline.results import (
    capture_console,
    configure_results_args,
    create_results_layout,
    write_manifest,
)

# Legacy underscore-prefixed names remain available for evaluation scripts.
# They are aliases, not wrappers, so there is still one implementation.
_collect_reports = collect_reports
_is_report_file = is_report_file
_check_pdg_artifact_version = check_pdg_artifact_version
_infer_source_root = infer_source_root
_resolve_paths = resolve_paths
_resolve_source_file_for_cfg = resolve_source_file_for_cfg
_infer_entry_function = infer_entry_function


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
    parser.add_argument(
        "--pdg",
        default=None,
        help="Path to an exact PDG JSON file; supplying it selects the exact "
             "backend.",
    )
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
    parser.add_argument("--provider", choices=sorted(PROVIDER_SPECS),
                        default="deepseek",
                        help="LLM provider (default: deepseek).")
    parser.add_argument("--model", default=None,
                        help="Provider model override; defaults are provider-specific.")
    parser.add_argument("--llm-timeout", type=float, default=120.0,
                        help="Per-request LLM timeout in seconds (default: 120).")
    parser.add_argument("--results-root", default=None,
                        help="Root for timestamped run outputs (default: ./results).")
    parser.add_argument("--api-key-file", default=str(DEFAULT_API_KEY_FILE),
                        help="JSON credential file (default: config/api_key.json).")
    parser.add_argument("--no-lite-graph", action="store_true",
                        help="Disable the Tree-sitter fast path and require an "
                             "exact Clang/PDG artifact.")
    parser.add_argument("--lite-graph-timeout", type=float, default=30.0,
                        help="Timeout in seconds for one isolated Tree-sitter "
                             "graph build (default: 30).")
    parser.add_argument(
        "--compile-commands",
        default=None,
        help="Compilation database for Tree-sitter preprocessing. Defaults to "
             "<source-root>/compile_commands.json.",
    )
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

    reports = collect_reports(args)
    if args.max_reports:
        reports = reports[: args.max_reports]
    if not reports:
        print("No reports found to process.", file=sys.stderr)
        sys.exit(1)

    project_hint = _project_hint(args, reports)
    layout = create_results_layout(args.results_root, project_hint)
    configure_results_args(args, layout)
    started_at = datetime.now(timezone.utc)
    manifest = {
        "run_id": layout.run_id,
        "started_at_utc": started_at.isoformat(),
        "status": "running",
        "project": project_hint,
        "provider": args.provider,
        "model": args.model or provider_spec(args.provider).default_model,
        "report_count": len(reports),
        "reports": [ref.key for ref in reports],
        "results": {
            "project_dir": str(layout.project_dir),
            "reports_dir": str(layout.reports_dir),
            "pocs_dir": str(layout.pocs_dir),
            "batch_summary": str(args.batch_summary),
            "log": str(layout.log_path),
        },
    }
    write_manifest(layout, manifest)

    spec = provider_spec(args.provider)
    try:
        api_key = load_api_key(args.provider, args.api_key_file)
    except Exception as exc:
        manifest.update({
            "status": "blocked",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        write_manifest(layout, manifest)
        print(
            f"❌ {exc}; "
            f"run metadata saved to {layout.manifest_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    args.api_key = api_key
    logger.info("Using LLM provider: %s", spec.name)

    # Fail fast: resolve every report's source root / PDG *before* the first LLM
    # call.  Otherwise a misinvocation is either silently analysed against the
    # wrong project (the removed fallback) or reported as a per-report ERROR row
    # only after tens of minutes of unrelated work.
    source_roots: set[Path] = set()
    for rp in reports:
        try:
            _, _, pdg_path, source_root = resolve_paths(
                args, rp, require_pdg=args.no_lite_graph,
            )
        except (ValueError, FileNotFoundError) as e:
            manifest.update({
                "status": "preflight_failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            })
            write_manifest(layout, manifest)
            print(f"❌ {rp.key}: {e}", file=sys.stderr)
            sys.exit(2)
        source_roots.add(source_root)
        logger.info(
            "Resolved %s → source_root=%s | exact_pdg=%s",
            rp.key, source_root, pdg_path.name if pdg_path else "not required",
        )

    compile_db_paths: set[Path] = set()
    if not args.no_lite_graph and not args.pdg:
        for source_root in source_roots:
            database = (
                Path(args.compile_commands).expanduser().resolve()
                if args.compile_commands
                else source_root / "compile_commands.json"
            )
            if not database.is_file():
                message = (
                    f"compile database not found: {database}. Place it at "
                    "<project-root>/compile_commands.json or pass "
                    "--compile-commands <path>."
                )
                manifest.update({
                    "status": "preflight_failed",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": message,
                })
                write_manifest(layout, manifest)
                print(f"❌ {message}", file=sys.stderr)
                sys.exit(2)
            compile_db_paths.add(database)

    actual_project = (
        next(iter(source_roots)).name if len(source_roots) == 1 else "mixed"
    )
    manifest.update({
        "project": actual_project,
        "source_roots": sorted(str(path) for path in source_roots),
        "exact_pdg": str(pdg_path) if pdg_path else None,
        "graph_strategy": "exact" if args.no_lite_graph else "tree-sitter-first",
        "compile_commands": sorted(str(path) for path in compile_db_paths),
    })
    write_manifest(layout, manifest)

    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(layout.log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(levelname)-5s %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)
    try:
        with capture_console(layout.log_path):
            summary = run_batch(args, reports)
        manifest.update({
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time.time() - started_at.timestamp(), 1),
            "summary": {
                key: summary[key]
                for key in ("total", "tp", "fp", "unknown", "error")
            },
        })
    except BaseException as exc:
        manifest.update({
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time.time() - started_at.timestamp(), 1),
            "error": str(exc),
        })
        raise
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()
        write_manifest(layout, manifest)


def _project_hint(args, reports) -> str:
    for value in (getattr(args, "project", None), getattr(args, "source_root", None)):
        if value:
            return Path(str(value)).name
    if getattr(args, "scan_dir", None):
        return Path(args.scan_dir).name
    if reports:
        return reports[0].path.parent.name
    return "unknown"


if __name__ == "__main__":
    main()
