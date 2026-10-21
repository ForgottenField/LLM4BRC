"""Work-item collection and the batch driver.

One *finding* is one unit of work (see ``docs/report-frontend.md``): a CSA report
expands to one item, a CodeQL SARIF document to one per ``results[]`` entry.
``_collect_reports`` asks the frontend for that expansion; ``run_batch`` then
resolves every item's project up front (fail-fast, before the first LLM call)
and processes them one at a time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from report_frontend import ReportFrontend, ReportRef

from pipeline.artifacts import report_fields
from pipeline.layout import REPO_ROOT, get_logger
from pipeline import process_report

logger = get_logger()

#: File names that are reports in *any* supported format.  CSA writes
#: ``report-*.html``; a CodeQL/Infer/Cppcheck export is whatever the user named
#: it, so the SARIF suffixes count on their own.  ``index.html`` (CSA's index
#: page) is deliberately excluded: it is a list of reports, not one.


def is_report_file(path: Path) -> bool:
    """True when *path* looks like a report a supported adapter can read."""
    name = path.name.lower()
    if name.endswith((".sarif", ".sarif.json")):
        return True
    return path.suffix == ".html" and path.name.startswith("report-")


def collect_reports(args):
    """Return the ordered list of work items (``ReportRef``) to process.

    A work item is one *finding*, not one file: a ``--report x.sarif`` with no
    selector expands to one ``<path>::<finding id>`` per result in the document,
    the same way a directory of CSA reports expands to one item per file.  The
    expansion is the frontend's — it knows each format's notion of a finding id
    (a CSA HTML file is one finding; a SARIF document is one per ``results[]``
    entry).
    """
    if args.scan_dir:
        scan = Path(args.scan_dir).resolve()
        if not scan.is_dir():
            raise FileNotFoundError(f"Scan dir not found: {scan}")
        found = sorted(p for p in scan.rglob("*") if is_report_file(p))
    elif args.project and not args.report:
        proj = Path(args.project)
        if proj.is_dir():
            reports_dir = proj / "reports"
        else:
            reports_dir = Path(args.projects_dir) / args.project / "reports"
        if not reports_dir.is_dir():
            raise FileNotFoundError(f"No reports dir found: {reports_dir}")
        found = sorted(p for p in reports_dir.rglob("*") if is_report_file(p))
    elif args.report:
        ref = ReportRef.parse(args.report)
        ref = ReportRef(ref.path.resolve(), ref.selector)
        frontend = ReportFrontend(report_format=getattr(args, "report_format", None))
        return frontend.expand([ref])
    else:
        raise SystemExit(
            "No reports to process: pass --report, or --project / --scan-dir"
        )

    frontend = ReportFrontend(report_format=getattr(args, "report_format", None))
    return frontend.expand(found)


def run_batch(args, reports) -> None:
    """Process every work item in *reports*, then print and save the aggregate.

    The caller (``main``) has already collected the work items and resolved each
    one's source root / PDG, so a misinvocation has failed in seconds rather than
    after tens of minutes of unrelated work.
    """
    print(f"Processing {len(reports)} report(s)...\n")

    aggregate = []
    t_all = time.time()
    for i, rp in enumerate(reports, 1):
        print(f"\n{'#' * 70}\n# [{i}/{len(reports)}] {rp.key}\n{'#' * 70}")
        t_report = time.time()
        try:
            out = process_report(args, rp)
            aggregate.append({
                **report_fields(rp),
                "project": out.get("project", ""),
                "bug_type": out.get("bug_type", ""),
                "verdict": out.get("verdict", "?"),
                "verdict_step": out.get("verdict_step", ""),
                "verdict_reason": out.get("verdict_reason", ""),
                "elapsed_sec": out.get("total_elapsed_sec", 0),
                "paths": (out.get("path_space", {}) or {}).get(
                    "total_feasible_candidates", 0),
                "bound": (out.get("context_insensitive", {}) or {}).get(
                    "collapsed_bound",
                    (out.get("path_space", {}) or {}).get("whole_path_bound", 0)),
            })
        except Exception as e:
            logger.error("Report %s failed: %s", rp, e, exc_info=True)
            aggregate.append({
                **report_fields(rp),
                "project": "",
                "bug_type": "",
                "verdict": "ERROR",
                "verdict_step": "",
                "verdict_reason": f"运行异常: {str(e)[:80]}",
                "elapsed_sec": round(time.time() - t_report, 1),
                "error": str(e),
            })
        print(f"\n[{i}/{len(reports)}] done ({time.time() - t_report:.1f}s).")

    # ── Batch summary ──
    n_tp = sum(1 for a in aggregate if a["verdict"] == "TP")
    n_fp = sum(1 for a in aggregate if a["verdict"] == "FP")
    n_unknown = sum(1 for a in aggregate if a["verdict"] == "UNKNOWN")
    n_err = len(aggregate) - n_tp - n_fp - n_unknown

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"{'report':<50}{'bug_type':<18}{'verdict':>6}  {'paths':>5} {'bound':>10}  {'time/s':>7}  step | reason")
    print("-" * 128)
    for a in aggregate:
        step = (a.get("verdict_step") or "")[:24]
        reason = (a.get("verdict_reason") or "").replace("\n", " ")[:52]
        et = a.get("elapsed_sec", 0) or 0
        paths = a.get("paths", 0) or 0
        bound = a.get("bound", 0) or 0
        bound_str = f"{bound:_}" if bound < 1e12 else f"{bound:.2e}"
        print(f"{a['report'][:50]:<50}{a['bug_type'][:18]:<18}{a['verdict']:>6}  "
              f"{paths:>5} {bound_str:>10}  {et:>7.1f}  {step:<24} | {reason}")
    print("-" * 128)
    print(f"TOTAL: {len(aggregate)} | TP={n_tp}  FP={n_fp}  UNKNOWN={n_unknown}  "
          f"ERROR={n_err}  ({time.time() - t_all:.1f}s)")

    # Write aggregate JSON
    agg_out = (Path(args.batch_summary) if args.batch_summary
               else REPO_ROOT / "output" / "batch_summary.json")
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps({
        "total": len(aggregate), "tp": n_tp, "fp": n_fp, "error": n_err,
        "results": aggregate,
    }, indent=2, default=str))
    print(f"\nBatch summary saved to: {agg_out}")
