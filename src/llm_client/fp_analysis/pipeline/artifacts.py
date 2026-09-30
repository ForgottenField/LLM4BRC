"""Result identity, output paths, and the one place a result is written.

Every artifact the pipeline produces per work item is named here: the
filesystem-safe key, the identifying fields every result JSON carries, where the
JSON goes, and the ``code_model`` block that says what the report's index made
the code model read.

``_write_output`` exists because the write itself was three identical copies
(the stage-6 exit, the stage-7 exit, the full-path exit) that had to stay
identical by hand — the printed lines are part of the tool's output and a
divergence between them would be invisible until someone read two logs side by
side.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from llm_client.fp_analysis.models import code_model_of
from llm_client.report_frontend import ReportRef

from llm_client.fp_analysis.pipeline.layout import REPO_ROOT, get_logger

logger = get_logger()


def report_key(report_path: Path | ReportRef) -> str:
    """The filesystem-safe name for a work item's outputs.

    A :class:`ReportRef` answers this itself (``ref.key``) — the report's stem,
    with the finding selector appended when the document holds several findings,
    which is what keeps 62 SARIF findings from writing to one filename.  A bare
    path keeps its own stem, exactly as before.
    """
    if isinstance(report_path, ReportRef):
        return report_path.key
    return Path(report_path).stem


def report_fields(report_ref: ReportRef) -> dict:
    """The identifying fields every result JSON carries.

    ``report`` is the file's name for an unambiguous single-finding run — so
    every existing CSA result JSON is unchanged — and the *work item*'s key once
    a selector is in play: a SARIF document holding sixty findings would
    otherwise write sixty identical values, and the batch summary could not tell
    them apart.  The file it came from and the selector inside it are kept
    alongside either way.
    """
    name = report_ref.path.name if not report_ref.selector else report_ref.key
    return {
        "report": name,
        "report_file": report_ref.path.name,
        "finding": report_ref.selector,
    }


def code_model_report(parsed_report, source_root) -> dict:
    """The result JSON's ``code_model`` block — what the code model read.

    Emitted by **every** result, including the ones that exit before stage 8:
    the report's index is what every stage reads code through, so the early
    exits say what it cost exactly as the full path does.  Visibility only — it
    never participates in a verdict, and it is not a switch.
    """
    model = code_model_of(parsed_report)
    if model is None:  # pragma: no cover — code_model_of always answers
        return {"drift": [], "drift_truncated": 0, "code_set": None,
                "lazy_reads": [], "lazy_reads_truncated": 0}
    drift = model.drift
    if drift:
        logger.warning(
            "code model: the report's embedded copy and the local tree under "
            "%s disagree on %d line(s) across %d file(s); first: %s:%s "
            "(report %r vs tree %r)",
            source_root, len(drift),
            len({r["reported"] for r in drift}),
            drift[0]["reported"], drift[0]["line"],
            drift[0]["embedded"][:60], drift[0]["local"][:60],
        )
    return {
        "drift": drift,
        "drift_truncated": model.drift_truncated,
        # What the report's index made the code model read up front, and what
        # it still had to read afterwards.  The second list is the honest
        # measure of the first: a run that had to read more than the pre-load
        # holds is a run whose memory is not bounded by the report's files.
        "code_set": model.code_set,
        "lazy_reads": model.lazy_reads,
        "lazy_reads_truncated": model.lazy_reads_truncated,
    }


def report_out_path(args, report_path: Path | ReportRef) -> Path:
    """Where this report's result JSON is written.

    For a single ``--report`` run, ``--output`` names the file directly.  A batch
    run has no single report to name the file after, so there the same flag names
    a *directory* and each report gets ``<dir>/path_selection_<key>.json``.

    The aggregate goes to ``--batch-summary`` (default
    ``output/batch_summary.json``) and never to the same path as a per-report
    file: the two used to share ``--output``, so on a batch run every report's
    JSON overwrote the previous one and the summary overwrote them all.
    """
    default_name = f"path_selection_{report_key(report_path)}.json"
    if not getattr(args, "output", None):
        return REPO_ROOT / "output" / default_name
    out = Path(args.output)
    return out if getattr(args, "report", None) else out / default_name


def _write_output(out_path: Path, output: dict, total_start: float) -> None:
    """Write a result JSON and print the two lines every exit prints.

    The single writer for all three exits (stage 6, stage 7, full path): they
    used to carry byte-identical copies of this trio, so the ordering
    "``Full results saved to:`` then ``Total pipeline time:``" was an invariant
    maintained by hand.
    """
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull results saved to: {out_path}")
    print(f"Total pipeline time: {time.time() - total_start:.1f}s")
