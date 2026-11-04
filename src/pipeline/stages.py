"""The canonical stage numbering and the deterministic stage drivers.

The classification method has exactly TEN ordered stages; the authoritative
description of each (inputs, outputs, verdict exits, code locations) is
``docs/pipeline-stages.md``.  Every console banner and every ``verdict_step``
string is generated from ``STAGES`` below, so the labels can never drift apart
again (they previously read 1/2/3/3.5/4/5/5.5/6/6.5/5.5b/7, with "Step 4" reused
by two different things and the domain supplement printing "Step 5.5" *after*
Step 6).

The stage drivers here are stages 1-5 (parsing, segmentation, slicing,
assumption state, branch filter).  They are deterministic apart from stage 4,
which the caller supplies an analyzer for.  Stages 6-10 live in
:mod:`pipeline.selection`, ``exploration`` and
``runner`` — the parts that own the report-level verdict.
"""

from __future__ import annotations

from pathlib import Path

from fp_analysis.cfg_branch_extractor import CFGBranchExtractor
from fp_analysis.cfg_branch_models import Postcondition
from fp_analysis.cfg_cache_set import CFGCacheSet, find_fn_key
from fp_analysis.cfg_feasibility import segment_path as _segment_path
from fp_analysis.condition_extractor import extract_condition_from_lines
from fp_analysis.conflict_checker import ConflictChecker
from fp_analysis.models import code_model_of
from fp_analysis.path_seed_extractor import (
    build_caller_map,
    build_path_annotations,
    extract_seeds_from_bug_path,
)
from fp_analysis.pdg_models import SDG
from fp_analysis.report_bridge import to_legacy
from fp_analysis.slice_mask import SliceMask
from fp_analysis.slicer import SlicerConfig, compute_slice_from_seeds
from fp_analysis.structural_pruning import build_pruner_from_segments
from report_frontend import ReportFrontend

from pipeline.layout import get_logger
logger = get_logger()


# ═══════════════════════════════════════════════════════════════════════════════════
# Canonical stage numbering — the single source of truth
# ═══════════════════════════════════════════════════════════════════════════════════
#
# The classification method has exactly TEN ordered stages; the authoritative
# description of each (inputs, outputs, verdict exits, code locations) is
# ``docs/pipeline-stages.md``.  Every console banner and every ``verdict_step``
# string is generated from this table, so the labels can never drift apart again
# (they previously read 1/2/3/3.5/4/5/5.5/6/6.5/5.5b/7, with "Step 4" reused by
# two different things and the domain supplement printing "Step 5.5" *after*
# Step 6).
#
# Stage numbers and names are stable; the *keyword phrases* used by
# ``stage_step`` callers are too — ``tools/summarize_faiss_eval.py`` matches on
# them (e.g. ``"遍历超限"`` marks a traversal-limit UNKNOWN).
STAGES: tuple[tuple[int, str], ...] = (
    (1, "报告解析与工程装配"),
    (2, "路径分段"),
    (3, "PDG 反向切片"),
    (4, "假设可行性"),
    (5, "分支过滤"),
    (6, "确定性一致性闸门"),
    (7, "语义有限域冲突"),
    (8, "路径空间收集与压缩"),
    (9, "路径选择"),
    (10, "POC 验证与结论"),
)
_STAGE_NAMES: dict[int, str] = dict(STAGES)

assert tuple(n for n, _ in STAGES) == tuple(range(1, len(STAGES) + 1))


def stage_label(n: int) -> str:
    """The console/JSON label for a stage, e.g. ``Step 6 报告路径自相矛盾``."""
    return f"Step {n} {_STAGE_NAMES[n]}"


def stage_step(n: int, suffix: str) -> str:
    """A ``verdict_step`` row: canonical stage prefix + a decisive suffix.

    ``suffix`` names *why* the verdict was reached and is what the eval tooling
    greps for (``tools/summarize_faiss_eval.py`` matches ``遍历超限``); keep the
    Chinese keyword phrases in it stable.
    """
    sep = "" if suffix.startswith("（") else " "
    return f"{stage_label(n)}{sep}{suffix}"


def parse_report(ref, source_root: Path, report_format: str | None = None):
    """Stage 1: parse any supported report format into the pipeline's views.

    The parsing is the frontend's (CSA HTML, SARIF — see
    ``docs/report-frontend.md``); ``report_bridge`` renders the canonical
    finding back into the ``ParsedReport``/``BugPath`` every stage below reads.
    For CSA input the rendering is byte-identical to the old parser's output, so
    this is a substitution and not a behaviour change.

    Returns ``(parsed, bug_path, loaded)``; the ``LoadedReport`` is the
    canonical model — the finding, the document it came from and the code it
    resolved — which the caller reports to the result JSON.
    """
    frontend = ReportFrontend(source_root=str(source_root), report_format=report_format)
    loaded = frontend.load(ref)
    parsed, bug_path = to_legacy(loaded)
    for warning in loaded.warnings:
        logger.warning("report frontend: %s", warning)
    logger.info(
        "Parsed report: %s | %s | %d events (%s/%s)",
        parsed.metadata.function_name,
        parsed.metadata.bug_type,
        len(parsed.path_events),
        loaded.document.format.value,
        loaded.finding.analyzer.value,
    )
    return parsed, bug_path, loaded


def infer_entry_function(sdg, parsed_report) -> str:
    """The function a finding's ``(file, line)`` lands in, per the PDG.

    A CSA report states its entry function in a ``FUNCTIONNAME`` comment; a
    CodeQL or Infer SARIF does not name it at all, and an empty entry function
    costs the pipeline two things: the CFG-cache last-resort lookup
    (:func:`_resolve_cfg_cache`, which probes for a function *name*) and the
    slice's permissive-mask fallback both go blind.

    The PDG is authoritative here rather than a heuristic: every function it
    knows carries a source file and a node set, and the function a line lands in
    is the one whose node lines *tightly* contain it.  Only used when the report
    named nothing — a CSA report's own answer is never second-guessed — and it
    returns ``""`` rather than guessing when no PDG range contains the line.
    """
    if parsed_report.metadata.function_name:
        return parsed_report.metadata.function_name
    bug_file = parsed_report.metadata.bug_file or ""
    bug_line = int(parsed_report.metadata.bug_line or 0)
    if not bug_file or not bug_line:
        return ""
    bug_name = Path(bug_file).name

    best: tuple[int, str] = (0, "")
    for fn_name, pdg in sdg.pdgs.items():
        if Path(pdg.source_file or "").name != bug_name:
            continue
        lines = [n.source_line for n in pdg.nodes.values() if n.source_line]
        if not lines:
            continue
        lo, hi = min(lines), max(lines)
        if not (lo <= bug_line <= hi):
            continue
        # Tightest containing range wins: an out-of-line definition and its
        # in-class declaration both "contain" the line otherwise.
        span = hi - lo
        if not best[1] or span < best[0]:
            best = (span, fn_name)
    if best[1]:
        logger.info(
            "report names no entry function — using %s (PDG range for %s:%d)",
            best[1], bug_name, bug_line,
        )
    return best[1]


_infer_entry_function = infer_entry_function


def segment_path(parsed, sdg=None, source_root=None):
    segments = _segment_path(parsed, sdg=sdg, source_root=source_root)
    logger.info("Segmented into %d segment(s)", len(segments))
    for s in segments:
        logger.info("  Seg %d: L%d-L%d (%s) | postcond: %s",
                    s.segment_index, s.line_start, s.line_end,
                    s.function_name, s.format_postcondition())
    return segments


def build_slice_mask(sdg, parsed_report, bug_path, source_root: Path,
                     with_slice_result: bool = False, file_index=None):
    """Build SliceMask from PDG backward slicing.

    Returns the ``SliceMask``; when ``with_slice_result=True`` returns
    ``(mask, slice_result)`` so callers can read the slice's ``truncated`` flag.

    ``file_index`` is the optional ``SDG.build_file_index()`` map; passing it in
    avoids re-scanning all 24K functions per path event.
    """
    logger.info("Building SliceMask from PDG ...")

    seeds = extract_seeds_from_bug_path(
        bug_path, sdg,
        source_root=str(source_root),
        fallback_trigger_fn=parsed_report.metadata.function_name,
        fallback_trigger_line=parsed_report.metadata.bug_line,
    )

    if not seeds:
        logger.warning("No seeds extracted — using permissive mask")
        mask = _permissive_mask(sdg, parsed_report.metadata.function_name)
        return (mask, None) if with_slice_result else mask

    annotations = build_path_annotations(
        bug_path, sdg, source_root=str(source_root), file_index=file_index,
    )
    caller_map = build_caller_map(bug_path, sdg, entry_fn=parsed_report.metadata.function_name)
    slicer_cfg = SlicerConfig(max_depth=10)
    slice_result = compute_slice_from_seeds(
        sdg, seeds, slicer_cfg,
        annotations=annotations,
        caller_map=caller_map,
    )

    if not slice_result.nodes:
        logger.warning("Slice produced 0 nodes — using permissive mask")
        mask = _permissive_mask(sdg, parsed_report.metadata.function_name)
        return (mask, slice_result) if with_slice_result else mask

    mask = SliceMask.from_slice_result(slice_result, sdg)
    logger.info("SliceMask: %d functions, %d PDG nodes",
                len(mask.pdg_retained_nodes),
                sum(len(v) for v in mask.pdg_retained_nodes.values()))
    return (mask, slice_result) if with_slice_result else mask


def _permissive_mask(sdg, entry_fn_name: str = "") -> SliceMask:
    for name, pdg in sdg.pdgs.items():
        if entry_fn_name and entry_fn_name in name:
            return SliceMask(
                pdg_retained_nodes={entry_fn_name: set(pdg.nodes.keys())},
            )
    if sdg.pdgs:
        first = next(iter(sdg.pdgs.values()))
        return SliceMask(
            pdg_retained_nodes={first.function_name: set(first.nodes.keys())},
        )
    return SliceMask()


def run_branch_filter(
    segments: list,
    parsed_report,
    cfg_cache: dict,
    sdg: SDG,
    slice_mask: SliceMask,
    source_root: Path,
    established_state: dict | None = None,
    cache_set: CFGCacheSet | None = None,
    path_annotations: dict | None = None,
) -> tuple[list[dict], dict[int, int]]:
    """Run CFG branch extraction + conflict checking for all segments.

    Args:
        path_annotations: ``{(qualified_fn, line): branch_taken}`` from
            :func:`build_dense_path_annotations`.  Lets the extractor expand only
            the calls the reported path can actually reach.  When omitted, the
            extractor falls back to plain line-range scoping.

    Returns:
        (segment_results, branch_counts) where branch_counts is
        {seg_idx: count_of_insufficient_info_branches}.
    """
    bug_file = parsed_report.metadata.bug_file
    model = code_model_of(parsed_report, str(source_root))
    bug_hit = model.resolve_strong(bug_file)
    resolved_src = str(bug_hit) if bug_hit else bug_file
    src_lines = model.lines(bug_hit) if bug_hit else None
    pruner = build_pruner_from_segments(segments, resolved_src,
                                         path_events=parsed_report.path_events,
                                         source_lines=src_lines)

    # Bug-function source boundaries (when the resolved source is readable).
    # Used to guard the empty-function-name fallback: a segment whose function
    # name failed to resolve should only inherit the BUG function's branches if
    # its line range actually falls inside the bug function.  Otherwise it is
    # left as a linear segment (no branch tree), instead of wrongly extracting
    # the bug function's branches for every unresolved segment.
    bug_fn_name = parsed_report.metadata.function_name
    bug_fn_bounds: list[tuple[int, int]] = []
    if src_lines:
        from fp_analysis.cfg_feasibility import _find_function_boundaries
        for _n, _s, _e in _find_function_boundaries(Path(resolved_src), src_lines):
            if _n == bug_fn_name:
                bug_fn_bounds.append((_s, _e))

    results = []
    insufficient_counts: dict[int, int] = {}

    def _in_bug_fn(segment) -> bool:
        if not bug_fn_bounds:
            return False
        # A segment in a *different* file can never be inside the bug function,
        # no matter how the line numbers happen to overlap.
        seg_file = getattr(segment, "function_file", "") or ""
        if seg_file and resolved_src:
            try:
                if Path(seg_file).resolve() != Path(resolved_src).resolve():
                    return False
            except OSError:
                pass
        return any(
            s <= segment.line_start <= e or s <= segment.line_end <= e
            for s, e in bug_fn_bounds
        )

    extractor = CFGBranchExtractor(cfg_cache, sdg, slice_mask, max_depth=3,
                                   path_annotations=path_annotations,
                                   source_root=str(source_root))

    # Function each segment really lives in, resolved by (file, line range)
    # evidence rather than by name.  Segments carry bare method names — and, for
    # events inside a macro expansion, the macro's name or nothing at all — so a
    # name-only lookup silently lands on a same-named function (a different
    # overload, an unrelated class, or a libstdc++ one).  Resolving once up
    # front serves both the branch extraction below and the "has its own
    # segments" set, which must agree.
    seg_pdgs: dict[int, object] = {}
    for seg_idx, segment in enumerate(segments):
        seg_pdgs[seg_idx] = extractor.resolve_segment_pdg(
            segment.function_name or "",
            getattr(segment, "function_file", "") or "",
            segment.line_start, segment.line_end,
        )

    # Functions that have their own explicit segments: their internal branches
    # belong to those segments and must not be re-expanded as callees here.
    segmented_callees: set[str] = {s.function_name for s in segments
                                   if s.function_name}
    segmented_callees |= {
        pdg.function_name for pdg in seg_pdgs.values() if pdg is not None
    }

    for seg_idx, segment in enumerate(segments):
        # Build postcondition from CSA path event
        pc = _build_postcondition(segment, parsed_report, source_root)

        # Resolve function name.  Fall back to the report's bug function ONLY
        # when the segment's own name is empty AND its line range lies inside the
        # bug function.  Otherwise leave it empty → the segment is treated as a
        # linear segment (no branch tree), rather than attributing the bug
        # function's branches to unrelated cross-file segments.
        seg_fn_name = segment.function_name
        if not seg_fn_name and _in_bug_fn(segment):
            seg_fn_name = bug_fn_name
        # A merged multi-TU cache can hold the same function from several
        # translation units; prefer the one belonging to this segment's file.
        seg_fn_file = getattr(segment, "function_file", "") or ""
        if cache_set is not None:
            fn_key = cache_set.find_key(seg_fn_name, source_file=seg_fn_file)
        else:
            fn_key = find_fn_key(cfg_cache, seg_fn_name, source_file=seg_fn_file)

        # Without a CFG key the extractor can still work from the PDG — that is
        # the primary extraction path anyway — so only a segment that resolves
        # to neither is dropped.
        if fn_key is None and seg_pdgs.get(seg_idx) is None:
            logger.warning("  Seg %d: function '%s' not in CFG cache and no PDG "
                           "covering L%d-L%d of %s — skipping",
                           seg_idx, seg_fn_name, segment.line_start,
                           segment.line_end, Path(seg_fn_file).name or "?")
            continue

        tree = extractor.extract(
            segment_index=seg_idx,
            function_name=fn_key or seg_fn_name or "",
            function_file=seg_fn_file,
            line_start=segment.line_start,
            line_end=segment.line_end,
            postcondition=pc,
            segmented_callees=segmented_callees,
        )

        # Run conflict checking (tags branches as consistent/contradictory/insufficient_info).
        # Pass in assumption-derived established_state (when available) so Layers 1b/1.5
        # (state-matching + PDG variable-invariance) can actually fire instead of returning
        # insufficient_info for every branch.
        # Source lines are best-effort context for conflict checking; a report may
        # reference a source file not present on disk (e.g. a gtest/folly dep header
        # resolved to a non-existent fbcode_builder temp path). Missing source must
        # not crash the report — fall back to None (conflict checker runs without it).
        checker = ConflictChecker(
            sdg, cfg_cache, slice_mask, structural_pruner=pruner,
        )
        tree = checker.check_segment(
            tree, segment, established_state or {},
            path_events=parsed_report.path_events,
            event_range=segment.event_range,
            source_lines=src_lines,
        )
        tree.update_stats()

        # Attach CFG branch tree to segment for shortest-path computation
        segment.cfg_branch_tree = tree

        results.append({
            "segment_index": seg_idx,
            "num_branches": tree.num_branches,
            "num_consistent": tree.num_consistent,
            "num_contradictory": tree.num_contradictory,
            "num_insufficient": tree.num_insufficient,
        })

        insufficient_counts[seg_idx] = tree.num_insufficient

    return results, insufficient_counts


def _build_postcondition(segment, parsed_report, source_root: Path) -> Postcondition:
    """Build a Postcondition from the segment's control event."""
    evt_desc = ""
    evt_line = segment.line_end
    branch_taken = (
        segment.control_branch_taken
        if segment.control_branch_taken is not None
        else True
    )
    boundary_event = None

    constraints = [
        c for c in (getattr(segment, "control_constraints", None) or [])
        if int(c.get("line") or 0) == int(segment.line_end)
        and isinstance(c.get("branch_taken"), bool)
    ]
    if constraints:
        if segment.control_event_number is not None:
            boundary_event = next(
                (
                    c for c in constraints
                    if c.get("event_number") == segment.control_event_number
                ),
                None,
            )
        if boundary_event is None:
            boundary_event = min(
                constraints,
                key=lambda c: int(c.get("event_number") or 0),
            )
        branch_taken = bool(boundary_event["branch_taken"])

    event_number = (
        boundary_event.get("event_number")
        if boundary_event is not None
        else segment.control_event_number
    )
    if event_number is not None:
        for evt in parsed_report.path_events:
            if evt.get("path_number") == event_number:
                evt_desc = evt.get("description", "")
                evt_line = evt.get("line_number", segment.line_end)

                # Try to extract concrete condition from source code
                source_file = (
                    evt.get("file_path", "")
                    or getattr(
                        getattr(parsed_report, "metadata", None),
                        "bug_file",
                        "",
                    )
                )
                if source_file:
                    model = code_model_of(parsed_report, str(source_root))
                    hit = model.resolve_strong(source_file)
                    lines = model.lines(hit) if hit else []
                    extracted = extract_condition_from_lines(lines, evt_line)
                    if extracted.confidence > 0:
                        evt_desc = extracted.expression
                        evt_line = extracted.source_line
                break

    return Postcondition(
        function_name=segment.function_name or parsed_report.metadata.function_name,
        source_line=evt_line,
        condition_expr=evt_desc,
        branch_taken=branch_taken,
    )
