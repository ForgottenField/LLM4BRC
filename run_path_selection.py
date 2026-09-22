#!/usr/bin/env python3
"""End-to-end path selection pipeline: segmentation + branch filter + path selection.

Runs the complete deterministic pipeline (PDG slicing, CFG branch extraction,
conflict checking) followed by the segment-by-segment path selection algorithm.
Everything runs by default — there are no mode switches to pass:

- Path selection is conflict-driven reselection (the only mechanism; the legacy
  retry path is gone).
- After a TP verdict the POC is generated, LLM-audited and simulated, and the
  feedback loop can settle the verdict as FP or UNKNOWN.

Usage:
    python3 run_path_selection.py --report <path/to/report.html>
    python3 run_path_selection.py --source-root ~/csa_reports/project/faiss --report <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
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

# Default base directory containing the project source trees + reports.  The
# reports/sources live under ~/csa_reports/project/<name>/ (NOT ./project, which
# is not present in this checkout).  Override with --projects-dir when needed.
DEFAULT_PROJECTS_DIR = Path.home() / "csa_reports" / "project"

# ─── Import all encapsulated source code logic ───────────────────────────────────

from llm_client.fp_analysis.cfg_branch_models import Postcondition
from llm_client.fp_analysis.cfg_branch_extractor import CFGBranchExtractor
from llm_client.fp_analysis.conflict_checker import ConflictChecker
from llm_client.fp_analysis.simulation_verifier import SimulationVerifier
from llm_client.fp_analysis.path_space import (
    _chosen_by_seg,
    annotate_ci_groups,
    collect_path_space,
    render_path_space_ascii,
)
from llm_client.fp_analysis.combination_enum import (
    combo_signature,
    iter_combinations,
)
from llm_client.fp_analysis.branch_relevance import (
    classify_branch_relevance,
    reduce_path_space,
)
from llm_client.fp_analysis.cause_invariance import (
    cause_invariant,
    extract_deref_pointer,
)

from llm_client.fp_analysis.condition_extractor import extract_condition
from llm_client.fp_analysis.structural_pruning import build_pruner_from_segments

from llm_client.fp_analysis.html_parser import FPReportParser
from llm_client.fp_analysis.pdg_loader import load_pdg
from llm_client.fp_analysis.pdg_models import SDG
from llm_client.fp_analysis.pdg_augment import augment_pdg_sdg

from llm_client.fp_analysis.path_seed_extractor import (
    build_caller_map,
    build_path_annotations,
    build_dense_path_annotations,
    extract_seeds_from_bug_path,
)
from llm_client.fp_analysis.slicer import SlicerConfig, compute_slice_from_seeds
from llm_client.fp_analysis.slice_mask import SliceMask
from llm_client.fp_analysis.domain_facts import extract_domain_facts, format_domain_facts
from llm_client.fp_analysis.report_consistency import detect_report_contradictions

from llm_client.fp_analysis.cfg_feasibility import (
    segment_path as _segment_path,
    _resolve_source_file_for_cfg,
)
from llm_client.fp_analysis.cfg_cache_set import CFGCacheSet, find_fn_key
from llm_client.fp_analysis.cfg_models import SegmentInfo

from llm_client.fp_analysis.models import (
    CompletedPath, CompletedStep, PathAnalysis, Confidence,
)
from llm_client.fp_analysis.path_analyzer import FPAnalyzer, PathSpaceExhausted
from llm_client.base import LLMConfig
from llm_client.factory import ProviderFactory


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
    (6, "报告路径自相矛盾"),
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


def report_out_path(args, report_path: Path) -> Path:
    """Where this report's result JSON is written.

    For a single ``--report`` run, ``--output`` names the file directly.  A batch
    run has no single report to name the file after, so there the same flag names
    a *directory* and each report gets ``<dir>/path_selection_<stem>.json``.

    The aggregate goes to ``--batch-summary`` (default
    ``output/batch_summary.json``) and never to the same path as a per-report
    file: the two used to share ``--output``, so on a batch run every report's
    JSON overwrote the previous one and the summary overwrote them all.
    """
    default_name = f"path_selection_{report_path.stem}.json"
    if not getattr(args, "output", None):
        return HERE / "output" / default_name
    out = Path(args.output)
    return out if getattr(args, "report", None) else out / default_name


# ═══════════════════════════════════════════════════════════════════════════════════
# Helpers — reuse from run_wslay_branch_analysis.py
# ═══════════════════════════════════════════════════════════════════════════════════


# Markers that identify a directory as a project's source root.  A report whose
# grandparent carries none of them is not a project tree, and *guessing* a
# different one silently produced plausible-looking runs against the wrong
# project — see _resolve_paths.
_PROJECT_ROOT_MARKERS = (
    "deps", "lib", "src", ".git", "CMakeLists.txt", "Makefile", "meson.build",
    "configure", "setup.py",
)


def _looks_like_project_root(path: Path) -> bool:
    """True when *path* is an existing directory that carries any project marker."""
    return path.is_dir() and any(
        (path / marker).exists() for marker in _PROJECT_ROOT_MARKERS
    )


def _resolve_paths(args, report_path: Path | str) -> tuple[Path, Path, Path, Path]:
    """Resolve (report, cfg_cache, pdg, source_root) for one report.

    Every input is validated and a bad one raises ``ValueError`` with the
    command-line fix in the message.  It must never fall back to another
    project: the old ``source_root = HERE/"project"/"aria2"`` substitution made
    ``--report <faiss report>`` load ``pdg_aria2.json`` and an empty
    ``cfg_cache/aria2/``, which produced a *plausible-looking* wrong run
    (18 branches, bound 32768→1, UNKNOWN) instead of an error.
    """
    report_path = Path(report_path).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    if args.project:
        # --project may be a full path to the source root, or a bare name
        # resolved under --projects-dir.
        proj = Path(args.project)
        if proj.is_dir():
            source_root = proj.resolve()
        else:
            source_root = (Path(args.projects_dir) / args.project).resolve()
        if not source_root.is_dir():
            raise ValueError(
                f"--project {args.project!r} resolved to {source_root}, which is "
                f"not a directory (a bare name is looked up under --projects-dir "
                f"{args.projects_dir})."
            )
    elif args.source_root:
        source_root = Path(args.source_root).resolve()
        if not source_root.is_dir():
            raise ValueError(
                f"--source-root {args.source_root!r} resolved to {source_root}, "
                f"which is not a directory."
            )
    else:
        # Infer from the report path: <project>/reports/{TP,FP}/report.html
        source_root = report_path.parent.parent.parent
        if not _looks_like_project_root(source_root):
            raise ValueError(
                f"cannot infer the source root of {report_path.name} from its "
                f"path: {source_root} is not a directory, or has none of "
                f"{', '.join(_PROJECT_ROOT_MARKERS)}. Pass --project <name> "
                f"(looked up under --projects-dir {args.projects_dir}) or "
                f"--source-root <dir>."
            )

    project_name = source_root.name

    if args.pdg:
        pdg_path = Path(args.pdg).resolve()
    else:
        pdg_path = HERE / f"pdg_{project_name}.json"
    if not pdg_path.is_file():
        # The PDG drives every stage; another project's PDG parses fine and
        # silently analyses the wrong call graph.
        raise ValueError(
            f"SDG/PDG file not found: {pdg_path} (project {project_name!r}). "
            f"Build it offline for this project, or pass --pdg <path>."
        )

    if args.cfg_cache:
        cfg_cache_path = Path(args.cfg_cache).resolve()
    else:
        cfg_cache_path = None

    return report_path, cfg_cache_path, pdg_path, source_root


def _cache_mentions_function(
    cache_path: Path, fn_name: str, chunk: int = 4 << 20
) -> bool:
    """Probe a CFG cache for *fn_name* without parsing it.

    Cache files reach hundreds of MB, so a substring scan over the raw JSON is
    dramatically cheaper than ``json.loads`` — and sufficient: the function's
    dump signature is the key of its own entry, so it appears verbatim.
    """
    needle = fn_name.encode()
    tail = b""
    with cache_path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                return False
            if needle in tail + buf:
                return True
            tail = buf[-(len(needle) - 1):] if len(needle) > 1 else b""


def _resolve_cfg_cache(source_root: Path, entry_fn_name: str) -> Path | None:
    """Find the CFG cache holding *entry_fn_name*.

    Last-resort lookup for reports whose bug file is not itself a translation
    unit (typically a header) and whose path files could not be resolved.
    Returns ``None`` when nothing matches — this used to fall back to the
    *largest* cache file, which silently handed the pipeline an unrelated
    translation unit's CFG, worse than no cache at all because the failure
    surfaced deep inside the branch filter instead of at load time.
    """
    cache_dir = HERE / "cfg_cache" / source_root.name
    if not cache_dir.is_dir():
        return None
    caches = sorted(cache_dir.glob("*.json"))
    if not caches:
        return None
    if len(caches) == 1:
        return caches[0]

    for cache_path in caches:
        try:
            if _cache_mentions_function(cache_path, entry_fn_name):
                return cache_path
        except OSError:
            continue
    return None


def _find_fn_key(cfg_cache, function_name: str, source_file: str = ""):
    """Resolve *function_name* to a key of *cfg_cache* (see ``find_fn_key``)."""
    return find_fn_key(cfg_cache, function_name, source_file=source_file)


# ═══════════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ═══════════════════════════════════════════════════════════════════════════════════


def load_cfg_cache(path: Path) -> dict:
    from llm_client.fp_analysis.cfg_feasibility import _load_cfg_cache
    if not path.exists():
        logger.error("CFG cache not found: %s", path)
        return {}
    cache = _load_cfg_cache(path)
    logger.info("Loaded CFG cache: %d functions (%.1f KiB)",
                len(cache), path.stat().st_size / 1024)
    return cache or {}


def load_sdg(path: Path) -> SDG:
    if not path.exists():
        logger.error("PDG file not found: %s", path)
        return SDG(pdgs={})
    pdgs = load_pdg(str(path))
    sdg = SDG(pdgs=pdgs)
    augment_pdg_sdg(sdg)
    logger.info("Loaded SDG: %d functions", len(sdg.pdgs))
    return sdg


def parse_report(path: Path, source_root: Path):
    parser = FPReportParser()
    parsed = parser.parse(str(path))
    bug_path = parser.parse_to_bug_path(str(path))
    logger.info("Parsed report: %s | %s | %d events",
                parsed.metadata.function_name,
                parsed.metadata.bug_type,
                len(parsed.path_events))
    return parsed, bug_path


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
    from llm_client.fp_analysis.path_analyzer import _resolve_source_path

    bug_file = parsed_report.metadata.bug_file
    resolved_src = _resolve_source_path(bug_file, str(source_root))
    pruner = build_pruner_from_segments(segments, resolved_src,
                                         path_events=parsed_report.path_events)

    # Bug-function source boundaries (when the resolved source is readable).
    # Used to guard the empty-function-name fallback: a segment whose function
    # name failed to resolve should only inherit the BUG function's branches if
    # its line range actually falls inside the bug function.  Otherwise it is
    # left as a linear segment (no branch tree), instead of wrongly extracting
    # the bug function's branches for every unresolved segment.
    bug_fn_name = parsed_report.metadata.function_name
    bug_fn_bounds: list[tuple[int, int]] = []
    if resolved_src:
        from llm_client.fp_analysis.cfg_feasibility import _find_function_boundaries
        try:
            _flines = Path(resolved_src).read_text(
                encoding="utf-8", errors="replace").splitlines()
            for _n, _s, _e in _find_function_boundaries(Path(resolved_src), _flines):
                if _n == bug_fn_name:
                    bug_fn_bounds.append((_s, _e))
        except OSError:
            pass

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
            fn_key = _find_fn_key(cfg_cache, seg_fn_name, source_file=seg_fn_file)

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
        src_lines = None
        if resolved_src:
            try:
                src_lines = Path(resolved_src).read_text().splitlines()
            except OSError as e:
                logger.warning(
                    "branch filter: cannot read source %s (%s); continuing without source lines",
                    resolved_src, e,
                )
        checker = ConflictChecker(sdg, cfg_cache, slice_mask)
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
    branch_taken = segment.control_branch_taken if segment.control_branch_taken is not None else True

    if segment.control_event_number is not None:
        for evt in parsed_report.path_events:
            if evt.get("path_number") == segment.control_event_number:
                evt_desc = evt.get("description", "")
                evt_line = evt.get("line_number", segment.line_end)

                # Try to extract concrete condition from source code
                source_file = evt.get("file_path", "") or parsed_report.metadata.bug_file
                if source_file:
                    from llm_client.fp_analysis.path_analyzer import _resolve_source_path
                    resolved = _resolve_source_path(source_file, str(source_root))
                    extracted = extract_condition(resolved, evt_line, source_root=str(source_root))
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


# ═══════════════════════════════════════════════════════════════════════════════════
# Path selection (stage 9) — used by process_report and the harness tools
# ═══════════════════════════════════════════════════════════════════════════════════


def run_path_selection(
    analyzer: FPAnalyzer,
    parsed_report,
    segments: list,
    slice_mask: SliceMask,
    source_root: Path,
    cfg_cache: dict | None = None,
    extra_preconditions: list[str] | None = None,
    hard_constraints: list[str] | None = None,
    blocked_paths: list[str] | None = None,
    blocked_candidates: list[dict] | None = None,
    combo: tuple | None = None,
    path_space: dict | None = None,
) -> tuple[list[dict] | None, dict]:
    """Run segment-by-segment path selection via FPAnalyzer (conflict-driven).

    Two selection modes:

    - ``combo is None``: the LLM-driven selection
      (``_select_feasible_path_segments``).  Best-feasible-first path that
      uses the global constraints/state context.  This is the attempt-1 seed.
    - ``combo is not None``: deterministic whole-path combination enumeration.
      ``combo`` is a 0-based candidate index per segment into ``path_space``'s
      per-segment candidates; selections are built without an LLM pick and
      conflict-checked.  ``selections is None`` here means THAT combination is
      globally infeasible (a whole-path FP) — the caller blocks it and enumerates
      the next, NOT a report-level FP.

    Args:
        extra_preconditions: Optional state facts to inject into the
            CompletedPath.preconditions (e.g. from assumption checking).
        hard_constraints: Hard constraints from assumption feasibility check
            (e.g. ``["blockLength_ > 0"]``).  Passed to path selection so
            constraint conflicts are discovered **by path selection**.
        blocked_paths: Whole-path signatures already tried and abandoned
            (simulation conflicted/unresolved/FP).  Passed to path selection so
            it avoids re-deriving them and picks a different feasible path.
        blocked_candidates: Kept for call-compat only — no longer used for
            re-selection (whole-path combination enumeration replaces the
            per-segment candidate blocking that over-pruned).  Passed empty.

    Returns (selections, stats_dict).  When ``combo is not None`` and the
    combination is globally infeasible, ``stats["combo_infeasible"]`` is True.
    """
    # Build a minimal CompletedPath — the path selection algorithm uses
    # completed.completed_steps / preconditions / concretized_values for
    # context prompts.  Other fields are irrelevant for path selection.
    preconditions = []
    if extra_preconditions:
        preconditions.extend(extra_preconditions)
    completed = CompletedPath(
        original_path_summary="",
        analysis=PathAnalysis(
            path_id="bug_path",
            is_likely_fp=False,
            gaps=[],
            confidence=Confidence.LOW,
            reasoning="placeholder for path selection",
        ),
        completed_steps=[],
        final_path_summary="",
        preconditions=preconditions,
        concretized_values={},
        error_start_is_real=True,
    )

    # Build a minimal AnalysisResult for the segment-by-segment selector.
    # _select_single_segment_path accesses: parsed_report, reduced_code,
    # source_root, annotated_path, call_chain_context, bug_path.
    from types import SimpleNamespace
    analysis_result = SimpleNamespace()
    analysis_result.parsed_report = parsed_report
    analysis_result.source_root = str(source_root)
    analysis_result.reduced_code = ""
    analysis_result.annotated_path = ""
    analysis_result.call_chain_context = ""
    analysis_result.bug_path = None

    t0 = time.time()

    logger.info("Conflict-driven path selection with shortest-path candidates ...")
    exhausted = False
    combo_infeasible = False
    try:
        if combo is not None and path_space is not None:
            # Deterministic whole-path combination: build + conflict-check only.
            selections = analyzer._select_combination_segments(
                analysis_result, completed, segments, path_space, combo,
            )
            if selections is None:
                combo_infeasible = True
        else:
            # LLM-driven selection (attempt-1 seed).  No blocked_candidates:
            # whole-path combination enumeration (in the caller) replaces the
            # per-segment candidate blocking that over-pruned.
            selections = analyzer._select_feasible_path_segments(
                analysis_result, completed, slice_mask,
                max_rounds=3,
                segments=segments,
                cfg_cache=cfg_cache,
                hard_constraints=hard_constraints,
                blocked_paths=blocked_paths,
                blocked_candidates=[],
            )
    except PathSpaceExhausted as e:
        # Exhaustion = no NEW distinct top-k whole-path remains (all candidates
        # of some segment are blocked).  This is NOT a conflict-based
        # infeasibility — it does not prove the report is FP, so we must NOT
        # fall through to the caller's hard-FP branch.  Signal it distinctly;
        # the caller routes it to bounded-exploration aggregation.
        logger.info("Path space exhausted: %s", e)
        exhausted = True
        selections = None

    elapsed = time.time() - t0

    stats = {
        "elapsed_sec": round(elapsed, 1),
        "num_segments": len(segments),
        "selected": selections is not None,
        "exhausted": exhausted,
        "combo_infeasible": combo_infeasible,
    }

    if selections:
        stats["num_selections"] = len(selections)
    elif combo_infeasible:
        stats["result"] = "Combination globally infeasible (whole-path FP) — not a report-level FP"
    elif exhausted:
        stats["result"] = "Exhausted top-k path space (no NEW distinct path) — not a conflict-based FP"
    else:
        stats["result"] = "FP — no globally consistent path"

    return selections, stats


def path_signature(selections):
    """Stable, human-readable signature of a whole-path branch combination.

    Used to mark a path as *blocked* after its simulation came back
    conflicted/unresolved, so a later path-selection attempt (with
    ``blocked_paths``) avoids re-deriving the same combination.
    """
    parts = []
    for sel in (selections or []):
        seg_idx = sel.get("segment_index", "?")
        bds = sel.get("branch_decisions", [])
        b = ",".join(
            f"L{d.get('line', d.get('source_line', '?'))}:"
            f"{'T' if d.get('branch_taken', True) else 'F'}"
            for d in bds
        )
        parts.append(f"seg{seg_idx}[{b}]")
    return ";".join(parts)


def _blocked_candidate_map(selections):
    """Per-segment ``{segment_index: selected_candidate}`` for a selection.

    Used alongside ``path_signature`` so a blocked path can be re-avoided
    DETERMINISTICALLY (by excluding the chosen candidate indices from the next
    selection), not just via the advisory prompt hint.

    FORCED segments are excluded: a segment with ``num_candidates <= 1`` (or
    linear, carrying no ``selected_candidate``) has exactly one path that every
    whole-path must pass through.  Blocking it would eliminate *every*
    whole-path and yield a premature FP, without ever checking whether the
    other (free) segments can combine to reach the bug.  Forced segments are
    fixed constraints of the search, never blocked — FP for them must come from
    cross-segment conflict (or bounded exploration of the free segments), not
    from "blocking" the only path.
    """
    return {
        sel.get("segment_index"): sel.get("selected_candidate")
        for sel in (selections or [])
        if sel.get("selected_candidate")
        and (sel.get("num_candidates", 0) or 0) > 1
    }


def _last_round_reason(sim_res) -> str:
    """The last simulation round's own reason text (``""`` if there is none)."""
    rounds = (sim_res or {}).get("rounds") or []
    if not rounds:
        return ""
    sim = rounds[-1].get("sim")
    return str(getattr(sim, "reason", "") or "")


def _tp_verdict_reason(leak_tp: bool = False, round_reason: str = "",
                       leak_reason: str = "") -> str:
    """Why the report is a TP — the reason a TP verdict carries (item C).

    The TP exit used to leave ``final_reason`` untouched, while the FP exits all
    set it.  Since ``final_reason`` lives outside the attempt loop, a report whose
    attempt 1 concluded FP (e.g. the leak layer's "stored/owned on well-formed
    paths" argument) and whose attempt 2 concluded TP printed attempt 1's FP
    reasoning next to a TP verdict, in both the summary and the result JSON.
    Every verdict now owns its reason.
    """
    base = str(round_reason or "").strip()
    if leak_tp:
        head = (
            "leak 层判定 TP：对象在 well-formed 调用路径上被 abandon 且无人接管"
            "（未释放、未移交 owner/return）→ 真实可达泄漏"
        )
        base = str(leak_reason or base).strip()
    else:
        head = "模拟执行判定触发 bug（测试用例复现 CSA 报告路径）"
    return f"{head}。模拟依据：{base[:300]}" if base else head


def _verify_path(analyzer, report_path, source_root, out_path, report_stem,
                 selections=None):
    """7a–7e: generate a POC (LLM-only, no real compile), LLM-audit its
    correctness, and LLM-simulate execution with a feedback loop.

    Returns the outcome WITHOUT mutating the verdict — the caller's path-traversal
    loop decides TP / FP / UNKNOWN based on ``conclusion``:

        {"poc_result", "conclusion": "tp"|"fp"|"unresolved"|None,
         "final_reason", "poc_code", "poc_path", "sim_res"}

    ``conclusion is None`` means verification errored (environmental), not a
    classification verdict.
    """
    poc_result: dict | None = None
    sim_res = None
    poc_code = None
    poc_path = None
    final_reason = None

    # HACK: set project_name to enable project-specific compile config
    if not analyzer._project_name:
        analyzer._project_name = source_root.name

    try:
        # 7a: Parse report + extract source context via analyzer
        t0 = time.time()
        analysis_poc = analyzer.analyze(str(report_path), str(source_root))
        print(f"  7a. Source context extracted [{time.time()-t0:.1f}s]")

        # 7b: Constraint completion → CompletedPath.  Anchored to the selected
        # feasible path so a different selection yields a different POC (P1).
        t0 = time.time()
        print("  7b. Running constraint completion (LLM) ...")
        completed_path = analyzer._complete_constraints(
            analysis_poc, selected_branches=selections,
        )
        print(f"  7b. Constraint completion done: "
              f"{len(completed_path.preconditions)} preconditions, "
              f"{len(completed_path.concretized_values)} concretized values "
              f"[{time.time()-t0:.1f}s]")
        if completed_path.preconditions:
            for p in completed_path.preconditions[:5]:
                print(f"      - {p}")
        if completed_path.concretized_values:
            for k, v in list(completed_path.concretized_values.items())[:5]:
                print(f"      - {k} = {v}")

        # 7c: Generate POC — LLM only, NO real compilation. Correctness is
        # audited by the LLM during the simulated execution.
        t0 = time.time()
        print("  7c. Generating POC (LLM, no compilation) ...")
        poc_code = analyzer.generate_poc(
            completed_path, analysis_poc, selected_branches=selections,
        )
        poc_elapsed = time.time() - t0

        poc_result = {
            "compiled": False,
            "compile_attempts": 0,
            "compile_mode": "llm_only",
            "elapsed_sec": round(poc_elapsed, 1),
            "poc_code_lines": len(poc_code.splitlines()) if poc_code else 0,
            "compile_log": [],
        }

        if poc_code:
            print(f"      POC code: {len(poc_code.splitlines())} lines — "
                  f"LLM-generated, correctness audited during simulation")
            poc_path = out_path.parent / f"poc_{report_stem}.cpp"
            poc_path.write_text(poc_code)
            print(f"      POC saved to: {poc_path}")
        else:
            print("      POC generation returned no code.")

        # 7d: LLM simulated execution + feedback loop.
        if not poc_code:
            poc_result["simulation"] = {
                "performed": False,
                "triggered": False,
                "verified_status": "NONE",
                "reason": "No POC generated — simulation skipped",
            }
            conclusion = "unresolved"
        else:
            t0 = time.time()
            print("  7d. LLM simulated execution + feedback loop ...")
            sim_verifier = SimulationVerifier(analyzer)
            sim_res = sim_verifier.verify_with_feedback(
                poc_code, completed_path, analysis_poc, max_rounds=3,
                selected_branches=selections,
            )
            sim_elapsed = time.time() - t0
            conclusion = sim_res.get("conclusion", "unresolved")
            verified_status = (
                "VERIFIED" if conclusion == "tp" else "UNVERIFIED"
            )
            poc_result["simulation"] = {
                "performed": True,
                "compiled": False,
                "compile_mode": "llm_only",
                "triggered": sim_res["triggered"],
                "conclusion": conclusion,
                "verified_status": verified_status,
                "num_rounds": len(sim_res["rounds"]),
                "elapsed_sec": round(sim_elapsed, 1),
                "test_case_correct": sim_res["rounds"][-1]["sim"].test_case_correct,
                "test_case_issues": sim_res["rounds"][-1]["sim"].test_case_issues,
                "path_conformance_ok": sim_res["rounds"][-1]["sim"].path_conformance_ok,
                "parse_degraded": sim_res["rounds"][-1]["sim"].parse_degraded,
                "rounds": [
                    {
                        "round": r["round"],
                        "triggered": r["sim"].triggered,
                        "reason": r["sim"].reason,
                        "trace": r["sim"].trace,
                        "blocking_reason": r["sim"].blocking_reason,
                        "suggested_input_change": r["sim"].suggested_input_change,
                        "csa_contradiction": r["sim"].csa_contradiction,
                        "fp_likelihood": r["sim"].fp_likelihood,
                        "test_case_correct": r["sim"].test_case_correct,
                        "test_case_issues": r["sim"].test_case_issues,
                        "path_conformance": r["sim"].path_conformance,
                        "path_conformance_ok": r["sim"].path_conformance_ok,
                        "parse_degraded": r["sim"].parse_degraded,
                        # the leak verdict's raw evidence (memory-leak reports only), so a
                        # report-level verdict can be audited after the fact
                        "leak_chain": r["sim"].leak_chain,
                    }
                    for r in sim_res["rounds"]
                ],
                "leak_verdict": sim_res.get("leak_verdict"),
                "final_reason": sim_res.get("final_reason"),
            }
            # Step-by-step path conformance (hallucination guard): print, per
            # round, whether the POC reproduced each CSA report path step.
            for r in sim_res["rounds"]:
                rsim = r["sim"]
                conf = getattr(rsim, "path_conformance", None) or []
                ok = getattr(rsim, "path_conformance_ok", True)
                if not conf:
                    if getattr(rsim, "parse_degraded", False):
                        print(f"      路径一致性 [第{r['round']}轮]: ⚠️ 响应被截断/"
                              f"审计不可用 — 逐步审计缺失，本轮不足以判定 TP")
                    continue
                n_bad = sum(1 for c in conf if not bool(c.get("satisfied", True)))
                flag = "✅ 一致" if ok else f"❌ 不一致({n_bad} 步)"
                print(f"      路径一致性 [第{r['round']}轮]: {flag} "
                      f"— {len(conf) - n_bad}/{len(conf)} 步与 CSA 报告一致")
                for c in conf:
                    if bool(c.get("satisfied", True)):
                        continue
                    print(f"        ✗ 步骤 {c.get('step', '?')}: "
                          f"报告预期={str(c.get('report_expectation', '?'))[:80]} | "
                          f"POC 实际={str(c.get('poc_result', '?'))[:80]}")

            if sim_res["triggered"]:
                print(f"      TP 模拟验证: ✅ 触发 bug [{sim_elapsed:.1f}s]")
            else:
                if conclusion == "fp":
                    print(f"      TP 模拟验证: ⚠️ 模拟执行判定 FP → 停止细化，翻转判定为 FP "
                          f"[{sim_elapsed:.1f}s]")
                else:
                    print(f"      TP 模拟验证: ⚠️ 未触发且无明确 FP 结论 → UNVERIFIED "
                          f"[{sim_elapsed:.1f}s]")
                if sim_res.get("final_reason"):
                    print(f"          依据: {sim_res['final_reason'][:200]}")

        if conclusion == "fp":
            final_reason = (
                (sim_res.get("final_reason") if sim_res and sim_res.get("final_reason")
                 else poc_result.get("simulation", {}).get("final_reason", ""))
            )
        elif conclusion == "tp":
            # The TP exit must own its reason too (see `_tp_verdict_reason`): leaving it
            # unset let a report whose earlier attempt had concluded FP print/carry that
            # attempt's FP (or leak-FP) reason next to a TP verdict.
            final_reason = (
                (sim_res or {}).get("final_reason")
                or _tp_verdict_reason(
                    leak_tp=(sim_res or {}).get("leak_verdict") == "tp",
                    round_reason=_last_round_reason(sim_res),
                )
            )

        return {
            "poc_result": poc_result,
            "conclusion": conclusion,
            "final_reason": final_reason,
            "poc_code": poc_code,
            "poc_path": poc_path,
            "sim_res": sim_res,
            "leak_verdict": (sim_res or {}).get("leak_verdict"),
        }

    except Exception as e:
        logger.error("POC verification failed: %s", e, exc_info=True)
        poc_result = {"compiled": False, "error": str(e)}
        print(f"  ❌ POC verification error: {e}")
        return {
            "poc_result": poc_result,
            "conclusion": None,
            "final_reason": str(e),
            "poc_code": None,
            "poc_path": None,
            "sim_res": None,
        }


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════


def process_report(args, report_path):
    """Run the full pipeline for a single report and write its result JSON.

    Returns the output dict (with a ``verdict`` field) for batch aggregation.
    """
    # Resolve paths
    report_path, cfg_cache_path, pdg_path, source_root = _resolve_paths(args, report_path)
    total_start = time.time()

    # Verdict story: which stage judged TP/FP, plus the supporting evidence.
    verdict_steps: list[str] = []          # chronological stage notes
    final_step: str = ""                   # the decisive stage name
    final_reason: str = ""                 # why TP/FP (FP cause or TP evidence summary)
    poc_code: str = ""                     # generated test case (empty if no POC obtained)
    poc_path = None                        # file the test case was saved to

    print("=" * 70)
    print("END-TO-END PATH SELECTION PIPELINE")
    print("=" * 70)
    print(f"Report:      {report_path}")
    print(f"Project:     {source_root}      ← 项目源树（报告/源在此）")
    print(f"  └ projects-dir: {args.projects_dir}")
    print(f"PDG:         {pdg_path}")
    print(f"CFG Cache:   {cfg_cache_path}")
    print(f"Algorithm:   conflict-driven path selection + POC verification")
    print(f"  (项目源+报告默认在 {DEFAULT_PROJECTS_DIR}/<name>/；"
          f"pdg/cfg 缓存在仓库根 pdg_<name>.json + cfg_cache/<name>/ 下)")
    print()

    # ── Stage 1: parse the report, then assemble the project data ──
    t0 = time.time()
    parsed_report, bug_path = parse_report(report_path, source_root)
    entry_fn = parsed_report.metadata.function_name
    print(f"{stage_label(1)}: Parsed report — {entry_fn} | {parsed_report.metadata.bug_type} "
          f"({len(parsed_report.path_events)} events) [{time.time()-t0:.1f}s]")

    # ── Stage 1 (cont.): load the SDG + the CFG cache ──
    # The CFG cache is per translation unit, so start from the bug host's own
    # cache and pull in the rest on demand once segmentation reveals which
    # files the path actually visits (stage 2 below).
    t0 = time.time()
    cache_set = CFGCacheSet(
        project_name=source_root.name, source_root=str(source_root),
    )
    if cfg_cache_path is not None:
        cache_set.add_cache_file(cfg_cache_path)
    else:
        bug_source = _resolve_source_file_for_cfg(parsed_report, str(source_root))
        if not cache_set.ensure(bug_source):
            logger.warning(
                "no CFG cache for bug file %s — path files will be loaded on "
                "demand; preheat cfg_cache/%s/ offline to cover the rest",
                bug_source, source_root.name,
            )
    cfg_cache = cache_set.as_dict()
    sdg = load_sdg(pdg_path)
    print(f"{stage_label(1)}: Loaded CFG ({len(cfg_cache)} fn) + SDG ({len(sdg.pdgs)} fn) "
          f"[{time.time()-t0:.1f}s]")

    # ── Stage 2: segment the bug path ──
    t0 = time.time()
    segments = segment_path(parsed_report, sdg=sdg, source_root=str(source_root))
    print(f"{stage_label(2)}: Segmented into {len(segments)} segment(s) "
          f"[{time.time()-t0:.1f}s]")

    # ── Stage 2 (cont.): load the CFG of every file the path crosses ──
    # ``cfg_cache`` is the live merged view, so these show up for stage 5.
    t0 = time.time()
    new_tus = [seg.function_file for seg in segments
               if seg.function_file and cache_set.ensure(seg.function_file)]
    if new_tus:
        print(f"{stage_label(2)}: +{len(new_tus)} CFG cache(s) for path files "
              f"({len(cfg_cache)} fn total) [{time.time()-t0:.1f}s]")

    # Nothing anywhere: the bug file wasn't a TU and the path's own files are
    # unresolved (report without per-file sections).  Locate the TU that
    # defines the entry function as a last resort.
    if not cfg_cache and cfg_cache_path is None:
        fallback = _resolve_cfg_cache(source_root, entry_fn)
        if fallback is not None:
            logger.warning(
                "no CFG cache for any path file — falling back to %s "
                "(defines '%s')", fallback.name, entry_fn,
            )
            cache_set.add_cache_file(fallback)
    cache_set.warn_missing()

    # ── Stage 3: build the SliceMask (PDG backward slice) ──
    t0 = time.time()
    file_index = sdg.build_file_index()
    slice_mask, slice_result = build_slice_mask(
        sdg, parsed_report, bug_path, source_root, with_slice_result=True,
        file_index=file_index,
    )

    # Dense CSA decisions: which branch each predicate line took on the reported
    # path.  Used by the branch filter to expand only calls the path can reach —
    # a segment's line range routinely spans the then-body of a branch the CSA
    # took as false (`_find_calls_in_pdg_range` alone would expand it).
    # Deliberately NOT fed to the slicer: the slice's own BTBS pass keeps its
    # narrower CONTROL-only view.
    path_annotations = build_dense_path_annotations(
        bug_path, sdg, source_root=str(source_root), file_index=file_index,
    )
    print(f"{stage_label(3)}: {len(path_annotations)} CSA branch decision line(s)")


    # ── LLM provider + FPAnalyzer (infrastructure for stages 4 and 9) ──
    # Created here (before the branch filter) so the assumption-derived state
    # can feed into conflict checking (Layers 1b/1.5), which previously never
    # received any established_state and therefore pruned nothing.
    api_key = os.environ["DEEPSEEK_API_KEY"]
    config = LLMConfig(
        provider_name="deepseek",
        api_key=api_key,
        default_model="deepseek-chat",
        default_timeout=120,
    )
    # FPAnalyzer creates LLMProvider internally from config
    analyzer = FPAnalyzer(
        llm_config=config,
        source_root=str(source_root),
        pdg_path=str(pdg_path),
    )
    print(f"  FPAnalyzer ready (model={analyzer._model}) [{time.time()-t0:.1f}s]")

    # ── Stage 4: assumption feasibility (runs before the branch filter) ──
    t0 = time.time()
    global_constraints = analyzer._load_global_constraints(
        str(source_root),
        target_classes=analyzer._extract_class_names_from_path(parsed_report),
    )
    hard_constraints, cumulative_state = analyzer.check_assumptions_feasibility(
        segments=segments,
        parsed=parsed_report,
        global_constraints=global_constraints,
    )
    assumption_elapsed = time.time() - t0
    n_hc = len(hard_constraints)
    n_state = len(cumulative_state)
    print(f"{stage_label(4)}: Assumption check — {n_hc} hard constraint(s), "
          f"{n_state} state binding(s) [{assumption_elapsed:.1f}s]")
    for hc in hard_constraints:
        print(f"    🔒 {hc}")
    for var, val in cumulative_state.items():
        print(f"    📎 {var}: {val}")

    # ── Stage 5: branch filter (seeded with assumption-derived state) ──
    t0 = time.time()
    branch_results, insufficient_counts = run_branch_filter(
        segments, parsed_report, cfg_cache, sdg, slice_mask, source_root,
        established_state=cumulative_state, cache_set=cache_set,
        path_annotations=path_annotations,
    )
    print(f"{stage_label(5)}: Branch filter ({len(branch_results)} segments) "
          f"[{time.time()-t0:.1f}s]")

    # ── Print branch filter summary ──
    total_b = sum(r["num_branches"] for r in branch_results)
    total_con = sum(r["num_consistent"] for r in branch_results)
    total_ctr = sum(r["num_contradictory"] for r in branch_results)
    total_ins = sum(r["num_insufficient"] for r in branch_results)
    print(f"  Branches: {total_b} total | {total_con} consistent | {total_ctr} pruned | {total_ins} insufficient")

    # ── Stage 6: report self-contradiction gate (deterministic, no LLM) ──
    # The report's OWN path states a direction for every branch condition it
    # walks.  When one condition is asserted BOTH ways on that path, with no
    # write to its variables in between, no input can realize the path: the
    # report is an FP, provable with no path enumeration, no POC and no model
    # call — settled here, ahead of path selection, POC generation and the
    # simulated execution (the stage-4 assumption check is the only LLM call
    # that has run by this point).  CSA cannot refute such a pair itself when an
    # operand comes from an out-of-line function (faiss' `fourcc`): its two call
    # results are independent free symbols to the analyzer's solver, which is
    # why the report was emitted at all.
    # `fp_analysis/report_consistency.py` documents the soundness guards (same
    # function+file, different lines, no intervening write, no call step
    # between the claims, unambiguous line pairing) — every one fails closed, so
    # a missed contradiction only degrades to the pipeline's usual behaviour.
    report_contradictions: list = []
    try:
        report_contradictions = detect_report_contradictions(
            segments, parsed_report.path_events, source_root=str(source_root))
    except Exception as e:  # noqa: BLE001 — gate must never be fatal
        logger.warning("Report self-contradiction gate failed: %s", e)
        report_contradictions = []
    if report_contradictions:
        _c0 = report_contradictions[0]
        print(f"\n{stage_label(6)}: report path self-contradiction — "
              f"{len(report_contradictions)} pair(s) "
              f"[{time.time()-t0:.1f}s]")
        for _c in report_contradictions[:3]:
            print(f"    ✗ {_c.condition} — L{_c.first.line}"
                  f"({'TRUE' if _c.first.direction else 'FALSE'}, "
                  f"event #{_c.first.event_number}) vs L{_c.second.line}"
                  f"({'TRUE' if _c.second.direction else 'FALSE'}, "
                  f"event #{_c.second.event_number})")
        _fp_reason = _c0.reason()
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print("  Verdict:    ❌ FP (no feasible path — the report contradicts itself)")
        print(f"  Reason:     {_fp_reason}")
        _out = report_out_path(args, report_path)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _output = {
            "report": report_path.name,
            "algorithm": "conflict-driven",
            "entry_function": entry_fn,
            "bug_type": parsed_report.metadata.bug_type,
            "report_self_contradiction": {
                "decisive_fp": True,
                "reason": _fp_reason,
                "condition": _c0.condition,
                "variables": list(_c0.variables),
                "first": {
                    "line": _c0.first.line,
                    "direction": _c0.first.direction,
                    "event_number": _c0.first.event_number,
                },
                "second": {
                    "line": _c0.second.line,
                    "direction": _c0.second.direction,
                    "event_number": _c0.second.event_number,
                },
                "function": _c0.first.function_name,
                "file": _c0.first.source_file,
                "pairs": [
                    {
                        "condition": c.condition,
                        "line": c.first.line,
                        "event_number": c.first.event_number,
                        "other_line": c.second.line,
                        "other_event_number": c.second.event_number,
                    }
                    for c in report_contradictions
                ],
            },
            "verdict": "FP",
            "verdict_step": stage_step(6, "（同一条件同路径反向）"),
            "verdict_reason": _fp_reason,
            "verdict_steps": [f"报告自相矛盾判定: {_fp_reason}"],
            "total_elapsed_sec": round(time.time() - total_start, 1),
        }
        _out.write_text(json.dumps(_output, indent=2, default=str))
        print(f"\nFull results saved to: {_out}")
        print(f"Total pipeline time: {time.time() - total_start:.1f}s")
        return _output

    # ── Stage 7 preamble: inject assumption-derived state as preconditions ──
    # (the stage-4 state becomes part of the selection prompt in stage 9)
    extra_preconditions: list[str] = []
    if cumulative_state:
        state_lines = "\n".join(
            f"  - {var}: {val}" for var, val in cumulative_state.items()
        )
        extra_preconditions.append(
            f"[Assumption-derived state from Error Start / CSA assumptions]\n"
            f"{state_lines}"
        )

    # ── Stage 7 (supplement, advisory): guard-pinned / finite-domain variables ──
    # CSA (and the tool's own constraint model) treat some variables — e.g. a file-read
    # header discriminator compared against an out-of-line constant factory such as
    # `fourcc("rrot")` in another TU — as able to take *any* value, when source actually
    # pins them to a small, fixed constant set.  Surfacing that to the path-selection and
    # simulation LLM lets it conclude reachability is impossible where CSA saw none.
    # This half is advisory prose only; the decisive gate is immediately below.
    try:
        target_fns = {
            s.function_name for s in segments if getattr(s, "function_name", "")
        }
        target_fns.add(parsed_report.metadata.function_name)
        domain_facts = extract_domain_facts(sdg, slice_mask, target_fns)
        # Default: no facts (nothing appended to selection prompt nor the simulator).
        analyzer._domain_facts_text = ""
        if domain_facts:
            fact_text = format_domain_facts(domain_facts)
            extra_preconditions.append(
                "[Guard-pinned / finite-domain variables (semantic completion)]\n"
                f"{fact_text}"
            )
            # Also stash on the analyzer so the simulation auditor (reuses this same
            # FPAnalyzer) can append the facts to its own prompt (see simulate_execution).
            analyzer._domain_facts_text = fact_text
            print(f"{stage_label(7)}: domain supplement — {len(domain_facts)} guard-pinned "
                  f"variable fact(s) surfaced to LLM")
            for df in domain_facts:
                ffn = df.function_name.split("::")[-1] or df.function_name
                print(f"    🎯 {df.variable} ({ffn}): {len(df.folded)} distinct constants")
    except Exception as e:  # noqa: BLE001 — best-effort supplement, never fatal
        print(f"{stage_label(7)}: domain supplement skipped ({e})")

    # ── Stage 7: decisive semantic finite-domain conflict gate (pre-enumeration) ──
    # Before enumerating the (possibly huge) path space, ask the LLM whether the
    # reported path forces a guard-pinned / finite-domain variable (e.g. a fourcc
    # tag `h`, which CSA treats as free because `fourcc` is an opaque/out-of-line
    # constant factory) into a self-contradiction — admitted into the deref region
    # only when `h ∈ S`, yet every pointer-assigning dispatch arm taken FALSE so the
    # pointer stays null.  If so the reported bug state is UNREACHABLE => decisive
    # FP, concluded BEFORE path selection.  Best-effort + conservative: a single
    # LLM call; on any failure or non-decisive answer we proceed to normal selection.
    semantic_fp_reason: str | None = None
    try:
        semantic_fp_reason = analyzer.semantic_fp_reason(parsed_report)
    except Exception as e:  # noqa: BLE001 — gate must never be fatal
        logger.warning("Semantic finite-domain FP gate failed: %s", e)
        semantic_fp_reason = None
    if semantic_fp_reason:
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print("  Verdict:    ❌ FP (no feasible path — semantic finite-domain conflict)")
        print(f"  Reason:     {semantic_fp_reason}")
        _out = report_out_path(args, report_path)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _output = {
            "report": report_path.name,
            "algorithm": "conflict-driven",
            "entry_function": entry_fn,
            "bug_type": parsed_report.metadata.bug_type,
            "semantic_feasibility": {
                "decisive_fp": True,
                "reason": semantic_fp_reason,
            },
            "verdict": "FP",
            "verdict_step": stage_step(7, "（约束补全，枚举前判定）"),
            "verdict_reason": semantic_fp_reason,
            "verdict_steps": [f"约束补全判定: {semantic_fp_reason}"],
            "total_elapsed_sec": round(time.time() - total_start, 1),
        }
        _out.write_text(json.dumps(_output, indent=2, default=str))
        print(f"\nFull results saved to: {_out}")
        print(f"Total pipeline time: {time.time() - total_start:.1f}s")
        return _output

    report_stem = report_path.stem
    out_path = report_out_path(args, report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(stage_label(8))
    print(f"{'='*70}")

    # ── Path traversal with a third outcome (UNKNOWN).  When a selected path's
    # simulation comes back conflicted/unresolved, we abandon that path, mark it
    # blocked, and re-select a DIFFERENT feasible path, up to ``max_path_attempts``.
    # Exceeding the threshold with no decisive TP/FP → UNKNOWN (path space too
    # large / undecided). ──
    #
    # Diversity mechanism = SYSTEMATIC WHOLE-PATH COMBINATION ENUMERATION.  We
    # collect the per-segment feasible candidates once (deterministic, no LLM)
    # and walk their cross-product on re-selection.  We block WHOLE-PATH
    # combinations (never a per-segment branch): infeasibility is combination-
    # dependent — (segA=b1, segB=c1) being FP does not make b1 infeasible, since
    # (segA=b1, segB=c2) may be feasible.  So a path's FP blocks that exact
    # combination and the next enumeration step tries a different one.
    max_path_attempts = getattr(args, "max_path_attempts", 3) or 3
    blocked_paths: list[str] = []
    path_space_enum = collect_path_space(analyzer, segments)

    # ── Slice-driven branch relevance: fix root-cause-irrelevant branches and
    # enumerate only the relevant ones.  Compressing the space both avoids wasted
    # enumeration over branches that cannot change the verdict and, when the
    # reduced bound collapses to 1, lets a single decisive FP generalize to the
    # whole space (graph evidence, not text matching). ──
    bug_file = parsed_report.metadata.bug_file
    slice_truncated = bool(getattr(slice_result, "truncated", False))
    slice_relevance = classify_branch_relevance(
        sdg, slice_mask, path_space_enum,
        bug_file=bug_file, truncated=slice_truncated,
    )
    reduced_ps, slice_reduce_stats = reduce_path_space(
        path_space_enum, slice_relevance["node_relevance"], segments,
    )
    # Context-insensitive collapse: segments carrying the IDENTICAL branch set
    # (the same function's branches re-appearing across multiple segments, e.g.
    # insertThousandsGroupingUnsafe's loop B5/B8/B10 in 4 segments) become ONE
    # enumeration dimension — their directions are bound together (全路径一致),
    # collapsing the whole-path cross-product from ∏counts to the distinct
    # branch-set product (GlogFormatter: 4096 → 8).
    reduced_ps = annotate_ci_groups(reduced_ps, segments)
    ci_stats = {
        "collapsed_bound": int(reduced_ps.get("whole_path_bound", 1)),
        "num_groups": len(reduced_ps.get("ci_groups", [])),
        "grouped_segments": reduced_ps.get("ci_groups", []),
    }
    reduced_bound_one = reduced_ps.get("whole_path_bound", 1) == 1

    # Exploration budget.  When the slice-compressed relevant space is small we
    # explore the WHOLE space (full-space exploration — the user's intent), not
    # just ``max_path_attempts``; only a still-huge space falls back to the
    # max_path_attempts bound.  A decisive FP then generalizes over the whole
    # enumerated relevant space, not a 3-of-16 sample.
    _FULL_EXPLORE_THRESHOLD = 32
    # (There used to be a `_STRUCTURAL_GP_MAX = 1000` cap here, gating the
    # structural-FP cause-invariance generalization on the reduced bound.  It was
    # a *soundness* cap, not a cost one: `cause_invariant` enumerated the
    # cross-product, which combination_enum bounds (200k enumerated / 100k
    # yielded), so above the cap the "invariant" answer would have come from a
    # partial scan.  The scan is now exhaustive without enumerating the
    # cross-product, so a decisive structural FP generalizes inside a huge space
    # too — that is what makes it possible to conclude FP instead of UNKNOWN
    # there.  Do not reintroduce a bound-based cap without re-checking that.)
    reduced_bound = int(reduced_ps.get("whole_path_bound", 1))
    if reduced_bound <= _FULL_EXPLORE_THRESHOLD:
        loop_attempts = max(reduced_bound, 1)
        full_space_explore = reduced_bound > 1  # bound==1 handled by generalization
    else:
        loop_attempts = max_path_attempts
        full_space_explore = False
    combo_iter = iter_combinations(
        reduced_ps, segments,
        cap=max(loop_attempts, max_path_attempts * 8, 16),
    )
    verdict: str | None = None
    final_step: str | None = None
    final_reason: str | None = None
    selections = None
    ps_stats: dict = {}
    poc_result: dict | None = None
    poc_code = None
    poc_path = None

    # Exploration state: a decisive FP on ONE path never concludes the report is
    # FP (that path may just be the wrong one).  We collect evidence across the
    # distinct feasible paths we explore (blocking each FP signature and
    # re-selecting another), then aggregate at the end.
    saw_unresolved = False
    fp_observations: list[dict] = []  # {signature, final_reason}
    # Leak-layer (memory-leak reports) verdicts per attempt.  The leak layer settles a
    # leak from the object's ownership chain — a code-level claim, not a per-path
    # execution result — so the report-level loop requires the attempts to AGREE
    # before either direction is accepted (see the post-loop consensus).  Keeping them
    # here also keeps the verdict independent of the attempt order.
    leak_fp_reasons: list[str] = []
    leak_tp_reasons: list[str] = []

    combo = None
    combo_sig = None
    print(f"\n{'='*70}")
    print(stage_label(9))
    print(f"{'='*70}")
    for attempt in range(1, loop_attempts + 1):
        print(f"\n── Path-selection attempt {attempt}/{loop_attempts} "
              f"(blocked paths: {len(blocked_paths)})"
              + (" — 全空间探索(压缩相关空间小)" if full_space_explore else "")
              + " ──")
        if attempt == 1:
            # Best-feasible-first LLM path (uses global constraints/state context).
            combo = None
            combo_sig = None
            selections, ps_stats = run_path_selection(
                analyzer, parsed_report, segments, slice_mask,
                source_root=source_root, cfg_cache=cfg_cache,
                extra_preconditions=extra_preconditions,
                hard_constraints=hard_constraints,
                blocked_paths=blocked_paths,
            )
        else:
            # Systematic whole-path combination enumeration: pull the next untried
            # combination (skip ones already blocked as whole-path), build +
            # conflict-check it deterministically.
            combo = None
            combo_sig = None
            while True:
                nxt = next(combo_iter, None)
                if nxt is None:
                    break
                c, _sel = nxt
                csig = combo_signature(reduced_ps, c)
                if csig in blocked_paths:
                    continue
                combo, combo_sig = c, csig
                break
            if combo is None:
                # Every whole-path combination tried → not a conflict-based FP;
                # route to the bounded-exploration aggregation.
                print(f"  attempt {attempt}: ⚠️ 全部整路径组合已探索 → 交由探索聚合判定")
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → 全部整路径组合已探索，交由路径探索聚合判定"
                )
                break  # verdict stays None → post-loop aggregation
            selections, ps_stats = run_path_selection(
                analyzer, parsed_report, segments, slice_mask,
                source_root=source_root, cfg_cache=cfg_cache,
                extra_preconditions=extra_preconditions,
                hard_constraints=hard_constraints,
                blocked_paths=blocked_paths,
                combo=combo,
                path_space=reduced_ps,
            )

        if selections is None:
            if ps_stats.get("combo_infeasible"):
                # THIS whole-path combination is globally infeasible → block that
                # combination and enumerate the next.  NOT a report-level FP: a
                # different combination may be feasible (or reach the bug).
                sig = combo_sig or path_signature([])
                blocked_paths.append(sig)
                fp_observations.append({
                    "signature": sig,
                    "final_reason": "组合级全局冲突（该整路径组合不可行）",
                })
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → 组合 {combo} 全局不可行，"
                    f"屏蔽该整路径并继续枚举其它组合"
                )
                print(f"  attempt {attempt}: ⚠️ 组合 {combo} 全局不可行 → 屏蔽并继续枚举 "
                      f"(blocked={len(blocked_paths)})")
                poc_result = None
                continue
            if ps_stats.get("exhausted"):
                # No NEW distinct whole-path remains.  This is NOT a conflict-based
                # infeasibility — it does not prove the report is FP: there may be
                # untried combinations that reach the bug.  Route to the post-loop
                # bounded-exploration aggregation (FP only if every explored
                # distinct path was decisive-FP with no unresolved).
                print(f"  attempt {attempt}: ⚠️ 路径空间已穷尽（无新的不同组合）"
                      f" → 交由探索聚合判定")
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → 路径空间已穷尽，交由路径探索聚合判定"
                )
                break  # verdict stays None → post-loop aggregation
            # Genuine conflict-based infeasibility (conflict checker / stuck):
            # no globally-consistent feasible path exists → FP (sound).
            final_step = stage_label(9)
            final_reason = ps_stats.get("result", "未找到全局一致可行路径")
            verdict_steps.append(f"{stage_label(9)} → {final_reason}（FP）")
            verdict = "FP"
            break

        print(f"  attempt {attempt}: ✅ 找到全局一致可行路径（初始 TP）")
        print(f"\n  Segment Selections ({len(selections)} segments):")
        for sel in selections:
            seg_idx = sel.get("segment_index", "?")
            state = sel.get("established_state", "")
            if isinstance(state, str) and len(state) > 120:
                state = state[:120] + "..."
            branches = sel.get("branch_decisions", [])
            if branches:
                print(f"    Seg {seg_idx}: {len(branches)} branch decision(s)")
                for b in branches[:3]:  # show first 3
                    if isinstance(b, dict):
                        print(f"      - {b.get('node_id', '?')}: {b.get('branch_taken', '?')}")
            else:
                print(f"    Seg {seg_idx}: linear (no branch decisions)")

        # ── Generate POC + LLM simulate + feedback (see _verify_path) ──
        print(f"\n{'='*70}")
        print("POC VERIFICATION (generate + LLM correctness audit + simulate; no compile)")
        print(f"{'='*70}")
        vres = _verify_path(
            analyzer, report_path, source_root, out_path, report_stem,
            selections=selections,
        )
        poc_result = vres["poc_result"]
        poc_code = vres["poc_code"]
        poc_path = vres["poc_path"]
        concl = vres["conclusion"]
        leak_v = vres.get("leak_verdict")

        if concl == "tp":
            if leak_v == "tp" and not leak_fp_reasons and attempt < loop_attempts:
                # A leak-layer TP is a CODE-level ownership JUDGMENT, not
                # execution evidence (`triggered`), and the leak chain is read once
                # per path attempt — on read_index-484-1 the same function came back
                # leak-FP on one attempt and leak-TP on another, so accepting the
                # first decisive attempt made the verdict depend on attempt order
                # alone.  Block this path and keep exploring: the leak layer is
                # accepted once the conforming attempts agree (post-loop below).
                sig = path_signature(selections)
                blocked_paths.append(sig)
                leak_tp_reasons.append(vres.get("final_reason") or "")
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → leak 层判定 TP，屏蔽该路径并继续探索"
                    f"以求一致（leak 判定为整函数级结论，不由单次尝试定案）"
                )
                print(f"  attempt {attempt}: ✅ leak 层判定 TP → 屏蔽该路径并继续探索 "
                      f"以验证跨路径一致 (blocked={len(blocked_paths)})")
                poc_result = None
                continue
            final_step = stage_step(10, "模拟执行反馈")
            # Every verdict owns its reason (see `_tp_verdict_reason`): a TP row must
            # never inherit an earlier attempt's FP/leak-FP reasoning.
            final_reason = vres.get("final_reason") or _tp_verdict_reason(
                leak_tp=leak_v == "tp",
                round_reason=_last_round_reason(vres.get("sim_res")),
            )
            verdict_steps.append(
                stage_step(10, "模拟执行反馈 → leak 层一致性判定 TP（多路径一致）")
                if leak_v == "tp" else
                stage_step(10, "模拟执行反馈 → 测试用例模拟执行判定触发 bug（TP）")
            )
            verdict = "TP"
            break
        if concl == "fp":
            final_reason = vres["final_reason"] or (
                (poc_result or {}).get("simulation", {}).get("final_reason", "")
            )
            if leak_v == "fp":
                leak_fp_reasons.append(final_reason)
            if reduced_bound_one:
                # The compressed path space collapsed to a single root-cause
                # equivalence class: every enumerated branch is slice-confirmed
                # root-cause-irrelevant (its predicate has no data/control path to
                # the bug trigger).  No other feasible combination can therefore
                # change the verdict, so a decisive FP on THIS path generalizes to
                # the whole enumerated space — graph evidence, not a text match.
                sig = path_signature(selections)
                blocked_paths.append(sig)
                fp_observations.append({
                    "signature": sig, "final_reason": final_reason,
                })
                final_step = stage_step(9, "路径空间压缩推广")
                verdict_steps.append(
                    stage_step(9, "路径空间压缩推广 → 枚举分支全部根因无关"
                                  "（切片压缩 whole-path bound→1），"
                                  "该路径判定推广到整个路径空间 → FP")
                )
                ps_stats["generalized_fp"] = True
                verdict = "FP"
                print(f"  attempt {attempt}: ❌ 决定性 FP + 枚举分支全部根因无关"
                      f" → 压缩推广到整个路径空间 → FP (generalized)")
                break

            # A decisive STRUCTURAL FP — the CSA path is internally
            # self-contradictory (csa_contradiction) or the LLM marks it
            # definite_fp — is a report-level claim about the ROOT-CAUSE pointer
            # (e.g. "the deref'd pointer cannot be null under the real API
            # contract").  If no *relevant* branch that writes that pointer takes
            # a different direction anywhere in the reduced space, the
            # contradiction holds on every combination → the single decisive FP
            # generalizes to the whole (relevant) space (16→1 simulation).  This
            # is a deterministic PDG check, not a text match.  Soundness: it only
            # fires on a structural FP + a successfully-recovered pointer + a
            # non-truncated slice, and never re-opens a TP.
            #
            # The generalization is a property of the code and the reduced space, not
            # of a particular attempt, so it is attempted on EVERY decisive-FP attempt
            # (the reference path is the one that produced the decisive verdict).  The
            # old `attempt == 1` restriction silently dropped the generalization exactly
            # when re-selection was what revealed the root cause, and the old
            # `reduced_bound <= _STRUCTURAL_GP_MAX` cap existed only because the scan
            # used to be a bounded prefix of the cross-product; `cause_invariant` is now
            # exhaustive without enumerating it, so a decisive structural FP in an
            # astronomically large space can conclude FP instead of always UNKNOWN.
            structural_fp = any(
                r.get("csa_contradiction") or r.get("fp_likelihood") == "definite_fp"
                for r in ((poc_result or {}).get("simulation", {}).get("rounds") or [])
            )
            if structural_fp:
                var = extract_deref_pointer(parsed_report, sdg)
                if var and cause_invariant(
                    sdg, selections, reduced_ps, segments,
                    slice_relevance["node_relevance"], var,
                    truncated=slice_truncated, bug_file=bug_file,
                ):
                    final_step = stage_step(9, "结构性FP根因不变性推广")
                    verdict_steps.append(
                        stage_step(9, "结构性FP根因不变性推广 → 决定性结构性 FP"
                                      "（矛盾在 report 级 CSA 路径），"
                                      f"相关分支不写根因指针 {var} 且不随组合变化 → "
                                      "推广到整个相关路径空间 → FP")
                    )
                    ps_stats["structural_generalized_fp"] = True
                    verdict = "FP"
                    print(f"  attempt {attempt}: ❌ 决定性结构性 FP + 根因指针 {var} 不变"
                          f" → 推广到整个相关路径空间 → FP (structural generalized)")
                    break

            # A decisive FP on ONE path only shows THAT path is infeasible — it
            # does not prove the report is a false positive: other feasible
            # combinations might genuinely reach the bug.  So a single-path FP
            # never concludes the verdict by itself.  Instead we EXPLORE the
            # feasible path space: block this WHOLE-PATH combination (never a
            # per-segment branch — infeasibility is combination-dependent) and
            # enumerate the next distinct combination.  We only conclude FP after
            # the exploration loop has examined distinct combinations and found
            # none that is a genuine TP (see the post-loop aggregation).
            sig = path_signature(selections)
            blocked_paths.append(sig)
            fp_observations.append({
                "signature": sig,
                "final_reason": final_reason,
            })
            verdict_steps.append(
                f"{stage_label(9)} attempt {attempt} → 模拟执行判定单条路径 FP，"
                f"屏蔽该路径({sig})并重选其它可行路径继续探索"
            )
            print(f"  attempt {attempt}: ❌ 单路径判定 FP → 屏蔽该路径并探索其它路径 "
                  f"(blocked={len(blocked_paths)})")
            poc_result = None  # next attempt will rebuild
            if attempt < loop_attempts:
                continue
            # Fall through to post-loop aggregation (budget exhausted).
        if concl is None:
            # Verification errored (environmental, not a classification verdict):
            # keep the found path as TP (unverified), matching prior behavior.
            final_step = stage_label(9)
            verdict_steps.append(f"{stage_label(9)} → 找到全局一致可行路径（TP；验证出错未确认）")
            verdict = "TP"
            break

        # ── unresolved (conflict): abandon path, block it, retry a different one ──
        # Guarded to `concl == "unresolved"` — the FP branch already fell through
        # to post-loop aggregation when the budget is exhausted and must NOT be
        # re-classified as unresolved here.
        if concl == "unresolved":
            sig = path_signature(selections)
            blocked_paths.append(sig)
            saw_unresolved = True
            verdict_steps.append(
                f"{stage_label(9)} attempt {attempt} → 模拟执行冲突/无法触发，放弃并屏蔽该路径，重试"
            )
            print(f"  attempt {attempt}: ⚠️ 模拟执行冲突(unresolved) → 放弃该路径并重试 "
                  f"(blocked={len(blocked_paths)})")
        poc_result = None  # next attempt will rebuild

    # ── Post-loop aggregation (Tier 2) / Threshold exceeded → UNKNOWN ──
    # Reaching here means the loop ended with no decisive single-path TP/FP and
    # no path-invariance proof.  We aggregate the evidence gathered across the
    # distinct feasible paths we explored:
    #   - any path that triggered the bug        → TP (handled by immediate break)
    #   - FULL space exhausted AND all were FP  → FP (sound: no combo left untried)
    #   - bounded sample of a still-huge space  → UNKNOWN (unexplored combos may
    #     reach the bug, so a bounded all-FP sample does NOT prove FP)
    #   - any unresolved                        → UNKNOWN
    #
    # Note: a single-path FP NEVER concludes the verdict.  FP is only provable
    # when the slice-compressed relevant space was exhaustively enumerated
    # (`full_space_explore`: reduced_bound was small, ≤ _FULL_EXPLORE_THRESHOLD).
    # A BOUNDED exploration of a huge uncompressible space (reduced_bound >
    # threshold) samples only `loop_attempts` combos of an astronomically larger
    # set — every one being FP does not rule out a feasible path in the untried
    # combinations, so the honest verdict is UNKNOWN, not FP.
    if verdict is None:
        # ── Leak-layer consensus (memory-leak reports) ──────────────────────
        # The leak layer decides a leak from the object's ownership chain: WHO ends up
        # owning the object on each continuation, and whether the abandon needs
        # out-of-contract input.  That is a claim about the CALLEE's code, not about the
        # enumerated branch directions — the enumerated branches select which index
        # format is read, i.e. *which* object exists, not who owns it.  So unlike a
        # per-path simulation FP (which stays a sample of the space and therefore needs
        # `full_space_explore`), a leak verdict is already universal over the space and
        # does not need the space exhausted: on a combo where the object is never
        # allocated nothing leaks, and where it is allocated the chain the leak layer
        # classified is the chain that runs.
        #
        # It is accepted only on AGREEMENT: every conforming attempt that produced a leak
        # verdict must say the same thing (`leak_tp_reasons` empty for FP), with at least
        # two witnesses when the space allowed more than one attempt.  A single decisive
        # attempt was previously enough — and since the leak chain is re-read per attempt,
        # read_index-484-1 came back FP,FP,TP across attempts, so attempt ORDER alone
        # decided the verdict.  Disagreement voids the leak layer (the report then falls
        # through to the deterministic aggregation, i.e. UNKNOWN on a huge space) rather
        # than picking whichever attempt concluded first.
        min_witnesses = min(2, loop_attempts)
        leak_contradicted = bool(leak_tp_reasons and leak_fp_reasons)
        if leak_contradicted:
            leak_note = (
                f"leak 层跨路径判定自相矛盾（TP {len(leak_tp_reasons)} 次 / "
                f"FP {len(leak_fp_reasons)} 次）"
            )
            verdict_steps.append(
                stage_step(9, f"leak 层一致性 → {leak_note}，该层证据作废"))
            print(f"  ⚠️ {leak_note} → leak 层证据作废，交由路径空间聚合判定")
        if (
            not leak_tp_reasons
            and len(leak_fp_reasons) >= min_witnesses
        ):
            verdict = "FP"
            final_step = stage_step(9, "leak 层一致性推广")
            final_reason = (
                f"leak 层在 {len(leak_fp_reasons)} 条不同可行路径上一致判定 FP："
                f"对象在其 ownership chain 上被释放/移交 owner（或仅在损坏输入下 abandon），"
                f"枚举分支只改变读取的索引类型、不改变该对象的归属 → 整个路径空间 → FP。"
                f"依据：{leak_fp_reasons[0][:300]}"
            )
            verdict_steps.append(
                stage_step(9, f"leak 层一致性推广 → "
                              f"{len(leak_fp_reasons)} 条路径一致判定 FP → FP")
            )
            ps_stats["leak_consensus_fp"] = True
            print(f"  ❌ leak 层一致性推广：{len(leak_fp_reasons)} 条路径一致判定 FP → FP")
        elif fp_observations and not saw_unresolved and full_space_explore:
            verdict = "FP"
            final_step = stage_step(9, "路径探索聚合")
            final_reason = (
                f"全空间探索（压缩相关空间穷尽）: 已探索 {len(fp_observations)} 条不同可行路径，"
                f"均模拟判定为决定性 FP，整个压缩路径空间无可行解 → FP"
            )
            verdict_steps.append(
                stage_step(9, f"路径探索聚合 → 压缩相关路径空间已穷尽，"
                              f"已探索 {len(fp_observations)} 条路径均判定 FP → FP")
            )
        else:
            if not full_space_explore:
                undecided = f"路径空间过大未穷举（仅作有界探索 {len(fp_observations)} 条）"
            elif saw_unresolved:
                undecided = f"探索中存在 unresolved（{len(fp_observations)} 条 FP 观察）"
            else:
                undecided = f"探索后仍无确定结果（{len(fp_observations)} 条 FP 观察）"
            if leak_contradicted:
                undecided += f"；{leak_note}"
            verdict = "UNKNOWN"
            final_step = stage_step(9, "路径遍历超限")
            final_reason = (
                f"{undecided}；未穷举的可行空间中仍可能存在可行解，"
                f"无法排除路径空间存在 TP，返回 UNKNOWN"
            )
            verdict_steps.append(
                stage_step(9, f"路径遍历超限 → UNKNOWN"
                              f"（{undecided}，无法排除未探索空间的可行解）")
            )

    # ── Print results ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Algorithm:  conflict-driven path selection + POC verification")
    print(f"  Duration:   {ps_stats.get('elapsed_sec', 0):.1f}s")
    print(f"  Segments:   {ps_stats.get('num_segments', 0)}")
    if verdict == "TP":
        print(f"  Verdict:    ✅ TP (feasible path found)")
    elif verdict == "FP":
        print(f"  Verdict:    ❌ FP (no feasible path)")
    else:
        print(f"  Verdict:    ❓ UNKNOWN (path space too large / undecided)")

    if fp_observations:
        print("\n  🧭 可行路径空间探索 (探索式 FP 判定):")
        print(f"    已探索不同可行路径: {len(fp_observations)} 条")
        for obs in fp_observations:
            print(f"      - {obs.get('signature', '')[:80]}")
            r = obs.get('final_reason', '')
            if r:
                print(f"         理由: {r[:120]}")

    # ── Feasible path-space collection + display ──
    # Deterministic, no LLM: re-enumerates the per-segment feasible candidates
    # (via compute_topk with an unbounded k) and serializes them, so future
    # efficient path-space-search algorithms can consume the structured data.
    chosen_by_seg = _chosen_by_seg(selections)
    path_space = collect_path_space(analyzer, segments, selections)
    path_space["attempts_used"] = len(blocked_paths) + 1 if verdict else 0
    path_space["max_path_attempts"] = loop_attempts
    path_space["configured_max_path_attempts"] = max_path_attempts
    path_space["full_space_explore"] = full_space_explore
    path_space["chosen_by_seg"] = {
        str(k): sorted(v) for k, v in chosen_by_seg.items()
    }
    if getattr(args, "path_space", False):
        print(f"\n{'='*70}")
        print("FEASIBLE PATH SPACE (per-segment candidates + whole-path bound)")
        print(f"{'='*70}")
        print(render_path_space_ascii(
            segments, chosen_by_seg, path_space,
            attempts=path_space["attempts_used"], max_attempts=loop_attempts,
        ))
    else:
        # Compact always-on summary in the RESULTS block.
        seg_summary = ", ".join(
            f"S{info.get('segment_index')}:{info.get('num_feasible_candidates')}"
            for info in path_space.get("per_segment", [])
        )
        print(f"\n  Path space:")
        print(f"    per-segment candidates: {seg_summary}")
        print(f"    total enumerated      : {path_space.get('total_feasible_candidates')}")
        print(f"    whole-path bound ∏N_i : {path_space.get('whole_path_bound'):_}"
              + (f" → {reduced_ps.get('whole_path_bound'):_}"
                 f" (上下文不敏感折叠 {len(reduced_ps.get('ci_groups', []))} 组)"
                 if reduced_ps.get('ci_groups') else ""))
        print(f"    attempts used         : {path_space.get('attempts_used')}/{loop_attempts}"
              + ("  [全空间探索]" if full_space_explore else ""))

    # ── Slice-driven branch-relevance summary (compression) ──
    rel_stats = slice_relevance["stats"]
    print(f"\n  Slice relevance (root-cause branches):")
    print(f"    branch nodes: {rel_stats['num_branch_nodes']} "
          f"| relevant: {rel_stats['num_relevant']} "
          f"| irrelevant: {rel_stats['num_irrelevant']} "
          f"(irrelevant ratio {rel_stats['irrelevant_ratio']})"
          + (" | ⚠️ slice truncated → 保守不压缩" if rel_stats["truncated"] else ""))
    print(f"    path space: {slice_reduce_stats['original_bound']:_} "
          f"→ {slice_reduce_stats['reduced_bound']:_}"
          f"  (compression {slice_reduce_stats['compression_ratio']})")

    # ── Save results ──

    output = {
        "report": str(report_path.name),
        "algorithm": "conflict-driven",
        "entry_function": entry_fn,
        "bug_type": parsed_report.metadata.bug_type,
        "assumption_check": {
            "num_hard_constraints": n_hc,
            "hard_constraints": hard_constraints,
            "num_state_bindings": n_state,
            "state_bindings": cumulative_state,
            "elapsed_sec": round(assumption_elapsed, 1),
        },
        "branch_filter": {
            "num_segments": len(branch_results),
            "total_branches": total_b,
            "consistent": total_con,
            "contradictory_pruned": total_ctr,
            "insufficient_info": total_ins,
            "per_segment": branch_results,
        },
        "path_selection": {
            "verdict": verdict,
            "elapsed_sec": ps_stats.get("elapsed_sec", 0),
            "num_selections": len(selections) if selections else 0,
            "selections": selections if selections else None,
            "blocked_paths": blocked_paths,
            "max_path_attempts": loop_attempts,
            "configured_max_path_attempts": max_path_attempts,
            "full_space_explore": full_space_explore,
            "generalized_fp": bool(ps_stats.get("generalized_fp", False)),
            "structural_generalized_fp": bool(ps_stats.get("structural_generalized_fp", False)),
            "leak_consensus_fp": bool(ps_stats.get("leak_consensus_fp", False)),
            "leak_layer": {
                "fp_verdicts": len(leak_fp_reasons),
                "tp_verdicts": len(leak_tp_reasons),
                "reasons": (leak_fp_reasons or leak_tp_reasons)[:3],
            },
            "fp_exploration": {
                "num_distinct_paths_fp": len(fp_observations),
                "saw_unresolved": saw_unresolved,
                "observations": fp_observations,
            },
        },
        "slice_relevance": {
            "branch_stats": slice_relevance["stats"],
            "reduction": slice_reduce_stats,
        },
        "context_insensitive": ci_stats,
        "total_elapsed_sec": round(time.time() - total_start, 1),
    }
    if poc_result is not None:
        output["poc_verification"] = poc_result
        sim = poc_result.get("simulation") or {}
        output["verification_status"] = sim.get("verified_status", "NONE")
    else:
        output["verification_status"] = "NONE"
    # Record whether the verification feedback loop flipped an initial TP to FP
    # (only a conclusive FP judgment flips; unresolved stays TP/UNVERIFIED).
    output["verification_flipped_to_fp"] = (
        (poc_result or {}).get("simulation", {}).get("conclusion") == "fp"
    )
    output["project"] = source_root.name
    output["verdict"] = verdict
    output["verdict_step"] = final_step
    output["verdict_reason"] = final_reason
    output["verdict_steps"] = verdict_steps
    output["path_space"] = path_space
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull results saved to: {out_path}")
    print(f"Total pipeline time: {time.time() - total_start:.1f}s")

    # ── Final consolidated verdict: decisive step + evidence (TP test case &
    #    execution-passed result; FP cause; UNKNOWN reason) ──
    print("\n" + "=" * 70)
    print("最终判定 VERDICT — 判定步骤与依据")
    print("=" * 70)
    if verdict == "TP":
        print(f"  Verdict:  ✅ TP (真阳性，路径可达且可复现)")
    elif verdict == "FP":
        print(f"  Verdict:  ❌ FP (假阳性，路径不可达/不成立)")
    else:
        print(f"  Verdict:  ❓ UNKNOWN (路径空间过大 / 经多轮探索仍无法确定)")
    print(f"  判定步骤: {final_step or '（未记录）'}")
    if verdict == "TP":
        if poc_code:
            print(f"  测试用例: {poc_path if poc_path else '（未保存到文件）'}  "
                  f"({len(poc_code.splitlines())} 行)")
            print("      -----------------------------------------------------")
            for ln in poc_code.splitlines():
                print(f"      {ln}")
            print("      -----------------------------------------------------")
            print("  反馈执行通过结果（模拟执行判定触发 bug 的轮次）：")
            rounds = (poc_result or {}).get("simulation", {}).get("rounds", [])
            if rounds:
                for r in rounds:
                    tag = "✅触发" if r.get("triggered") else "❌未触发"
                    print(f"      Round {r['round']} [{tag}] "
                          f"fp_likelihood={r.get('fp_likelihood')}")
                    for tr in (r.get("trace") or [])[:6]:
                        print(f"          {tr}")
                    print(f"          依据: {(r.get('reason') or '')[:220]}")
            else:
                print("      （模拟执行未产生轮次记录）")
        else:
            print("  测试用例: （未生成 —— 未获得 POC）")
    else:
        print(f"  原因: {final_reason or '（未提供具体原因）'}")
    if verdict_steps:
        print("  判定过程:")
        for s in verdict_steps:
            print(f"      · {s}")

    return output


def _collect_reports(args):
    """Return the ordered list of report HTML files to process."""
    if args.scan_dir:
        scan = Path(args.scan_dir).resolve()
        if not scan.is_dir():
            raise FileNotFoundError(f"Scan dir not found: {scan}")
        return sorted(p for p in scan.rglob("*.html") if p.name.startswith("report-"))
    if args.project and not args.report:
        proj = Path(args.project)
        if proj.is_dir():
            reports_dir = proj / "reports"
        else:
            reports_dir = Path(args.projects_dir) / args.project / "reports"
        if not reports_dir.is_dir():
            raise FileNotFoundError(f"No reports dir found: {reports_dir}")
        return sorted(p for p in reports_dir.rglob("*.html") if p.name.startswith("report-"))
    if args.report:
        return [Path(args.report).resolve()]
    raise SystemExit("No reports to process: pass --report, or --project / --scan-dir")


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
    args = parser.parse_args()

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
        print("No report HTML files found to process.", file=sys.stderr)
        sys.exit(1)

    # Fail fast: resolve every report's source root / PDG *before* the first LLM
    # call.  Otherwise a misinvocation is either silently analysed against the
    # wrong project (the removed fallback) or reported as a per-report ERROR row
    # only after tens of minutes of unrelated work.
    for rp in reports:
        try:
            _, _, pdg_path, source_root = _resolve_paths(args, rp)
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ {rp.name}: {e}", file=sys.stderr)
            sys.exit(2)
        logger.info("Resolved %s → source_root=%s | pdg=%s",
                    rp.name, source_root, pdg_path.name)

    print(f"Processing {len(reports)} report(s)...\n")

    aggregate = []
    t_all = time.time()
    for i, rp in enumerate(reports, 1):
        print(f"\n{'#' * 70}\n# [{i}/{len(reports)}] {rp.name}\n{'#' * 70}")
        t_report = time.time()
        try:
            out = process_report(args, rp)
            aggregate.append({
                "report": rp.name,
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
                "report": rp.name,
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
               else HERE / "output" / "batch_summary.json")
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps({
        "total": len(aggregate), "tp": n_tp, "fp": n_fp, "error": n_err,
        "results": aggregate,
    }, indent=2, default=str))
    print(f"\nBatch summary saved to: {agg_out}")


if __name__ == "__main__":
    main()
