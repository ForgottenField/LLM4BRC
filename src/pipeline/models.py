"""The dataclass contracts the stage functions hand each other.

``process_report`` used to be one 1123-line function body in which every stage
read and wrote the enclosing scope's locals.  The split into ``_stageN_*``
functions needs those values to travel explicitly, and *how* they travel is the
part that decides whether the split is safe:

``@dataclass(frozen=True)``
    For a stage's **outputs** — one immutable record per stage, named after the
    stage.  A frozen record cannot be quietly mutated by a later stage, which is
    what makes "stage 5 ran with the state stage 4 produced" checkable by
    reading a signature instead of tracing a local.

``@dataclass``
    For the two **mutable state carriers** the attempt loop updates in place:
    :class:`VerdictState` (the verdict and the story of how it was reached) and
    :class:`ExplorationState` (everything the exploration accumulates across
    attempts).  They are passed by reference on purpose — the loop rebinds
    ``verdict``, ``selections`` and ``ps_stats`` on various branches and its
    ``break``s land at different points, so threading them back through return
    values would have meant re-deriving which of twenty values were live at each
    exit.  They are the *same* names the old enclosing scope held.

:class:`SelectionContext`
    The frozen bundle of everything the attempt loop only ever **reads**.  It
    exists to keep ``_run_path_attempts`` a three-argument function instead of a
    twenty-argument one, and — more usefully — to make the read-only half of the
    loop's inputs explicit, so a future edit that needs to write one has to say
    so by adding it to a state carrier rather than by mutating a tuple.

Nothing here computes: the records are plain value objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.outcomes import Verdict
from pipeline.protocols import PathAnalyzerPort


# ─── stage outputs (immutable) ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Stage1Parse:
    """Stage 1's parse half: the canonical report rendered into the legacy dicts.

    ``loaded`` is the frontend's :class:`LoadedReport` — kept because the banner
    prints the adapter and analyzer it detected, and ``bug_path`` is the
    ``BugPath`` the slicer and the gates consume.
    """

    parsed_report: object
    bug_path: object
    loaded: object
    entry_fn: str
    lite_graph: object = None


@dataclass(frozen=True)
class AssembledProject:
    """Stage 1's assembly half: the CFG cache view and the SDG.

    ``cfg_cache`` is ``cache_set.as_dict()`` — the *live merged* mapping, not a
    copy.  Stage 2 adds path files to the same :class:`CFGCacheSet` and stage 5
    must see them, so any ``dict(cfg_cache)`` here would silently drop the
    on-demand caches the whole multi-TU design exists for.
    """

    cache_set: object
    cfg_cache: dict
    sdg: object
    graph_backend: str = "exact"


@dataclass(frozen=True)
class Stage3Slice:
    """The backward slice, plus the dense CSA branch decisions for stage 5."""

    file_index: dict
    slice_mask: object
    slice_result: object
    path_annotations: dict


@dataclass(frozen=True)
class Stage4Assumptions:
    """The assumption-derived state (the only LLM call before the branch filter).

    The two counts travel with the values because the result JSON reports them
    as numbers and the console prints the values themselves.
    """

    hard_constraints: list
    cumulative_state: dict
    elapsed_sec: float
    num_hard_constraints: int
    num_state_bindings: int


@dataclass(frozen=True)
class Stage5Branches:
    """The branch filter's per-segment results and its four totals.

    The four totals are computed here (lines 262-265) but read in the result
    JSON five stages later, so they travel with the results rather than being
    recomputed at the other end — a second summation is a second definition of
    "contradictory pruned".
    """

    results: list
    insufficient_counts: dict
    total_branches: int
    total_consistent: int
    total_contradictory: int
    total_insufficient: int


@dataclass(frozen=True)
class Stage8PathSpace:
    """Everything stage 8 decides, including the exploration budget.

    ``combo_iter`` is a live iterator over the whole-path combinations and is
    therefore consumed exactly once, by the attempt loop — it is state in a
    frozen record, which is the one place in this module that is true and is why
    it is called out here.
    """

    report_stem: str
    out_path: Path
    max_path_attempts: int
    path_space_enum: dict
    slice_truncated: bool
    slice_relevance: dict
    reduced_ps: dict
    slice_reduce_stats: dict
    ci_stats: dict
    reduced_bound_one: bool
    reduced_bound: int
    loop_attempts: int
    full_space_explore: bool
    combo_iter: object


# ─── mutable state carriers ──────────────────────────────────────────────────────


@dataclass
class VerdictState:
    """The report-level verdict and the chronological story of reaching it.

    ``final_step``/``final_reason`` start as ``""`` — not ``None`` — because that
    is what the old enclosing scope initialized them to, and stage 8 *re*-sets
    them to ``None`` at the stage-9 boundary.  Both steps are load-bearing: the
    result JSON's ``verdict_step`` is ``final_step`` verbatim, and the console
    tests ``_NO_PATH_STEP_KEYWORD in (final_step or "")``, which needs the empty
    string to behave like "no decisive step" and not like a match.
    """

    steps: list = field(default_factory=list)
    final_step: object = ""
    final_reason: object = ""
    verdict: Verdict | None = None


@dataclass
class ExplorationState:
    """What the attempt loop accumulates across the distinct paths it explores.

    Every field is read by the post-loop aggregation, by the result JSON, or by
    both — ``blocked_paths`` is the loop's own diversity mechanism (a blocked
    whole-path combination is never re-selected), and the two leak lists exist
    only so the aggregation can require the leak layer to *agree* rather than
    accept whichever attempt concluded first.
    """

    blocked_paths: list = field(default_factory=list)
    fp_observations: list = field(default_factory=list)
    leak_fp_reasons: list = field(default_factory=list)
    leak_tp_reasons: list = field(default_factory=list)
    saw_unresolved: bool = False
    selections: object = None
    ps_stats: dict = field(default_factory=dict)
    poc_result: object = None
    poc_code: object = None
    poc_path: object = None
    combo: object = None
    combo_sig: object = None


@dataclass(frozen=True)
class SelectionContext:
    """Everything the attempt loop reads and never writes.

    Frozen so that the read-only half of the loop's inputs is stated once.  A
    value the loop must update belongs on :class:`VerdictState` or
    :class:`ExplorationState`; adding it here instead is the mistake this
    dataclass is shaped to make visible.
    """

    args: object
    report_ref: object
    analyzer: PathAnalyzerPort
    parsed_report: object
    loaded: object
    segments: list
    slice_mask: object
    sdg: object
    source_root: Path
    cfg_cache: dict
    extra_preconditions: list
    hard_constraints: list
    reduced_ps: dict
    reduced_bound_one: bool
    slice_truncated: bool
    slice_relevance: dict
    combo_iter: object
    loop_attempts: int
    max_path_attempts: int
    full_space_explore: bool
    slice_reduce_stats: dict
    ci_stats: dict
    out_path: Path
    report_stem: str
    entry_fn: str
    lite_graph: object
    total_start: float
