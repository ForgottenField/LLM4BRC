"""LLM-based variable relevance filtering for backward slices.

After static backward slicing, the slice may include nodes that were pulled
in through variables that are irrelevant to the specific bug type (e.g.,
assignment targets that don't affect the bug trigger).  This module uses
a fast LLM call to classify variables by relevance, enabling pruning of
extraneous slice nodes.

Integration
-----------
Call :func:`filter_slice_variables` between slicing and code reduction::

    from llm_client.fp_analysis.var_relevance import filter_slice_variables

    filtered = filter_slice_variables(
        slice_result, bug_type="Uninitialized argument value",
        trigger_line=172, trigger_expr="findById(lopt)",
        provider=llm_provider,
    )
    reduced = reduce_code(filtered, ...)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from llm_client.base import LLMProvider, LLMRequest
from llm_client.fp_analysis.pdg_models import SliceResult, SliceNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variable collection
# ---------------------------------------------------------------------------


def _collect_all_vars(slice_result: SliceResult) -> set[str]:
    """Collect all unique variable names referenced by slice nodes."""
    vars_set: set[str] = set()
    for sn in slice_result.nodes:
        if sn.node.variable:
            vars_set.add(sn.node.variable)
        for param in sn.node.actual_params:
            if param and param != "?":
                vars_set.add(param)
    vars_set.update(slice_result.input_variables)
    return vars_set


# ---------------------------------------------------------------------------
# Variable role descriptions for the prompt
# ---------------------------------------------------------------------------


def _trunc(s: str | None, max_len: int = 60) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _build_var_descriptions(slice_result: SliceResult, all_vars: set[str]) -> str:
    """Build a human-readable description of each variable's occurrences."""
    # Collect occurrences grouped by variable
    var_occurrences: dict[str, list[str]] = {v: [] for v in all_vars}
    for sn in slice_result.nodes:
        if sn.node.variable and sn.node.variable in var_occurrences:
            var_occurrences[sn.node.variable].append(
                f"def at L{sn.node.source_line}: {_trunc(sn.node.expression)}"
            )
        if sn.node.is_call:
            for i, param in enumerate(sn.node.actual_params):
                if param in var_occurrences:
                    var_occurrences[param].append(
                        f"arg[{i}] at L{sn.node.source_line} -> {_trunc(sn.node.expression)}"
                    )
        # Also detect variable references in the expression text
        if sn.node.expression:
            for v in all_vars:
                if re.search(
                    r"\b" + re.escape(v) + r"\b", sn.node.expression
                ):
                    pass  # too noisy; skip expression text matching
    lines: list[str] = []
    for var in sorted(all_vars):
        details = var_occurrences.get(var, [])
        if details:
            lines.append(f"  - {var}: {'; '.join(details[:4])}")
        else:
            lines.append(f"  - {var}: (input variable)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_VARIABLE_RELEVANCE_PROMPT = """\
You are analyzing a static-analysis bug report. Your task is to determine which
variables are genuinely relevant to understanding the root cause of the bug.

Bug type: {bug_type}
Trigger location: {trigger_function}:L{trigger_line}
Trigger expression: {trigger_expr}

The following variables appeared in the backward slice (def-use chain) from the
trigger point.  For EACH variable, decide whether it is relevant.

{var_descriptions}

Classification rules for "{bug_type}":
- RELEVANT: the variable whose value is directly implicated in the bug (e.g. the
  uninitialized variable), variables whose values flow into it, and variables
  that control whether the buggy code path is reached.
- IRRELEVANT: variables that are output targets of the trigger statement
  (e.g. the LHS of an assignment where the bug is in an argument), temporary
  variables that don't affect the bug's core data flow, or variables whose
  definition/use is unrelated to the bug type.

Respond with EXACTLY one line per variable in exactly this format
(no extra commentary, no markdown):

<variable_name>: RELEVANT
<variable_name>: IRRELEVANT
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_relevance_response(response: str) -> dict[str, bool]:
    """Parse ``{var: RELEVANT|IRRELEVANT}`` lines into {var: is_relevant}."""
    result: dict[str, bool] = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        var, label = line.split(":", 1)
        var = var.strip()
        label = label.strip().upper()
        # Strip markdown bold/italic markers
        var = var.strip("*_`")
        label = label.strip("*_`")
        if label in ("RELEVANT", "IRRELEVANT"):
            result[var] = label == "RELEVANT"
    return result


# ---------------------------------------------------------------------------
# Node filtering
# ---------------------------------------------------------------------------


def _filter_nodes(
    slice_result: SliceResult,
    relevant_vars: set[str],
) -> list[SliceNode]:
    """Remove slice nodes whose *only* variable references are irrelevant.

    A node is kept if:
    - It is the trigger node (always).
    - It defines a relevant variable.
    - It uses a relevant variable as a call parameter.
    - It has no variable references at all (structural / control-flow node).
    """
    trigger_node_id = slice_result.trigger_node_id
    filtered: list[SliceNode] = []
    dropped_ids: list[str] = []

    for sn in slice_result.nodes:
        # ---- always keep trigger ----
        if sn.node.id == trigger_node_id:
            filtered.append(sn)
            continue

        # ---- collect variables this node touches ----
        node_vars: set[str] = set()
        if sn.node.variable:
            node_vars.add(sn.node.variable)
        for param in sn.node.actual_params:
            if param and param != "?":
                node_vars.add(param)

        # ---- keep if structural (no variables) or touches a relevant variable ----
        if not node_vars or (node_vars & relevant_vars):
            filtered.append(sn)
        else:
            dropped_ids.append(f"{sn.node.id} ({','.join(node_vars)})")

    if dropped_ids:
        logger.info(
            "Filtered %d node(s) via LLM variable relevance: %s",
            len(dropped_ids),
            ", ".join(dropped_ids),
        )

    return filtered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_slice_variables(
    slice_result: SliceResult,
    bug_type: str,
    trigger_line: int,
    trigger_expr: str,
    provider: LLMProvider,
    model: str = "",
) -> SliceResult:
    """Use LLM to prune slice nodes whose variables are irrelevant to the bug.

    Args:
        slice_result: The backward slice to filter.
        bug_type: Clang Static Analyzer bug type (e.g. ``"Uninitialized argument value"``).
        trigger_line: 1-based source line of the bug trigger.
        trigger_expr: Expression text at the trigger point.
        provider: An initialised :class:`LLMProvider`.
        model: Optional model override (defaults to the provider's default).

    Returns:
        A new :class:`SliceResult` with irrelevant nodes removed.  If the LLM
        response cannot be parsed, returns the original slice unchanged.
    """
    # ---- Step 1: collect variables ----
    all_vars = _collect_all_vars(slice_result)
    if not all_vars:
        return slice_result

    # ---- Step 2: build prompt ----
    var_descs = _build_var_descriptions(slice_result, all_vars)
    prompt = _VARIABLE_RELEVANCE_PROMPT.format(
        bug_type=bug_type,
        trigger_function=slice_result.trigger_function,
        trigger_line=trigger_line,
        trigger_expr=_trunc(trigger_expr, 80),
        var_descriptions=var_descs,
    )

    # ---- Step 3: call LLM ----
    messages = [
        {
            "role": "system",
            "content": "You are a precise code-analysis assistant.  Output only the requested format.",
        },
        {"role": "user", "content": prompt},
    ]
    request = LLMRequest(
        messages=messages,
        model=model or provider.config.default_model,
        temperature=0.0,
        max_tokens=1024,
    )

    logger.info("Requesting variable relevance classification from LLM ...")
    response = provider.send(request)
    logger.debug("LLM response (first 600 chars):\n%s", response.content[:600])

    # ---- Step 4: parse ----
    relevance = _parse_relevance_response(response.content)
    if not relevance:
        logger.warning(
            "Could not parse any variable relevance from LLM response; "
            "keeping all nodes"
        )
        return slice_result

    relevant_vars = {v for v, is_rel in relevance.items() if is_rel}
    logger.info(
        "Variable relevance: %d relevant, %d irrelevant (of %d total)",
        len(relevant_vars),
        len(relevance) - len(relevant_vars),
        len(relevance),
    )

    # ---- Step 5: filter nodes ----
    filtered_nodes = _filter_nodes(slice_result, relevant_vars)
    removed = len(slice_result.nodes) - len(filtered_nodes)
    if removed:
        logger.info("Removed %d/%d slice node(s) via LLM variable filtering", removed, len(slice_result.nodes))

    return SliceResult(
        trigger_function=slice_result.trigger_function,
        trigger_node_id=slice_result.trigger_node_id,
        nodes=filtered_nodes,
        input_variables=slice_result.input_variables,
        involved_functions=slice_result.involved_functions,
        involves_globals=slice_result.involves_globals,
        truncated=slice_result.truncated,
    )
