"""Provenance checks for trigger values explicitly present in a CSA path."""

from __future__ import annotations

import re

from fp_analysis.models import code_model_of

_NULL_INIT_RE = re.compile(r"\b(?:nullptr|NULL)\b")
_NULL_BUG_RE = re.compile(r"\b(?:null|nullptr|nonnull|dereference)\b", re.I)


def _event_source(parsed, event: dict, source_root: str | None) -> str:
    line = int(event.get("line_number") or 0)
    snippets = getattr(parsed, "source_snippets", {}) or {}
    file_path = str(event.get("file_path") or "")
    bug_file = str(
        getattr(getattr(parsed, "metadata", None), "bug_file", "") or ""
    )
    if line in snippets and (not file_path or file_path == bug_file):
        return str(snippets[line])
    if not file_path or line <= 0:
        return ""
    try:
        model = code_model_of(parsed, source_root)
        return str(model.snippet(str(file_path), line, radius=0).get(line) or "")
    except (AttributeError, OSError, TypeError, ValueError):
        return ""


def report_path_literal_trigger(
    parsed,
    source_root: str | None = None,
) -> dict | None:
    """Return explicit null-literal evidence from the report's own path.

    A value that the report initializes in its caller frame is not invented by
    the generated POC merely because the POC writes the same literal.  This
    distinguishes a report-path argument from a POC-only argument without
    guessing whether the reported test is a desirable public API call.
    """
    metadata = getattr(parsed, "metadata", None)
    bug_type = str(getattr(metadata, "bug_type", "") or "")
    if not _NULL_BUG_RE.search(bug_type):
        return None
    events = list(getattr(parsed, "path_events", None) or [])
    if not events:
        return None

    initialization: list[dict] = []
    passing_null: list[int] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        number = int(event.get("path_number") or 0)
        description = str(event.get("description") or "")
        source = _event_source(parsed, event, source_root)
        description_lower = description.lower()
        if (
            "initialized to a null pointer" in description_lower
            or _NULL_INIT_RE.search(source)
        ):
            initialization.append({
                "event_number": number,
                "line": int(event.get("line_number") or 0),
                "description": description,
                "source": source,
            })
        if "passing null pointer" in description_lower:
            passing_null.append(number)

    if not initialization:
        return None
    return {
        "kind": "report_path_literal",
        "value": "null",
        "initialization_events": initialization,
        "passing_events": passing_null,
        "reason": (
            "The CSA path explicitly initializes the trigger value to null in "
            "the report's own caller path; the POC is reproducing report "
            "provenance, not inventing the value."
        ),
    }
