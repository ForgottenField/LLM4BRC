"""Input format detection for bug reports."""

from __future__ import annotations

import json

from llm_client.bug_analysis.schemas import InputFormat


def detect_format(content: str) -> InputFormat:
    """Detect the format of *content* by inspecting its structure.

    Detection logic (in order):

    1. **SARIF** — Content is valid JSON containing ``"$schema"`` with
       ``"sarif"`` or a top-level ``"version"`` of ``"2.1.0"``.
    2. **JSON** — Content is valid JSON (but does *not* match SARIF).
    3. **Plain text** — Everything else.

    Returns
    -------
    InputFormat
        The detected format.
    """
    stripped = content.strip()
    if not stripped:
        return InputFormat.PLAIN_TEXT

    # Attempt JSON parsing first (covers both SARIF and generic JSON).
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return InputFormat.PLAIN_TEXT

    if _is_sarif(data):
        return InputFormat.SARIF

    return InputFormat.JSON


def _is_sarif(data: object) -> bool:
    """Heuristic: does *data* look like a SARIF 2.1 document?"""
    if not isinstance(data, dict):
        return False

    # SARIF 2.1 uses "$schema" pointing to a sarif URI, or "version": "2.1.0".
    schema = data.get("$schema", "")
    if isinstance(schema, str) and "sarif" in schema.lower():
        return True

    version = data.get("version", "")
    if version == "2.1.0":
        return True

    return False
