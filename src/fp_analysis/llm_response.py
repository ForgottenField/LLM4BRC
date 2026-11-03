"""Parsing and repair helpers for LLM text responses."""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def _close_open_structures(text: str) -> str:
    """Append braces needed to balance truncated JSON."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
    return text + "".join("}" if char == "{" else "]" for char in reversed(stack))


def repair_truncated_json(text: str) -> str | None:
    """Return the shortest reparsed JSON prefix, or ``None``."""
    if not text:
        return None
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError as exc:
        error_position = exc.pos

    cuts: list[int] = []
    seen: set[int] = set()

    def add_cut(index: int) -> None:
        if 0 < index < len(text) and index not in seen:
            seen.add(index)
            cuts.append(index)

    add_cut(error_position)
    for index in range(min(error_position, len(text)) - 1, -1, -1):
        if text[index] in ",{[]}" and (index == 0 or text[index - 1] != "\\"):
            add_cut(index)
        if len(cuts) > 250:
            break
    cuts.sort(reverse=True)

    for cut in cuts:
        prefix = text[:cut].rstrip().rstrip(",").rstrip()
        if prefix.endswith(":"):
            match = re.search(r'"([^"\\]|\\.)*"\s*$', prefix.rstrip()[:-1].rstrip())
            if match:
                prefix = prefix[:match.start()].rstrip()
        closed = _close_open_structures(prefix)
        try:
            json.loads(closed)
            return closed
        except json.JSONDecodeError:
            continue
    return None


def _json_text(raw: str) -> str:
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.match(r"```(?:json)?\s*\n?(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_json_response(raw: str, context: str) -> dict:
    """Parse a JSON object from a response, including fenced markdown."""
    text = _json_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse LLM %s output: %s\nRaw: %s",
            context, exc, raw[:500],
        )
        raise RuntimeError(
            f"LLM returned invalid JSON for {context}: {exc}"
        ) from exc


def parse_json_response_tolerant(raw: str, context: str) -> dict:
    """Parse a response, repairing a truncated JSON prefix if necessary."""
    text = _json_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = repair_truncated_json(text)
        if repaired is not None:
            try:
                logger.warning(
                    "LLM %s output was truncated at token limit; salvaged "
                    "a repaired JSON prefix (%d chars).", context, len(repaired),
                )
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        logger.error(
            "Failed to parse LLM %s output: %s\nRaw: %s",
            context, exc, raw[:500],
        )
        raise RuntimeError(
            f"LLM returned invalid JSON for {context}: {exc}"
        ) from exc


def extract_code(raw: str) -> str:
    """Extract C++ code from a response, stripping markdown fences."""
    text = raw.strip()
    match = re.search(r"```(?:cpp|c\+\+)?\s*\n?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text
