"""Unified facts and source effects for global report constraint solving."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fp_analysis.cfg_feasibility import extract_assumptions_from_events
from fp_analysis.models import code_model_of

_SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx"}
_PROJECT_LINES: dict[str, dict[Path, list[str]]] = {}
_STRING_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_ASSIGN_RE = re.compile(
    r"\b(?P<subject>[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*)*)"
    r"\s*=\s*(?P<value>true|false|null|nullptr|-?\d+|\"(?:[^\"\\]|\\.)*\")",
    re.I,
)
_INIT_RE = re.compile(
    r"\b(?P<subject>[A-Za-z_]\w*)\s*\(\s*"
    r"(?P<value>true|false|-?\d+)\s*\)"
)
_RETURN_VALUE_RE = re.compile(r"Returning the value\s+(-?\d+|true|false|null)", re.I)
_CALL_RE = re.compile(r"(?:Calling|Entered call from)\s+'([^']+)'")


@dataclass(frozen=True)
class ConstraintEvidence:
    """A source-grounded constraint input."""

    evidence_id: str
    kind: str
    source_file: str
    source_line: int
    text: str
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstraintFact:
    """One report or source-derived value fact."""

    fact_id: str
    subject: str
    operator: str
    value: str
    event_numbers: tuple[int, ...]
    source_file: str
    source_line: int
    provenance: str
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_numbers"] = list(self.event_numbers)
        return data


@dataclass(frozen=True)
class ConstraintEffect:
    """A source assignment/initialization affecting a reported subject."""

    effect_id: str
    subject: str
    operator: str
    value: str
    condition: str
    source_file: str
    source_line: int
    source_text: str
    source_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalConstraintBundle:
    """Complete deterministic context passed to the global constraint solver."""

    facts: list[ConstraintFact] = field(default_factory=list)
    effects: list[ConstraintEffect] = field(default_factory=list)
    evidence: list[ConstraintEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [item.to_dict() for item in self.facts],
            "effects": [item.to_dict() for item in self.effects],
            "evidence": [item.to_dict() for item in self.evidence],
        }


def build_global_constraint_bundle(
    parsed_report,
    source_root: str | Path | None = None,
) -> GlobalConstraintBundle:
    """Extract report facts, call inputs, and source effects."""
    model = code_model_of(parsed_report, str(source_root) if source_root else None)
    events = list(getattr(parsed_report, "path_events", None) or [])
    bundle = GlobalConstraintBundle()
    facts: list[ConstraintFact] = []
    evidence: list[ConstraintEvidence] = []
    last_call = ""
    bug_file = getattr(getattr(parsed_report, "metadata", None), "bug_file", "")
    event_sources = {
        int(event.get("path_number") or 0): _event_source(
            model, event, bug_file,
        )
        for event in events
    }

    assumptions = extract_assumptions_from_events(events, (0, 10**9))
    for index, assumption in enumerate(assumptions, 1):
        source_file = event_sources.get(
            assumption.event_number, ("", "", ""),
        )[0]
        facts.append(ConstraintFact(
            fact_id=f"F{index}",
            subject=assumption.variable,
            operator=assumption.operator,
            value=assumption.value,
            event_numbers=(assumption.event_number,),
            source_file=source_file,
            source_line=assumption.line_number,
            provenance="report_explicit",
            source_text=assumption.raw_description,
        ))

    next_fact = len(facts) + 1
    for event in events:
        number = int(event.get("path_number") or 0)
        line = int(event.get("line_number") or 0)
        desc = str(event.get("description") or "")
        called = _CALL_RE.search(desc)
        if called:
            last_call = called.group(1)
        source_file, source_text, context = event_sources.get(
            number, ("", "", ""),
        )
        evidence.append(ConstraintEvidence(
            evidence_id=f"E{len(evidence) + 1}",
            kind="report_path",
            source_file=source_file,
            source_line=line,
            text=f"{desc} | {source_text}".strip(" |"),
            context=context,
        ))

        if called:
            for literal_index, literal in enumerate(
                _STRING_LITERAL_RE.findall(context or source_text), 1,
            ):
                facts.append(ConstraintFact(
                    fact_id=f"C{next_fact}",
                    subject=(
                        f"call#{number}:{called.group(1)}:"
                        f"arg_literal[{literal_index}]"
                    ),
                    operator="==",
                    value=_decode_cpp_string(literal),
                    event_numbers=(number,),
                    source_file=source_file,
                    source_line=line,
                    provenance="source_literal",
                    source_text=source_text,
                ))
                next_fact += 1

        returned = _RETURN_VALUE_RE.search(desc)
        if returned and last_call:
            facts.append(ConstraintFact(
                fact_id=f"R{next_fact}",
                subject=f"return:{last_call}",
                operator="==",
                value=returned.group(1).lower(),
                event_numbers=(number,),
                source_file=source_file,
                source_line=line,
                provenance="report_explicit",
                source_text=desc,
            ))
            next_fact += 1

    bundle.facts = facts
    bundle.evidence = evidence
    bundle.effects = _extract_source_effects(
        facts, source_root or getattr(parsed_report, "source_root", None),
    )
    return bundle


def solve_immediate_conflicts(
    bundle: GlobalConstraintBundle,
) -> dict[str, Any]:
    """Solve direct equality conflicts without calling an LLM."""
    by_subject: dict[str, list[ConstraintFact]] = {}
    for fact in bundle.facts:
        if fact.operator != "==":
            continue
        by_subject.setdefault(_subject_key(fact.subject), []).append(fact)

    contradictions = []
    for subject, facts in by_subject.items():
        values = [_normalize_value(item.value) for item in facts]
        unique = set(values)
        if len(unique) > 1:
            contradictions.append({
                "subject": subject,
                "values": sorted(unique),
                "basis_ids": [item.fact_id for item in facts],
                "reason": (
                    f"Report/source facts assign incompatible values to {subject}: "
                    + ", ".join(
                        f"{item.fact_id}={item.value}" for item in facts
                    )
                ),
            })
    guarded = _solve_guarded_literal_flags(bundle)
    contradictions.extend(guarded)
    if contradictions:
        return {
            "status": "unsat",
            "contradictions": contradictions,
            "derived_facts": [],
            "missing_information": [],
            "reason": contradictions[0]["reason"],
        }
    return {
        "status": "unknown",
        "contradictions": [],
        "derived_facts": [],
        "missing_information": [],
        "reason": "",
    }


def _solve_guarded_literal_flags(
    bundle: GlobalConstraintBundle,
) -> list[dict[str, Any]]:
    """Evaluate literal-controlled boolean flags with a single path input."""
    literals = [
        fact for fact in bundle.facts
        if fact.provenance == "source_literal"
    ]
    if len(literals) != 1:
        return []
    literal = literals[0]
    input_text = str(literal.value)
    if "{" not in input_text or "}" not in input_text:
        return []
    conflicts: list[dict[str, Any]] = []
    report_flags = [
        fact for fact in bundle.facts
        if fact.provenance == "report_explicit"
        and fact.operator == "=="
        and _normalize_value(fact.value) in ("true", "false")
        and not fact.subject.startswith("condition@")
    ]
    for fact in report_flags:
        subject = _subject_key(fact.subject).split(".")[-1]
        effects = [
            effect for effect in bundle.effects
            if _subject_key(effect.subject).split(".")[-1] == subject
        ]
        initial = next(
            (
                effect for effect in effects
                if effect.condition == "unconditional"
                and effect.value.lower() in ("true", "false")
                and (
                    (match := _INIT_RE.search(effect.source_text)) is not None
                    and _subject_key(match.group("subject")).split(".")[-1]
                    == subject
                )
            ),
            None,
        )
        if initial is None:
            continue
        guarded = next(
            (
                effect for effect in effects
                if re.search(r"==\s*'(.)'", effect.condition)
                and effect.value.lower() in ("true", "false")
            ),
            None,
        )
        if guarded is None:
            continue
        match = re.search(r"==\s*'(.)'", guarded.condition)
        if match is None:
            continue
        guard_char = match.group(1)
        derived = (
            guarded.value.lower()
            if guard_char in input_text
            else initial.value.lower()
        )
        claimed = _normalize_value(fact.value)
        if derived == claimed:
            continue
        conflicts.append({
            "subject": subject,
            "values": sorted((claimed, derived)),
            "basis_ids": [
                fact.fact_id, literal.fact_id,
                initial.effect_id, guarded.effect_id,
            ],
            "reason": (
                f"{subject} starts as {initial.value}; it becomes {guarded.value} "
                f"only when the input contains {guard_char!r}. The reported path "
                f"input {literal.value!r} "
                f"{'contains' if guard_char in input_text else 'does not contain'} "
                f"{guard_char!r}, so the derived value is {derived}, not {claimed}."
            ),
        })
    return conflicts


def validate_solution(
    raw: dict[str, Any],
    bundle: GlobalConstraintBundle,
) -> dict[str, Any]:
    """Validate an LLM solution before it may influence the verdict."""
    known = {
        item.fact_id: item for item in bundle.facts
    } | {
        item.effect_id: item for item in bundle.effects
    }
    status = str(raw.get("status") or "unknown").strip().lower()
    if status not in ("sat", "unsat", "unknown"):
        status = "unknown"
    contradictions = []
    for item in raw.get("contradictions") or []:
        if not isinstance(item, dict):
            continue
        basis = [str(value) for value in (item.get("basis_ids") or [])]
        existing = [known[value] for value in basis if value in known]
        if len(set(basis)) < 2 or len(existing) < 2:
            continue
        if not any(getattr(entry, "provenance", "") == "report_explicit"
                   for entry in existing if isinstance(entry, ConstraintFact)):
            continue
        if not any(
            isinstance(entry, ConstraintEffect)
            or (
                isinstance(entry, ConstraintFact)
                and entry.provenance in ("source_effect", "source_literal")
            )
            for entry in existing
        ):
            continue
        contradictions.append({
            "subject": str(item.get("subject") or ""),
            "values": list(item.get("values") or []),
            "basis_ids": basis,
            "reason": str(item.get("reason") or ""),
        })
    if status == "unsat" and not contradictions:
        status = "unknown"
    return {
        "status": status,
        "contradictions": contradictions,
        "derived_facts": [
            item for item in (raw.get("derived_facts") or [])
            if isinstance(item, dict)
        ],
        "missing_information": [
            str(item) for item in (raw.get("missing_information") or [])
        ],
        "reason": str(raw.get("reason") or ""),
    }


def render_bundle(bundle: GlobalConstraintBundle, max_effects: int = 80) -> str:
    """Render the complete structured bundle for the global solver prompt."""
    parts = [
        "FACTS:",
        *[
            f"- {item.fact_id}: {item.subject} {item.operator} {item.value} "
            f"[{item.provenance}; events={list(item.event_numbers)}; "
            f"{item.source_file}:{item.source_line}]"
            for item in bundle.facts
        ],
        "",
        "SOURCE EFFECTS:",
        *[
            f"- {item.effect_id}: {item.subject} = {item.value} "
            f"when `{item.condition}` "
            f"[{item.source_file}:{item.source_line}]\n"
            f"{item.source_context}"
            for item in bundle.effects[:max_effects]
        ],
        "",
        "SOURCE CONTEXT:",
        *[
            f"- {item.evidence_id} ({item.kind}) {item.source_file}:"
            f"{item.source_line}: {item.text}\n{item.context}"
            for item in bundle.evidence[:max_effects]
        ],
    ]
    return "\n".join(parts)


def _event_source(model, event: dict, bug_file: str):
    raw_file = event.get("file_path") or bug_file
    if not raw_file:
        return "", "", ""
    resolved = model.resolve(str(raw_file))
    local = resolved.local if resolved is not None else None
    if local is None:
        return str(raw_file), "", ""
    lines = model.lines(local)
    line = int(event.get("line_number") or 0)
    if line <= 0 or line > len(lines):
        return str(local), "", ""
    lo = max(1, line - 2)
    hi = min(len(lines), line + 2)
    context = "\n".join(
        f"{number}: {lines[number - 1]}" for number in range(lo, hi + 1)
    )
    return str(local), lines[line - 1], context


def _extract_source_effects(
    facts: list[ConstraintFact],
    source_root: str | Path | None,
) -> list[ConstraintEffect]:
    if not source_root:
        return []
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        return []
    subject_info = {
        _subject_key(fact.subject).split(".")[-1]: fact.value.lower()
        for fact in facts
        if fact.provenance == "report_explicit"
        and not fact.subject.startswith(("condition@", "call#", "return:"))
        and (
            len(_subject_key(fact.subject).split(".")[-1]) > 2
            or fact.value.lower() in ("true", "false")
        )
    }
    subjects = sorted(
        subject_info,
        key=lambda subject: (
            subject_info[subject] not in ("true", "false", "null", "nullptr"),
            subject,
        ),
    )
    if not subjects:
        return []
    files = _project_source_lines(str(root))
    effects: list[ConstraintEffect] = []
    effect_index = 1
    for subject in subjects:
        token = re.compile(rf"\b{re.escape(subject)}\b")
        for path, lines in files.items():
            for index, line in enumerate(lines):
                if not token.search(line):
                    continue
                if line.lstrip().startswith(("//", "*", "/*")):
                    continue
                assignment = _ASSIGN_RE.search(line) or _INIT_RE.search(line)
                if not assignment:
                    continue
                assigned_subject = _subject_key(assignment.group("subject"))
                if assigned_subject.split(".")[-1] != subject:
                    continue
                condition = _guard_condition(lines, index)
                lo = max(0, index - 3)
                hi = min(len(lines), index + 4)
                effects.append(ConstraintEffect(
                    effect_id=f"S{effect_index}",
                    subject=subject,
                    operator="=",
                    value=str(assignment.group("value")),
                    condition=condition,
                    source_file=str(path),
                    source_line=index + 1,
                    source_text=line.strip(),
                    source_context="\n".join(
                        f"{number + 1}: {lines[number]}"
                        for number in range(lo, hi)
                    ),
                ))
                effect_index += 1
                if effect_index > 400:
                    return effects
    return effects


def _guard_condition(lines: list[str], index: int) -> str:
    for offset in range(1, 5):
        current = index - offset
        if current < 0:
            break
        text = lines[current].strip()
        if text.startswith("if") or text.startswith("} else if"):
            return text
        if text and not text.startswith(("//", "*", "/*")):
            break
    return "unconditional"


def _project_source_lines(root_text: str) -> dict[Path, list[str]]:
    cached = _PROJECT_LINES.get(root_text)
    if cached is not None:
        return cached
    result: dict[Path, list[str]] = {}
    root = Path(root_text)
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in _SOURCE_SUFFIXES
            or ".git" in path.parts
        ):
            continue
        try:
            result[path] = path.read_text(
                encoding="utf-8", errors="replace",
            ).splitlines()
        except OSError:
            continue
    _PROJECT_LINES[root_text] = result
    return result


def _subject_key(subject: str) -> str:
    return re.sub(r"\s+", "", str(subject or ""))


def _normalize_value(value: Any) -> str:
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return "true"
    if text in ("false",):
        return "false"
    if text in ("null", "nullptr"):
        return "null"
    return text


def _decode_cpp_string(literal: str) -> str:
    try:
        return ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return literal[1:-1]
