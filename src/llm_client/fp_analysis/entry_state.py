"""Fabricated entry-state check (deterministic, no LLM).

A CSA null-deref report whose path *begins* with ``Assuming pointer value is
null`` is asserting that the flagged function was entered with a null pointer
argument — the analyzer treats the function as its own entry point.  When that
function's own body carries an **always-compiled** check that fails on a null
argument (``ABSL_RAW_CHECK(out != nullptr, ...)``, ``CHECK(...)``,
``FAISS_THROW_IF_NOT(...)``), and the report's own path records taking that
check's *failure* branch and then continues to the deref, the reported path is
not realizable: the failure arm terminates the process (absl
``raw_logging.cc``'s FATAL handler calls ``abort()``; ``FAISS_THROW_IF_NOT``
throws), so control never reaches the deref with a null argument.

This is the deterministic half of the "POC manufactures the bug's precondition"
defect (stage 10): the pipeline used to answer "can *code* make CSA's
precondition true?" instead of "can a real caller reach this?", so a POC that
passes the null itself counted as a trigger.  Here the same question is answered
from the report and the bug file alone — no path enumeration, no POC, no model
call — and the answer is FP.

Why the existing layers miss it
-------------------------------
* The simulated-execution audit sees the path steps and the bug file, but its
  manufactured-entry rules are keyed on *which function the POC calls* (an
  internal/leaf/template helper, a stubbed library function, a subclass that
  drops an override).  When the flagged function **is** a public API — absl's
  ``SimpleAtob`` — calling it directly is exactly what the POC is told to do,
  and none of the rules match.  With no callers and no callee bodies in the
  prompt, the audit has no evidence that the null is unreachable.
* The interior of a ``CHECK``-family macro is not a branch condition the
  CFG/PDG layer can compare (it expands to a statement, not a predicate pair),
  so ``report_consistency``'s same-condition-opposite-direction detector cannot
  see it either.

Guards (every one fails closed — a missed finding only means the pipeline
behaves as it did before):

* the bug type must mention a null dereference, and the entry assumption must be
  the report's first (or second, allowing the ``Error Start`` event) event;
* the dereferenced variable must be a **pointer parameter** of the function in
  the report's ``loaded from variable 'V'`` step — a member, a local, or an
  unresolvable signature yields no finding;
* the check must come from an always-compiled family (see ``_FATAL_GUARDS``);
  ``assert``/``DCHECK``/``FAISS_ASSERT`` are deliberately excluded because an
  NDEBUG build compiles them out, which is exactly why CSA's null survives;
* the check's condition must require that parameter to be **non-null** in a form
  this module can prove (``V != nullptr``, ``NULL != V``, ``V != 0``, bare
  ``V``) — anything else (``V == nullptr``, a compound condition) yields no
  finding rather than a guess;
* the report must itself record taking the *true* branch at the check's line
  (for this macro family the true branch of ``if (!(cond))`` is the failure
  arm), and the deref line must lie after it inside the same function.

The optional call-site scan is **corroboration only** (it counts how many real
call sites the function has in the source tree, and shows them in the reason).
It is never the sole basis of a verdict: "no in-tree caller" is weak evidence
for a public library API whose callers may live outside the corpus, so a
zero-call-site count alone does not conclude FP.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── report-side patterns ────────────────────────────────────────────────────

_ENTRY_NULL_RE = re.compile(r"assuming\s+pointer\s+value\s+is\s+null", re.I)
_NULL_VAR_RE = re.compile(
    r"(?:loaded|read)\s+from\s+(?:the\s+)?(?:variable|memory)\s+"
    r"['\"]([A-Za-z_]\w*)['\"]",
    re.I,
)


# ─── guard families whose FAILURE ARM DOES NOT RETURN ────────────────────────
#
# Each entry: (macro name, the semantics this module relies on).  All are
# *always compiled* — no NDEBUG switch — which is what makes the conclusion
# sound.  `assert` / `DCHECK` / `FAISS_ASSERT` are deliberately absent: under
# NDEBUG they expand to nothing, and a report produced in an NDEBUG build
# legitimately reaches the deref, so concluding FP from them would be unsound.
_FATAL_GUARDS: tuple[tuple[str, str], ...] = (
    ("ABSL_RAW_CHECK",
     "absl/base/internal/raw_logging.h:59 → ABSL_RAW_LOG(FATAL) → "
     "absl/base/internal/raw_logging.cc:183 `abort()`"),
    ("ABSL_INTERNAL_CHECK",
     "absl/base/internal/raw_logging.h:85 → FATAL log + ABSL_INTERNAL_UNREACHABLE"),
    ("ABSL_CHECK", "absl CHECK 族：失败即 FATAL 终止（无条件编译）"),
    ("GOOGLE_CHECK", "glog/absl CHECK 别名：失败即 FATAL 终止"),
    ("CHECK", "glog/absl CHECK 族：失败即 FATAL 终止（无条件编译）"),
    ("FAISS_THROW_IF_NOT_FMT", "FaissAssert.h：失败即抛异常，本函数内其后语句不可达"),
    ("FAISS_THROW_IF_NOT_MSG", "FaissAssert.h：失败即抛异常，本函数内其后语句不可达"),
    ("FAISS_THROW_IF_NOT", "FaissAssert.h：失败即抛异常，本函数内其后语句不可达"),
)

# `(?![_A-Z])` keeps `CHECK` from matching `CHECK_EQ`/`DCHECK` (and ABSL_CHECK
# from matching ABSL_CHECK_EQ): those take two operands, so "requires non-null"
# is not expressible from the first argument alone — skip them, fail closed.
_FATAL_GUARD_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n, _ in _FATAL_GUARDS) + r")(?![_A-Z])\s*\("
)
_GUARD_SEMANTICS: dict[str, str] = dict(_FATAL_GUARDS)


# ─── tiny lexical helpers ────────────────────────────────────────────────────

def _matching_paren(text: str, open_idx: int) -> int | None:
    """Index of the ``)`` matching ``text[open_idx] == '('``.

    String/char literals are skipped.  Returns ``None`` when the call does not
    close on this line (a multi-line macro invocation) — callers treat that as
    "cannot analyze", never as a finding.
    """
    if open_idx >= len(text) or text[open_idx] != "(":
        return None
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    break
                i += 1
        elif ch == "/" and i + 1 < n and text[i + 1] == "/":
            return None
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _first_arg(text: str, open_idx: int) -> str | None:
    """First top-level argument of the call whose ``(`` is at *open_idx*."""
    close = _matching_paren(text, open_idx)
    if close is None:
        return None
    inner = text[open_idx + 1:close]
    depth = 0
    for i, ch in enumerate(inner):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return inner[:i].strip()
    return inner.strip()


_NUM_ZERO_RE = re.compile(r"^0(?:[uUlL]+)?$")


def _requires_non_null(cond: str, var: str) -> bool:
    """Whether *cond* can only be true when *var* is a non-null pointer.

    Accepts ``V != nullptr``, ``nullptr != V``, ``V != NULL``, ``V != 0``,
    ``!!V`` and a bare ``V`` (truthiness of a pointer).  Everything else —
    including the opposite polarity (``V == nullptr``) and compound conditions
    — returns ``False`` so the caller stays silent rather than guessing.
    """
    c = re.sub(r"\s+", "", cond)
    if not c:
        return False
    c = re.sub(r"\b(?:nullptr|NULL|__null)\b", "NULL", c)
    # Peel redundant outer parentheses: `((out != nullptr))`
    while c.startswith("(") and c.endswith(")") and _matching_paren(c, 0) == len(c) - 1:
        c = c[1:-1]
    # `!!V` is truthiness (requires non-null); a single `!V` is the *opposite*
    # test — it holds only while V is null, so it does not refute the report's
    # assumption and must never be read as requiring non-null.
    bangs = len(c) - len(c.lstrip("!"))
    if bangs % 2:
        return False
    c = c[bangs:]
    v = re.escape(var)
    if c == var:
        return True
    ne = rf"^{v}!=(?:NULL|0)$"
    if re.fullmatch(ne, c):
        return bool(re.fullmatch(ne, c))
    return bool(re.fullmatch(rf"^(?:NULL|0)!={v}$", c))


# ─── function signature / range / parameter helpers ──────────────────────────

_DECLARATOR_PREFIX_RE = re.compile(r"[A-Za-z_][\w:<>,\*&\s]*")
_COMMENT_PREFIXES = ("//", "/*", "*", "#")


def _find_function_range(
    lines: list[str], name: str,
) -> tuple[int, int, str] | None:
    """``(start_line, end_line, params_text)`` of *name*'s definition, 1-based.

    Only single-line signatures with the body's ``{`` on the same or the next
    non-empty line are accepted; anything else (a split declaration, a
    macro-generated definition) returns ``None`` — the caller then produces no
    finding rather than matching a guard against the wrong scope.
    """
    pat = re.compile(rf"\b{re.escape(name)}\s*\(")
    for i, line in enumerate(lines):
        st = line.strip()
        if not st or st.startswith(_COMMENT_PREFIXES):
            continue
        for m in pat.finditer(line):
            idx = m.start()
            prefix = line[:idx].rstrip()
            if not prefix or not _DECLARATOR_PREFIX_RE.fullmatch(prefix):
                continue
            if ";" in line[idx:]:
                continue
            try:
                op = line.index("(", idx)
            except ValueError:  # pragma: no cover — the regex required it
                continue
            close = _matching_paren(line, op)
            if close is None:
                return None
            args = line[op + 1:close]
            rest = line[close + 1:].strip()
            if rest.startswith(";"):
                continue  # a declaration, not a definition
            brace_line = i
            if not rest.startswith("{"):
                # The body's `{` sits on a later line: accept exactly one
                # non-empty line of lag, otherwise give up.
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j >= len(lines) or not lines[j].strip().startswith("{"):
                    continue
                brace_line = j
            end = _body_end_line(lines, brace_line)
            if end is None:
                return None
            return (i + 1, end, args)
    return None


def _body_end_line(lines: list[str], brace_line: int) -> int | None:
    """1-based line of the ``}`` closing the body opened on *brace_line*."""
    depth = 0
    started = False
    for i in range(brace_line, len(lines)):
        for ch in _strip_line_noise(lines[i]):
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return i + 1
    return None


def _strip_line_noise(line: str) -> str:
    """Drop ``//`` comments and string/char literal contents from *line*."""
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch in "\"'":
            quote = ch
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _split_params(params: str) -> list[str]:
    """Top-level comma split of a parameter list."""
    if not params.strip() or params.strip() == "void":
        return []
    out: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(params):
        if ch in "([<{":
            depth += 1
        elif ch in ")]>}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(params[start:i].strip())
            start = i + 1
    out.append(params[start:].strip())
    return [p for p in out if p]


def _param_is_pointer_to(params: str, var: str) -> bool:
    """Whether *params* declares *var* as a pointer parameter."""
    for p in _split_params(params):
        if "(" in p:  # function-pointer parameter — unanalyzable here
            continue
        tokens = re.findall(r"[A-Za-z_]\w*", p)
        if not tokens or tokens[-1] != var:
            continue
        return "*" in p
    return False


# ─── source file resolution ──────────────────────────────────────────────────

def _resolve_local(file_path: str, source_root: str | Path | None) -> Path | None:
    """Map a CSA build path to the local source tree.

    Tries the path as-is, then the longest path *suffix* under *source_root*
    (``/home/me/projects/cpp-project/protobuf/src/a/b.cc`` →
    ``<source_root>/src/a/b.cc``).  Longest-suffix-first is deterministic, so a
    file whose basename occurs in several directories cannot silently resolve to
    the wrong one.
    """
    if not file_path:
        return None
    p = Path(file_path)
    if p.is_file():
        return p
    if not source_root:
        return None
    root = Path(source_root)
    parts = p.parts
    for i in range(1, len(parts)):
        cand = root.joinpath(*parts[i:])
        if cand.is_file():
            return cand
    return None


_LINE_CACHE: dict[tuple[str, float], list[str] | None] = {}


def _read_lines(path: Path) -> list[str] | None:
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    cached = _LINE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        logger.warning("entry-state: cannot read %s: %s", path, e)
        return None
    if len(_LINE_CACHE) > 256:  # cheap bound; the pipeline is short-lived
        _LINE_CACHE.clear()
    _LINE_CACHE[key] = lines
    return lines


# ─── call-site scan (corroboration only) ─────────────────────────────────────

_SOURCE_EXTS = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".inl", ".ipp")
_SKIP_DIRS = frozenset({
    ".git", ".svn", ".hg", "build", "builds", "node_modules", "__pycache__",
    ".cache", "out", ".idea", ".vscode",
})
_MAX_SCAN_FILES = 40_000
_MAX_SCAN_BYTES = 512 * 1024 * 1024
_MAX_FILE_BYTES = 8 * 1024 * 1024

# Characters that, immediately before the name, mean expression position.
_EXPR_CHARS = set("=(,[{;&|!?:+-*/%^~<>.")
_CALL_KEYWORDS = frozenset({
    "return", "co_return", "if", "while", "for", "switch", "case", "else",
    "do", "not", "and", "or", "throw", "delete", "new", "sizeof", "co_await",
})


@dataclass(frozen=True)
class CallSiteScan:
    """How the flagged function is called inside the source tree."""

    count: int
    samples: list[str] = field(default_factory=list)


_SCAN_CACHE: dict[tuple[str, str], "CallSiteScan | None"] = {}


def _classify_occurrence(line: str, idx: int) -> str:
    """``"definition"`` / ``"declaration"`` / ``"call"`` for a name at *idx*.

    Ambiguous shapes are classified as ``"call"``: over-counting call sites only
    makes the caller *less* willing to conclude, so the failure direction is
    safe.
    """
    prefix = line[:idx].rstrip()
    if prefix and prefix[-1] in _EXPR_CHARS:
        return "call"
    words = re.findall(r"[A-Za-z_]\w*", prefix)
    if words and words[-1] in _CALL_KEYWORDS:
        return "call"
    stripped = line.rstrip()
    if stripped.endswith(";") or stripped.endswith("{"):
        if prefix and _DECLARATOR_PREFIX_RE.fullmatch(prefix):
            return "definition" if stripped.endswith("{") else "declaration"
    return "call"


def scan_call_sites(
    source_root: str | Path | None, function_name: str,
) -> CallSiteScan | None:
    """Count real call sites of *function_name* under *source_root*.

    Returns ``None`` when the tree cannot be scanned (missing root, or the scan
    hit its size caps) — the caller must then treat the count as unknown, never
    as zero.
    """
    if not source_root or not function_name:
        return None
    root = Path(source_root)
    if not root.is_dir():
        return None
    key = (str(root), function_name)
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]

    name_re = re.compile(rf"\b{re.escape(function_name)}\b\s*\(")
    count = 0
    samples: list[str] = []
    files = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(_SOURCE_EXTS):
                continue
            path = Path(dirpath) / fn
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > _MAX_FILE_BYTES:
                continue
            files += 1
            total += size
            if files > _MAX_SCAN_FILES or total > _MAX_SCAN_BYTES:
                logger.info(
                    "entry-state: call-site scan of %s exceeded its size caps "
                    "(%d files / %d bytes) — count left unknown", root, files, total,
                )
                _SCAN_CACHE[key] = None
                return None
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if function_name not in text:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith(_COMMENT_PREFIXES):
                    continue
                for m in name_re.finditer(line):
                    if _classify_occurrence(line, m.start()) != "call":
                        continue
                    count += 1
                    if len(samples) < 3:
                        try:
                            rel = path.relative_to(root)
                        except ValueError:  # pragma: no cover
                            rel = path
                        samples.append(
                            f"{rel}:{lineno}: {' '.join(line.split())[:120]}"
                        )
    scan = CallSiteScan(count=count, samples=samples)
    _SCAN_CACHE[key] = scan
    logger.info(
        "entry-state: %s has %d real call site(s) under %s",
        function_name, count, root,
    )
    return scan


# ─── the finding ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EntryStateFinding:
    """The report's assumed entry state is refuted by the function's own check."""

    function: str
    file: str
    parameter: str
    guard_macro: str
    guard_condition: str
    guard_line: int
    guard_source: str
    guard_semantics: str
    assumption_event_number: int
    branch_event_number: int
    deref_line: int
    bug_type: str
    call_sites: int | None = None
    call_site_samples: list[str] = field(default_factory=list)

    kind = "fatal_guard_continued"

    def reason(self) -> str:
        f = Path(self.file).name
        parts = [
            f"报告入口状态不可实现：CSA 把 {f} 的形参 '{self.parameter}' "
            f"假设为 null（事件 #{self.assumption_event_number}），并要求执行继续到 "
            f"L{self.deref_line} 的 {self.bug_type}"
            f"（loaded from variable '{self.parameter}'）。"
            f"但 L{self.guard_line} 处的 {self.guard_macro}({self.guard_condition}, …) "
            f"是无条件编译的致命检查（{self.guard_semantics}）——"
            f"报告自身还记录在该行取得 TRUE 分支（事件 #{self.branch_event_number}，"
            f"即检查失败分支），失败分支不返回，故『该检查失败』与『继续解引用』"
            f"不可兼得，路径不可实现。"
        ]
        if self.call_sites is None:
            parts.append("（调用点扫描不可用，仅以上述检查为准）")
        elif self.call_sites == 0:
            parts.append(
                f"另外，函数 {self.function} 在工程源码内共 0 个真实调用点 —— "
                f"既无调用方能传入 null，也没有任何调用方走得到该解引用。"
            )
        else:
            sample = "；".join(self.call_site_samples)
            parts.append(
                f"（函数 {self.function} 在工程内有 {self.call_sites} 个调用点："
                f"{sample} —— 该结论不依赖调用点数量）"
            )
        return "".join(parts)


def detect_fabricated_entry_state(
    parsed, source_root: str | Path | None,
) -> EntryStateFinding | None:
    """Detect a manufactured entry state in *parsed*'s report, or ``None``.

    See the module docstring for the conditions — every one of them must hold,
    and every failure mode returns ``None`` (the pipeline then behaves exactly
    as it did before this check existed).
    """
    events = list(getattr(parsed, "path_events", None) or [])
    if len(events) < 3:
        return None
    md = getattr(parsed, "metadata", None)
    if md is None:
        return None
    bug_type = getattr(md, "bug_type", "") or ""
    if "null" not in bug_type.lower():
        return None

    # 1. the *entry* assumption: the report's first claim (index 1 is tolerated
    #    so an `Error Start` event may precede it) is a null pointer assumption.
    assumption_idx = None
    for i, ev in enumerate(events[:2]):
        if _ENTRY_NULL_RE.search(ev.get("description") or ""):
            assumption_idx = i
            break
    if assumption_idx is None:
        return None
    assumption = events[assumption_idx]

    # 2. which variable is dereferenced: the report's own final report event.
    report_ev = None
    for ev in reversed(events):
        if (ev.get("event_kind") or "") == "report":
            report_ev = ev
            break
    if report_ev is None:
        return None
    m = _NULL_VAR_RE.search(report_ev.get("description") or "")
    if not m:
        return None
    var = m.group(1)

    # 3. the flagged function's definition in the bug file.
    src = _resolve_local(getattr(md, "bug_file", "") or "", source_root)
    if src is None:
        return None
    lines = _read_lines(src)
    if lines is None:
        return None
    fn_name = getattr(md, "function_name", "") or ""
    if not fn_name:
        return None
    rng = _find_function_range(lines, fn_name)
    if rng is None:
        return None
    fn_start, fn_end, params = rng
    deref_line = int(getattr(md, "bug_line", 0) or 0)
    if not (fn_start <= deref_line <= fn_end):
        return None
    if not _param_is_pointer_to(params, var):
        return None

    # 4. an always-compiled check on that parameter *before* the deref.
    guard = None
    for lineno in range(fn_start, deref_line):
        line = lines[lineno - 1]
        if line.lstrip().startswith(_COMMENT_PREFIXES):
            continue
        gm = _FATAL_GUARD_RE.search(line)
        if not gm:
            continue
        try:
            op = line.index("(", gm.end() - 1)
        except ValueError:  # pragma: no cover
            continue
        cond = _first_arg(line, op)
        if cond is None or not _requires_non_null(cond, var):
            continue
        guard = (lineno, gm.group(1), cond, " ".join(line.split()))
        break
    if guard is None:
        return None
    guard_line, macro, cond, guard_source = guard

    # 5. the report must itself record taking the check's failure arm.
    branch_ev = None
    for ev in events:
        if ev.get("line_number") != guard_line:
            continue
        if ev.get("branch_condition") is True:
            branch_ev = ev
            break
    if branch_ev is None:
        return None

    scan = None
    try:
        scan = scan_call_sites(source_root, fn_name)
    except Exception as e:  # noqa: BLE001 — corroboration must never be fatal
        logger.warning("entry-state: call-site scan failed: %s", e)

    return EntryStateFinding(
        function=fn_name,
        file=str(src),
        parameter=var,
        guard_macro=macro,
        guard_condition=cond,
        guard_line=guard_line,
        guard_source=guard_source,
        guard_semantics=_GUARD_SEMANTICS.get(macro, "失败分支不返回"),
        assumption_event_number=int(assumption.get("path_number") or assumption_idx + 1),
        branch_event_number=int(branch_ev.get("path_number") or 0),
        deref_line=deref_line,
        bug_type=bug_type,
        call_sites=(scan.count if scan else None),
        call_site_samples=(list(scan.samples) if scan else []),
    )
