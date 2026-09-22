"""Project-wide constraint mining — discover global-state invariants from source.

Mines four types of constraints:
1. **Constructor invariants**: field initialization values and conditional init
2. **Field consistency**: relationships between co-occurring fields
3. **Configuration flags**: setter functions and their domains
4. **Singleton/global state**: lifecycle and initialization guarantees
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.constraint_models import (
    ConfigFlagInfo,
    ConstraintSource,
    FieldConsistencyRule,
    FieldInvariant,
    ParameterConstraint,
    ProjectConstraintDB,
    SingletonInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scanner: Constructor Invariants
# ---------------------------------------------------------------------------


def scan_constructor_invariants(file_path: Path) -> list[FieldInvariant]:
    """Scan a single .cc/.cpp file for constructor initializer lists.

    Finds patterns like::

        Class::Class(...)
            : field_(value1),
              field2_(value2)
        {
            if (cond) { field_ = new...; }
            else { field_ = nullptr; }
        }
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    invariants: list[FieldInvariant] = []

    # Pattern: ClassName::ClassName(params) : init1, init2, ... {
    ctor_re = re.compile(
        r"(\w+)::(\1)\s*\([^)]*\)\s*(?:\n\s*)?:"
    )

    init_re = re.compile(
        r"(\w+)_\(([^,)]+)\)"
    )

    for i, line in enumerate(lines):
        m = ctor_re.search(line)
        if not m:
            # Try multi-line: next line has ":"
            if i + 1 < len(lines) and "::" in line and "(" in line and ")" in line:
                combined = line + "\n" + lines[i + 1]
                m = ctor_re.search(combined)
        if not m:
            continue
        class_name = m.group(1)

        # Collect lines of the initializer list (until the opening {)
        init_lines: list[str] = [line]
        j = i + 1
        while j < len(lines) and "{" not in lines[j]:
            init_lines.append(lines[j])
            j += 1
        if j < len(lines):
            init_lines.append(lines[j])

        init_text = "\n".join(init_lines)

        # Extract each initializer: field_(value)
        for im in init_re.finditer(init_text):
            field = im.group(1)
            value = im.group(2).strip()
            inv = FieldInvariant(
                class_name=class_name,
                field_name=f"{field}_",
                initial_value=value,
                sources=[ConstraintSource(
                    file=str(file_path),
                    line=i + 1,
                    snippet=line.strip()[:100],
                )],
            )
            invariants.append(inv)

        # Also scan the constructor body for conditional init
        body_text = ""
        brace_depth = 0
        for k in range(j, min(j + 50, len(lines))):
            body_text += lines[k] + "\n"
            brace_depth += lines[k].count("{") - lines[k].count("}")
            if brace_depth > 0 and brace_depth <= 0:
                break

        # Look for "if (cond) { field_ = new...; } else { field_ = nullptr; }"
        cond_init_re = re.compile(
            r"if\s*\(([^)]+)\)\s*\{[^}]*?(\w+)_\s*=\s*(new\s[^;]+)"
            r"[^}]*?\}\s*else\s*\{[^}]*?\2_\s*=\s*(nullptr|NULL|0)\s*;"
        )
        for cm in cond_init_re.finditer(body_text):
            field = cm.group(2)
            cond = cm.group(1)
            new_val = cm.group(3)
            else_val = cm.group(4)
            invariants.append(FieldInvariant(
                class_name=class_name,
                field_name=f"{field}_",
                initial_value=new_val,
                is_conditional=True,
                condition=cond,
                confidence="medium",
                sources=[ConstraintSource(
                    file=str(file_path),
                    line=i + 1,
                    snippet=cm.group()[:120],
                )],
            ))

    return invariants


# ---------------------------------------------------------------------------
# Scanner: Field Implications (guard → allocation invariants)
# ---------------------------------------------------------------------------


def scan_field_implications(file_path: Path) -> list[FieldConsistencyRule]:
    """Scan for field implication rules from if-else allocation patterns.

    Pattern A (direct guard)::

        if (guardField_) {
            ptrField_ = new ... / malloc(...);
            ...
        } else {
            ...
            ptrField_ = nullptr / NULL;
        }
        → guardField_ == true ⇒ ptrField_ != nullptr

    Pattern B (inverted guard)::

        if (!guardField_) {
            ...
            ptrField_ = nullptr / NULL;
        } else {
            ptrField_ = new ... / malloc(...);
            ...
        }
        → guardField_ == true ⇒ ptrField_ != nullptr

    Pattern C (ensure-flag setter)::

        void enableFlag() {
            ensurePtrField();      // ensures ptrField_ is allocated
            flagField_ = true;     // then sets the flag
        }
        → flagField_ == true ⇒ ptrField_ != nullptr

    The resulting ``FieldConsistencyRule``\ s have ``rule_type="implication"``
    and are suitable for prompt injection as hard invariants.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rules: list[FieldConsistencyRule] = []

    # ------------------------------------------------------------------
    # Pattern A + B: if (guard_) { alloc } else { null }  (or swapped)
    # ------------------------------------------------------------------
    # Match even when ``else`` is on a separate line (common in aria2 style).
    pat_if_else = re.compile(
        r'if\s*\(\s*(\w+_)\s*\)'         # if (guardField_)
        r'\s*\{([^{}]*)\}'                # { if-body (no nested braces) }
        r'\s*else\s*'                     # else
        r'\s*\{([^{}]*)\}',               # { else-body (no nested braces) }
        re.DOTALL,
    )

    for m in pat_if_else.finditer(text):
        guard_field = m.group(1)
        if_body = m.group(2)
        else_body = m.group(3)

        # Look for "ptr_ = new/malloc" in if-body AND "ptr_ = nullptr" in else-body
        for alloc_m in re.finditer(
            r'(\w+_)\s*=\s*(?:new\s|malloc\s|calloc\s|realloc\s)', if_body
        ):
            ptr_field = alloc_m.group(1)
            null_pat = re.compile(
                rf'{re.escape(ptr_field)}\s*=\s*(?:nullptr|NULL|nullptr\w*|0)\s*;'
            )
            if null_pat.search(else_body):
                rules.append(FieldConsistencyRule(
                    description=(
                        f"`{guard_field}` == true ⇒ `{ptr_field}` != nullptr "
                        f"(allocated in if, nullified in else)"
                    ),
                    fields=[guard_field, ptr_field],
                    rule_type="implication",
                    confidence="high",
                    sources=[ConstraintSource(
                        file=str(file_path),
                        line=text[:m.start()].count('\n') + 1,
                        snippet=m.group()[:150],
                    )],
                ))

        # Also check inverted: if (!guard_) → if-body nullifies, else-body allocates
        alloc_in_else = re.finditer(
            r'(\w+_)\s*=\s*(?:new\s|malloc\s|calloc\s|realloc\s)', else_body
        )
        for alloc_m in alloc_in_else:
            ptr_field = alloc_m.group(1)
            null_pat = re.compile(
                rf'{re.escape(ptr_field)}\s*=\s*(?:nullptr|NULL|nullptr\w*|0)\s*;'
            )
            if null_pat.search(if_body):
                rules.append(FieldConsistencyRule(
                    description=(
                        f"`{guard_field}` == true ⇒ `{ptr_field}` != nullptr "
                        f"(nullified in if, allocated in else)"
                    ),
                    fields=[guard_field, ptr_field],
                    rule_type="implication",
                    confidence="high",
                    sources=[ConstraintSource(
                        file=str(file_path),
                        line=text[:m.start()].count('\n') + 1,
                        snippet=m.group()[:150],
                    )],
                ))

    # ------------------------------------------------------------------
    # Pattern C: setter → ensure → assign  (e.g. enableFilter)
    # ------------------------------------------------------------------
    # Find functions that set a field AND call an "ensure*" function.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        # Method signature: "ReturnType Class::method(...)"  [{ on this line or next]
        method_m = re.match(
            r'\s*(?:void|int|bool|size_t|int64_t|uint\d+t|\w+&?)\s+'
            r'(?:\w+::)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?',
            line,
        )
        if not method_m:
            continue
        method_name = method_m.group(1)

        # Check that a { appears on this line or the next (function definition, not declaration)
        rest_of_line = line[method_m.end():]
        next_line = lines[i + 1] if i + 1 < len(lines) else ''
        has_brace = '{' in rest_of_line or '{' in next_line
        if not has_brace:
            continue

        # Collect method body (up to 30 lines, approximate brace balance)
        body_lines: list[str] = []
        brace_depth = 0
        opened = False
        for j in range(i, min(i + 40, len(lines))):
            body_lines.append(lines[j])
            brace_depth += lines[j].count('{') - lines[j].count('}')
            if not opened and '{' in lines[j]:
                opened = True
            if opened and brace_depth <= 0:
                break
        body = '\n'.join(body_lines)

        # Look for: ensureCall(...); followed by field_ = true
        # OR: field_ = true; preceded by ensureCall(...)
        ensure_calls = re.findall(r'\b(ensure\w*|init\w*Field)\s*\(', body)
        assigns_true = re.findall(r'(\w+_)\s*=\s*(true|false)', body)

        if ensure_calls and assigns_true:
            # Check each ensure* call to see if it allocates a field
            for ensure_fn in set(ensure_calls):
                # Search the same file for the ensure function's body
                ensure_body = _find_function_body(text, ensure_fn)
                if not ensure_body:
                    continue

                # Does the ensure function allocate a specific field?
                alloc_fields = set(
                    re.findall(r'(\w+_)\s*=\s*(?:new\s|malloc\s|calloc\s)', ensure_body)
                )

                for (flag_field, flag_val) in set(assigns_true):
                    for ptr_field in alloc_fields:
                        if flag_val == 'true' and ptr_field:
                            rules.append(FieldConsistencyRule(
                                description=(
                                    f"`{flag_field}` == true ⇒ `{ptr_field}` != nullptr "
                                    f"(setter `{method_name}` calls `{ensure_fn}` "
                                    f"which allocates `{ptr_field}`)"
                                ),
                                fields=[flag_field, ptr_field],
                                rule_type="implication",
                                confidence="high",
                                sources=[ConstraintSource(
                                    file=str(file_path),
                                    line=i + 1,
                                    snippet=line.strip()[:120],
                                )],
                            ))

    return rules


def _find_function_body(text: str, fn_name: str) -> str | None:
    """Extract the body of a function named *fn_name* from *text*.

    Handles::

        ReturnType fn_name(Params) {
            body...
        }

    Returns the full function text (header + body), or ``None``.
    Uses a fixed-position scan to avoid catastrophic backtracking.
    """
    # Search for fn_name followed by (params) { using a line-by-line approach
    lines = text.splitlines()
    for i, line in enumerate(lines):
        idx = line.find(fn_name)
        if idx < 0:
            continue
        # Check that this is a function definition: fn_name(...) ... {
        if not re.search(r'\b' + re.escape(fn_name) + r'\s*\(', line):
            continue
        # Find the opening brace
        remaining = '\n'.join(lines[i:])
        brace_pos = remaining.find('{')
        if brace_pos < 0:
            continue
        # Track brace depth from the opening brace
        absolute_brace = sum(len(l) + 1 for l in lines[:i]) + brace_pos
        depth = 0
        started = False
        for j in range(absolute_brace, len(text)):
            if text[j] == '{':
                depth += 1
                started = True
            elif text[j] == '}':
                depth -= 1
                if started and depth == 0:
                    return text[max(0, absolute_brace - 80):j + 1].strip()
        break
    return None


# ---------------------------------------------------------------------------
# Scanner: Field Consistency
# ---------------------------------------------------------------------------


def scan_field_consistency(file_path: Path) -> list[FieldConsistencyRule]:
    """Scan for patterns where multiple fields are co-referenced in conditions.

    Finds patterns like::

        if (filterEnabled_) {
            ... filterBitfield_[i] ...
        }

        if (filterBitfield_ == nullptr) { return 0; }
        ... filterBitfield_ ...
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    rules: list[FieldConsistencyRule] = []

    # Pattern 1: "if (field_A_) { ... field_B_ ... }" or
    #            "if (field_A_) { ... use(field_B_) ... } else { ... }"
    # Pattern 2: "if (!field_A_) { ... } ... field_A_ ..."
    for i, line in enumerate(lines):
        # Find field pairs mentioned together in conditions
        m = re.search(
            r"if\s*\((!?\s*(\w+_)\s*\)?)\s*\{",
            line,
        )
        if not m:
            continue

        field1 = m.group(2)

        # Scan next few lines for other fields
        context = "\n".join(lines[i:min(i + 10, len(lines))])
        field_matches = set(re.findall(r"(\w+_)\s*(?:\[|\!=|==|\))", context))
        field_matches.discard(field1)

        for field2 in field_matches:
            if field2 in ("nullptr_", "NULL_", "0_"):
                continue
            rules.append(FieldConsistencyRule(
                description=(
                    f"`{field1}` and `{field2}` co-occur in conditional "
                    f"at {file_path.name}:{i + 1}"
                ),
                fields=[field1, field2],
                rule_type="implication",
                sources=[ConstraintSource(
                    file=str(file_path),
                    line=i + 1,
                    snippet=line.strip()[:100],
                )],
            ))

    return rules


# ---------------------------------------------------------------------------
# Scanner: Configuration Flags
# ---------------------------------------------------------------------------


def scan_configuration_flags(file_path: Path) -> list[ConfigFlagInfo]:
    """Scan for setter functions that configure flags/state.

    Finds patterns like::

        void wslay_event_config_set_no_buffering(ctx, int val) {
            ctx->config = val ? ... : ...;
        }

        void BitfieldMan::enableFilter() {
            ensureFilterBitfield();
            filterEnabled_ = true;
        }
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    flags: list[ConfigFlagInfo] = []

    # Pattern: setter function + assignment
    setter_re = re.compile(
        r"(?:void|int|bool)\s+(?:.*::)?(?:set|enable|disable|config_set)"
        r"(\w+)\s*\([^)]*\)\s*\{"
    )

    for i, line in enumerate(lines):
        m = setter_re.search(line)
        if not m:
            continue

        context = "\n".join(lines[i:min(i + 15, len(lines))])

        # Extract flag name from common assignment patterns
        flag_m = re.search(
            r"(?:(\w+)(?:->|\.)(\w+)\s*=\s*|(\w+)_\s*=\s*(?:true|false|\d))",
            context,
        )
        if flag_m:
            flag_name = flag_m.group(2) or flag_m.group(3) or ""
            is_bool = "true" in context or "false" in context
            flags.append(ConfigFlagInfo(
                flag_name=flag_name,
                setter_function=m.group(0).split("{")[0].strip(),
                setter_file=str(file_path),
                typical_values=["true", "false"] if is_bool else ["0", "1"],
                description=f"Controlled by `{m.group(1) + m.group(0)}`",
            ))

    return flags


# ---------------------------------------------------------------------------
# Scanner: Singleton / Global State
# ---------------------------------------------------------------------------


def scan_singleton_patterns(file_path: Path) -> list[SingletonInfo]:
    """Scan for singleton and global state patterns.

    Finds::

        static std::unique_ptr<T>& instance() { return instance_; }
        static const std::shared_ptr<T>& getInstance();
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    singletons: list[SingletonInfo] = []

    # Pattern: static ... instance() or getInstance()
    sing_re = re.compile(
        r"static.*\b(?:instance|getInstance)\s*\([^)]*\)"
    )
    for m in sing_re.finditer(text):
        singletons.append(SingletonInfo(
            class_name=str(file_path.stem),
            accessor=m.group()[:80],
        ))

    return singletons


# ---------------------------------------------------------------------------
# Helper: matching paren / arg splitting
# ---------------------------------------------------------------------------


def _matching_paren(text: str, open_pos: int) -> int:
    """Find the position of the closing paren matching the opening at *open_pos*."""
    if open_pos < 0 or open_pos >= len(text) or text[open_pos] != '(':
        return open_pos
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _split_args(args_text: str) -> list[str]:
    """Split comma-separated arguments handling nested parens/brackets.

    Only tracks ``()``, ``[]``, ``{}`` nesting — **not** ``<>`` (to avoid
    counting ``->``, ``operator<``, etc. as nesting).
    """
    args = []
    depth = 0
    current: list[str] = []
    for c in args_text:
        if c in '([{':
            depth += 1
            current.append(c)
        elif c in ')]}':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
    remaining = ''.join(current).strip()
    if remaining:
        args.append(remaining)
    return args


# ---------------------------------------------------------------------------
# Helper: find getter method return expression
# ---------------------------------------------------------------------------

_GETTER_CACHE: dict[str, str] = {}  # f"file:method_name" -> return expression


def _find_getter_return(text: str, method_name: str) -> str | None:
    """Find a simple getter's return expression by line-by-line scan.

    Only matches the simplest pattern::

        ReturnType getX() const? { return field_; }

    Returns the field name (e.g. ``blockLength_``) or ``None``.
    """
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if method_name not in line:
            continue
        # Check for inline getter: ... getX() const? { return Y; }
        m = re.search(
            r'\b' + re.escape(method_name) + r'\s*\(\s*\)\s*(?:const|noexcept|override)*\s*\{',
            line,
        )
        if not m:
            # Try multi-line: } getX(  ⟶  ) const { return Y; }
            if i + 1 < len(lines):
                combined = line + ' ' + lines[i + 1]
                m = re.search(
                    r'\b' + re.escape(method_name) + r'\s*\(\s*\)\s*(?:const|noexcept|override)*\s*\{',
                    combined,
                )
        if m:
            # Check this line or next 2 for "return X;"
            for j in range(max(0, i - 1), min(len(lines), i + 3)):
                ret_m = re.search(r'\breturn\s+(\w[\w_]*)\s*;', lines[j])
                if ret_m:
                    return ret_m.group(1).strip()
    return None


def _find_config_default(text: str, pref_name: str) -> str | None:
    """Search *text* for an OptionHandlerFactory definition of *pref_name* and extract its default.

    E.g. ``PREF_PIECE_LENGTH, TEXT_PIECE_LENGTH, "1M", 1_m, 1_g``
    returns ``"1M"``.
    """
    pat = re.compile(
        re.escape(pref_name) + r'\s*,[^,]+,\s*"([^"]+)"',
    )
    m = pat.search(text)
    if m:
        return m.group(1)
    # Try single-quoted default
    pat2 = re.compile(
        re.escape(pref_name) + r'\s*,[^,]+,\s*\'([^\']+)\'',
    )
    m2 = pat2.search(text)
    if m2:
        return m2.group(1)
    return None


# ---------------------------------------------------------------------------
# Scanner: Constructor / Function Call-Site Parameter Constraints
# ---------------------------------------------------------------------------


def scan_parameter_constraints(
    source_root: str | Path,
    target_classes: list[str] | None = None,
    extensions: tuple[str, ...] = (".cc", ".cpp", ".c"),
) -> list[ParameterConstraint]:
    """Scan source for constructor call sites of *target_classes* and derive parameter constraints.

    For each call site, classifies arguments into:
    - **Literal constants** (e.g. ``100``, ``"1M"``)
    - **Config options** (e.g. ``option->getAsInt(PREF_PIECE_LENGTH)``)
    - **Method calls** (e.g. ``downloadContext->getPieceLength()``) — with one-level
      tracing into the getter's return expression.
    - **Other** (recorded at low confidence)

    Results are deduplicated: if multiple call sites agree on the same constraint
    (e.g. every caller passes an option value), confidence is ``"high"``.
    """
    root = Path(source_root).resolve()
    if not root.exists() or not target_classes:
        return []

    # Pre-build a getter-return cache from header files
    getter_cache: dict[str, str] = {}
    header_extensions = (".h", ".hpp")
    for hext in header_extensions:
        for hf in sorted(root.rglob(f"*{hext}")):
            if not hf.is_file():
                continue
            try:
                htext = hf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Scan for inline getters: ReturnType getX() const? { return Y; }
            for m in re.finditer(
                r'(\w+)\s*\(\s*\)\s*(?:const|noexcept|override)*\s*\{[^}]*\breturn\s+(\w[\w_]*)\s*;',
                htext,
            ):
                name = m.group(1)
                ret = m.group(2)
                # Avoid clobbering with a longer (less likely) match
                if name not in getter_cache or len(name) > len(getter_cache.get(name, "")):
                    getter_cache[name] = ret

    result: list[ParameterConstraint] = []
    target_set = set(target_classes)
    seen: dict[tuple[str, str, str], list[ParameterConstraint]] = {}  # (class, func, param) -> list

    for ext in extensions:
        for fpath in sorted(root.rglob(f"*{ext}")):
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for cls_name in target_set:
                # Pattern A: make_unique<ClassName>(args) / make_shared<ClassName>(args)
                for m in re.finditer(
                    rf'make_(?:unique|shared)\s*<\s*{re.escape(cls_name)}\s*>\s*\(',
                    text,
                ):
                    end = _matching_paren(text, m.end() - 1)
                    args_text = text[m.end():end]
                    args = _split_args(args_text)
                    file_rel = str(fpath.relative_to(root))
                    line_no = text[:m.start()].count('\n') + 1
                    _analyze_and_record(
                        result, seen, cls_name, cls_name,
                        args, file_rel, line_no, text, root, getter_cache,
                    )

                # Pattern B: ClassName varName(args) — but NOT a method definition (no ::ClassName)
                pat_b = re.compile(
                    r'(?<![:\w])' + re.escape(cls_name) + r'\s+(\w+)\s*\(',
                )
                for m in pat_b.finditer(text):
                    # Skip method definitions / declarations
                    start_ctx = text[max(0, m.start() - 80):m.start()]
                    if '::' in start_ctx or 'template' in start_ctx:
                        continue
                    # Skip obvious declarations (return type before class name)
                    if re.search(r'\b(?:void|int|bool|char|long|size_t|auto|static|virtual)\s+\w', start_ctx):
                        continue
                    end = _matching_paren(text, m.end() - 1)
                    if end <= m.end():
                        continue
                    args_text = text[m.end():end]
                    args = _split_args(args_text)
                    if not args:
                        continue
                    file_rel = str(fpath.relative_to(root))
                    line_no = text[:m.start()].count('\n') + 1
                    _analyze_and_record(
                        result, seen, cls_name, cls_name,
                        args, file_rel, line_no, text, root, getter_cache,
                    )

                # Pattern C: new ClassName(args)
                for m in re.finditer(
                    rf'\bnew\s+{re.escape(cls_name)}\s*\(',
                    text,
                ):
                    end = _matching_paren(text, m.end() - 1)
                    if end <= m.end():
                        continue
                    args_text = text[m.end():end]
                    args = _split_args(args_text)
                    if not args:
                        continue
                    file_rel = str(fpath.relative_to(root))
                    line_no = text[:m.start()].count('\n') + 1
                    _analyze_and_record(
                        result, seen, cls_name, cls_name,
                        args, file_rel, line_no, text, root, getter_cache,
                    )

    # Deduplicate: merge by (class_name, function_name, simplified_constraint)
    merged: dict[str, ParameterConstraint] = {}
    for pc in result:
        key = f"{pc.class_name}|{pc.function_name}|{pc.parameter_name}|{pc.constraint_expr}"
        if key in merged:
            # Merge evidence sites
            merged[key].evidence_sites.extend(pc.evidence_sites)
            merged[key].evidence_sites = merged[key].evidence_sites[:20]  # cap
            # Upgrade confidence if multiple sites agree
            if merged[key].confidence == "low" and pc.confidence == "medium":
                merged[key].confidence = "medium"
        else:
            merged[key] = pc

    logger.info(
        "Mined %d parameter constraints (from %d raw) for %d target classes",
        len(merged), len(result), len(target_set),
    )
    return list(merged.values())


def _analyze_and_record(
    result: list[ParameterConstraint],
    seen: dict,
    cls_name: str,
    func_name: str,
    args: list[str],
    file_rel: str,
    line_no: int,
    file_text: str,
    project_root: Path | None = None,
    getter_cache: dict[str, str] | None = None,
) -> None:
    """Analyze constructor arguments and record constraints."""
    from llm_client.fp_analysis.constraint_models import ConstraintSource

    source = ConstraintSource(file=file_rel, line=line_no, snippet="")

    # Pre-load OptionHandlerFactory text for config default lookup
    _ohf_text: str | None = None
    if project_root:
        try:
            for ohf_file in sorted(project_root.rglob("*OptionHandlerFactory*")):
                if ohf_file.suffix in (".cc", ".cpp", ".c") and ohf_file.is_file():
                    _ohf_text = ohf_file.read_text(encoding="utf-8", errors="replace")
                    break
        except Exception:
            pass

    for i, arg in enumerate(args):
        arg_stripped = arg.strip()
        param_name = f"arg{i}"

        # 1. Literal number
        lit_m = re.match(r'^[+-]?\d+(?:\.\d+)?(?:[uU][lL]?[lL]?)?$', arg_stripped)
        if lit_m:
            result.append(ParameterConstraint(
                class_name=cls_name,
                function_name=func_name,
                parameter_name=param_name,
                constraint_expr=f"== {arg_stripped}",
                derived_from=f"literal constant at {file_rel}:{line_no}",
                confidence="high",
                evidence_sites=[source],
            ))
            continue

        # 2. String literal
        str_m = re.match(r'^"([^"]*)"$', arg_stripped)
        if str_m:
            result.append(ParameterConstraint(
                class_name=cls_name,
                function_name=func_name,
                parameter_name=param_name,
                constraint_expr=f'== "{str_m.group(1)}"',
                derived_from=f"string literal at {file_rel}:{line_no}",
                confidence="high",
                evidence_sites=[source],
            ))
            continue

        # 3. Config option: option->getAsInt(PREF_X) or similar
        opt_m = re.search(r'PREF_(\w+)', arg_stripped)
        if opt_m:
            pref_name = f"PREF_{opt_m.group(1)}"
            default_val = None
            if _ohf_text:
                default_val = _find_config_default(_ohf_text, pref_name)
            default_str = f", default {default_val}" if default_val else ""
            result.append(ParameterConstraint(
                class_name=cls_name,
                function_name=func_name,
                parameter_name=param_name,
                constraint_expr=f"> 0 (config option{default_str})",
                derived_from=f"config option {pref_name}{default_str}",
                confidence="high",
                evidence_sites=[source],
            ))
            continue

        # 4. Method call: obj->getX() or obj.getX()
        method_m = re.search(r'[\.\->](\w+)\s*\([^)]*\)\s*$', arg_stripped)
        if method_m:
            method_name = method_m.group(1)
            # Check current file first, then header cache
            ret_expr = _find_getter_return(file_text, method_name)
            if not ret_expr and getter_cache:
                ret_expr = getter_cache.get(method_name)
            if ret_expr and ret_expr != method_name:  # avoid self-referential
                result.append(ParameterConstraint(
                    class_name=cls_name,
                    function_name=func_name,
                    parameter_name=param_name,
                    constraint_expr=f"== (result of {method_name} → returns `{ret_expr}`)",
                    derived_from=f"method call `{arg_stripped}` at {file_rel}:{line_no}, "
                                 f"where {method_name} returns `{ret_expr}`",
                    confidence="low",
                    evidence_sites=[source],
                ))
            else:
                result.append(ParameterConstraint(
                    class_name=cls_name,
                    function_name=func_name,
                    parameter_name=param_name,
                    constraint_expr="(derived from method call, needs tracing)",
                    derived_from=f"method call `{arg_stripped}` at {file_rel}:{line_no}",
                    confidence="low",
                    evidence_sites=[source],
                ))
            continue

        # 5. Variable / other (record at low confidence)
        var_m = re.match(r'(\w+)', arg_stripped)
        if var_m:
            result.append(ParameterConstraint(
                class_name=cls_name,
                function_name=func_name,
                parameter_name=param_name,
                constraint_expr="(variable — uncertain)",
                derived_from=f"variable `{arg_stripped}` at {file_rel}:{line_no}",
                confidence="low",
                evidence_sites=[source],
            ))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def mine_all_constraints(
    source_root: str | Path,
    project_name: str = "unknown",
    extensions: tuple[str, ...] = (".cc", ".cpp", ".c"),
    target_classes: list[str] | None = None,
) -> ProjectConstraintDB:
    """Run all scanners across all source files in *source_root*.

    *target_classes* is an optional list of class names for which to mine
    parameter constraints (constructor call-site analysis).  If omitted,
    classes discovered via constructor-invariant scanning are used.

    Returns a ``ProjectConstraintDB`` with all mined constraints.
    """
    root = Path(source_root).resolve()
    if not root.exists():
        logger.warning("Source root not found: %s", root)
        return ProjectConstraintDB(project_name=project_name)

    db = ProjectConstraintDB(project_name=project_name)

    for ext in extensions:
        for fpath in sorted(root.rglob(f"*{ext}")):
            if not fpath.is_file():
                continue
            try:
                invs = scan_constructor_invariants(fpath)
                db.invariants.extend(invs)

                rules = scan_field_consistency(fpath)
                db.consistency_rules.extend(rules)

                impls = scan_field_implications(fpath)
                db.consistency_rules.extend(impls)

                cfgs = scan_configuration_flags(fpath)
                db.config_flags.extend(cfgs)

                sgls = scan_singleton_patterns(fpath)
                db.singleton_info.extend(sgls)
            except Exception as e:
                logger.warning("Error scanning %s: %s", fpath, e)

    # Parameter constraints: use discovered class names + explicitly provided targets
    param_targets: list[str] = target_classes or []
    if not param_targets:
        discovered = set(inv.class_name for inv in db.invariants)
        param_targets = sorted(discovered)
    if param_targets:
        param_constraints = scan_parameter_constraints(
            root, param_targets, extensions,
        )
        db.parameter_constraints.extend(param_constraints)

    logger.info(
        "Mined %d invariants, %d consistency rules, %d config flags, "
        "%d singletons, %d parameter constraints from %s",
        len(db.invariants), len(db.consistency_rules),
        len(db.config_flags), len(db.singleton_info),
        len(db.parameter_constraints),
        project_name,
    )

    return db
