"""Symbolic test harness generator for CSA path extraction.

Generates a C++ program that invokes a target function with symbolic inputs,
so CSA can explore all feasible execution paths and extract path constraints.

Symbolization strategies per parameter type:
  - Integer types (int, size_t, char, etc.) → scanf("%<fmt>", &param)
  - Pointer types (void*, char*, etc.)      → malloc(size) / free() pair
  - Array/buffer                            → malloc + memset
  - struct by value                         → bitwise copy via memcpy from symbolic
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# C type → scanf format map
# ---------------------------------------------------------------------------
SCANF_FORMATS = {
    "int":                "%d",
    "unsigned int":       "%u",
    "long":               "%ld",
    "unsigned long":      "%lu",
    "long long":          "%lld",
    "unsigned long long": "%llu",
    "size_t":             "%zu",
    "ssize_t":            "%zd",
    "char":               "%hhd",
    "signed char":        "%hhd",
    "unsigned char":      "%hhu",
    "short":              "%hd",
    "unsigned short":     "%hu",
    "float":              "%f",
    "double":             "%lf",
    "bool":               "%d",  # bool as int
}

# Types that should be treated as pointers/arrays
POINTER_LIKE = re.compile(
    r"(.*\*|.*\[\].*|^void\s*\*|^char\s*\*|^FILE\s*\*)"
)

# Known common typedefs / struct types that should not be scanf'd
COMPLEX_TYPES = re.compile(
    r"^(struct |class |std::|folly::|boost::)"
)


class ParamInfo:
    """Describes a single function parameter extracted from source or report."""
    def __init__(self, name: str, param_type: str, declaration: str = ""):
        self.name = name
        self.type = param_type.strip()
        self.declaration = declaration or f"{self.type} {self.name}"
        self.stripped_type = re.sub(
            r'__(restrict|restrict__|attribute__\(\(.*?\)\))\s*', '',
            self.type
        ).strip()
        self.is_pointer = bool(POINTER_LIKE.match(self.stripped_type))
        self.is_complex = bool(COMPLEX_TYPES.match(self.stripped_type))

    def scanf_format(self) -> Optional[str]:
        """Get scanf format for this type, if applicable."""
        base_type = re.sub(r"\s*\*+\s*$", "", self.stripped_type).strip()
        # Handle const
        base_type = re.sub(r"^const\s+", "", base_type).strip()
        # Handle references
        base_type = re.sub(r"\s*&+\s*$", "", base_type).strip()

        # Check direct match
        if base_type in SCANF_FORMATS:
            return SCANF_FORMATS[base_type]
        # Check with std:: prefix
        if base_type.replace("std::", "") in SCANF_FORMATS:
            return SCANF_FORMATS[base_type.replace("std::", "")]
        return None


def parse_function_params(source_code: str) -> list[ParamInfo]:
    """Parse function parameter list from source code at the bug line.

    Uses a simple regex-based approach. For production, use clang tooling.
    """
    # Find function signature containing the bug line
    # Match: return_type function_name(params) {
    # (handles ClassName::methodName and ~dtor)
    func_pattern = re.compile(
        r"([\w:<>,\s\*&]+)\s+([\w:~]+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:\{|override|final)",
        re.DOTALL
    )
    # Try multi-line function matching
    # First, find all function-like patterns
    lines = source_code.split('\n')
    full_text = source_code

    matches = list(func_pattern.finditer(full_text))
    params = []
    seen_names = set()

    for m in matches:
        decl = m.group(0)
        ret_type = m.group(1).strip()
        func_name = m.group(2).strip()
        param_text = m.group(3).strip()

        # Exclude control flow keywords
        if func_name in ("if", "while", "for", "switch", "catch", "else"):
            continue

        for p in split_params(param_text):
            p = p.strip()
            if not p or p == "void":
                continue
            # Parse "type name" or "type &name" or "type* name"
            # Handle default values: remove from declaration
            p_clean = re.sub(r"\s*=\s*[^,]+$", "", p).strip()

            # Split into type and name
            parts = p_clean.split()
            if len(parts) >= 2:
                # Handle "const type* name" etc.
                # Last word is the parameter name, rest is the type
                ptype = " ".join(parts[:-1])
                pname = parts[-1].lstrip("*&")  # strip leading pointer/reference markers
            elif len(parts) == 1:
                # Unnamed parameter (type only)
                ptype = parts[0]
                pname = f"arg_{len(params)}"

            # Clean up
            pname = pname.rstrip(",;")

            if pname and pname not in seen_names:
                seen_names.add(pname)
                params.append(ParamInfo(pname, ptype, p_clean))

    return params


def split_params(param_text: str) -> list[str]:
    """Split parameter text handling nested templates."""
    params = []
    depth = 0
    current = ""
    for ch in param_text:
        if ch in '<(':
            depth += 1
            current += ch
        elif ch in ')>':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        params.append(current.strip())
    return params


def generate_harness(
    func_name: str,
    params: list[ParamInfo],
    return_type: str = "void",
    includes: list[str] = None,
    project_root: Optional[Path] = None,
    extra_init: str = "",
) -> str:
    """Generate a C++ test harness with symbolic inputs.

    Each parameter is symbolized according to its type:
      - Integer → scanf()
      - Pointer  → malloc() + free()
      - Complex  → pass-through (needs manual setup)

    Args:
        func_name: Name of the function to test.
        params: Parameter list.
        return_type: Function return type.
        includes: Additional #include directives.
        project_root: Project root for resolving include paths.
        extra_init: Extra C++ code to insert before the function call.

    Returns:
        Complete C++ source code.
    """
    lines: list[str] = []
    incs = list(includes or [])
    normalized = []
    for h in incs:
        h = h.strip()
        if h.startswith("#include") or h.startswith("#  include"):
            normalized.append(h)
        elif h.startswith("<") and h.endswith(">"):
            normalized.append(f"#include {h}")
        else:
            normalized.append(f"#include <{h}>")
    incs = normalized

    # Add standard includes
    std_includes = [
        '#include <cstddef>',
        '#include <cstdio>',
        '#include <cstdlib>',
        '#include <cstring>',
    ]
    for si in std_includes:
        if si not in incs:
            incs.append(si)

    lines.extend(incs)
    lines.append("")

    # Forward declaration is omitted — the #included source file
    # already provides the declaration (or definition).  A separate
    # forward decl would conflict when the function lives in a
    # namespace or is a class method.

    # main() with symbolic harness
    lines.append("int main() {")

    # For each parameter: declare and symbolize
    var_decls = []
    cleanup_stmts = []
    for i, p in enumerate(params):
        var_name = f"sym_{p.name}"

        # Clean type for declaration
        decl_type = re.sub(r'\s*__(restrict|restrict__)\s*', ' ', p.type).strip()
        decl_type = re.sub(r'\s+', ' ', decl_type)

        if p.is_pointer:
            # Pointer: malloc with symbolic size
            size_var = f"sym_{p.name}_size"
            lines.append(f"    // Symbolic pointer: {p.name}")
            lines.append(f"    size_t {size_var};")
            lines.append(f"    scanf(\"%zu\", &{size_var});")
            # For const pointers, strip const from decl for malloc, then cast
            malloc_type = re.sub(r'\bconst\b', '', decl_type).strip()
            if malloc_type.endswith('*'):
                lines.append(f"    {decl_type} {var_name} = ({decl_type})malloc({size_var} ? {size_var} : 1);")
            else:
                lines.append(f"    {decl_type} {var_name};  // TODO: pointer-like but complex")
            cleanup_stmts.append(f"    free((void*){var_name});")
            var_decls.append((var_name, p))
        elif p.is_complex:
            lines.append(f"    // Complex type, cannot auto-symbolize: {p.name}")
            lines.append(f"    {decl_type} {var_name};")
            var_decls.append((var_name, p))
        else:
            # Integer/primitive: scanf
            fmt = p.scanf_format()
            if fmt:
                lines.append(f"    // Symbolic integer: {p.name}")
                lines.append(f"    {decl_type} {var_name};")
                lines.append(f"    scanf(\"{fmt}\", &{var_name});")
            else:
                lines.append(f"    // Unknown type: {p.name} ({p.type})")
                lines.append(f"    {decl_type} {var_name};")
            var_decls.append((var_name, p))

    lines.append("")

    # Extra init code
    if extra_init:
        lines.append(f"    {extra_init}")
        lines.append("")

    # Call the target function
    call_args = [vd[0] for vd in var_decls]
    if return_type == "void":
        lines.append(f"    // Invoke target with symbolic inputs")
        lines.append(f"    {func_name}({', '.join(call_args)});")
    else:
        lines.append(f"    // Invoke target with symbolic inputs")
        lines.append(f"    {return_type} result = {func_name}({', '.join(call_args)});")
        lines.append(f"    // Prevent unused warning")
        lines.append(f"    (void)result;")

    lines.append("")

    # Cleanup
    for c in cleanup_stmts:
        lines.append(c)

    lines.append("    return 0;")
    lines.append("}")

    return "\n".join(lines)


def save_harness(code: str, output_dir: Path, func_name: str) -> Path:
    """Save generated harness to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', func_name)
    path = output_dir / f"harness_{safe_name}.cpp"
    path.write_text(code)
    return path


def generate_entry_point_harness(
    entry_point_name: str,
    entry_point_source: Path,
    params: list[ParamInfo],
    return_type: str,
    project_root: Path,
    has_main_in_source: bool = False,
    extra_init: str = "",
) -> str:
    """Generate a symbolic test harness for an *entry point* function.

    Unlike ``generate_harness()`` which creates a harness for the bug
    function directly, this wraps an **entry point** (e.g. ``main()``,
    a public API) so that CSA can trace a path from a real project
    entry point through the full call chain to the bug location.

    Args:
        entry_point_name: Qualified name of the entry point function.
        entry_point_source: Path to the source file that defines the entry point.
        params: Parameter descriptors for the entry point.
        return_type: Return type of the entry point.
        project_root: Project root (for relative path display).
        has_main_in_source: If True, the source file defines its own ``main()``.
                            We rename it with a macro to avoid redefinition.
        extra_init: Extra C++ code before the call (e.g. ``using namespace``).

    Returns:
        Complete C++ source code.
    """
    lines: list[str] = []

    # Standard includes
    includes = [
        '#include <cstddef>',
        '#include <cstdio>',
        '#include <cstdlib>',
        '#include <cstring>',
    ]

    # Add the raw #include of the entry point source
    raw_include = f'#include "{entry_point_source.resolve()}"'
    if has_main_in_source:
        # Rename the source's main() to avoid conflict with our harness main()
        includes.insert(0, '#define main __cg_original_main')
    includes.insert(0, raw_include)

    lines.extend(includes)
    lines.append("")

    # main() with symbolic harness
    lines.append("int main(int argc, char **argv) {")
    lines.append("    (void)argc; (void)argv;")

    # For each parameter: declare and symbolize
    var_decls: list[str] = []
    cleanup_stmts: list[str] = []
    for i, p in enumerate(params):
        var_name = f"sym_{p.name}"

        decl_type = re.sub(r'\s+', ' ', p.type).strip()

        if p.is_pointer:
            size_var = f"sym_{p.name}_size"
            lines.append(f"    // Symbolic pointer: {p.name}")
            lines.append(f"    size_t {size_var};")
            lines.append(f"    scanf(\"%zu\", &{size_var});")
            malloc_type = re.sub(r'\bconst\b', '', decl_type).strip()
            if malloc_type.endswith('*'):
                lines.append(f"    {decl_type} {var_name} = ({decl_type})malloc({size_var} ? {size_var} : 1);")
            else:
                lines.append(f"    {decl_type} {var_name};  // TODO: pointer-like but complex")
            cleanup_stmts.append(f"    free((void*){var_name});")
            var_decls.append(var_name)
        elif p.is_complex:
            lines.append(f"    // Complex type, cannot auto-symbolize: {p.name}")
            lines.append(f"    {decl_type} {var_name};")
            var_decls.append(var_name)
        else:
            fmt = p.scanf_format()
            if fmt:
                lines.append(f"    // Symbolic integer: {p.name}")
                lines.append(f"    {decl_type} {var_name};")
                lines.append(f"    scanf(\"{fmt}\", &{var_name});")
            else:
                lines.append(f"    // Unknown type: {p.name} ({p.type})")
                lines.append(f"    {decl_type} {var_name};")
            var_decls.append(var_name)

    lines.append("")

    # Extra init code (e.g. using namespace)
    if extra_init:
        lines.append(f"    {extra_init}")
        lines.append("")

    # Call the entry point
    call_args = var_decls
    if return_type == "void":
        lines.append(f"    // Invoke entry point with symbolic inputs")
        lines.append(f"    {entry_point_name}({', '.join(call_args)});")
    else:
        lines.append(f"    // Invoke entry point with symbolic inputs")
        lines.append(f"    {return_type} result = {entry_point_name}({', '.join(call_args)});")
        lines.append(f"    (void)result;")

    lines.append("")

    # If the source had its own main(), also call it
    if has_main_in_source:
        lines.append(f"    // Also invoke the original main() from the source")
        lines.append(f"    __cg_original_main(0, nullptr);")
        lines.append("")

    # Cleanup
    for c in cleanup_stmts:
        lines.append(c)

    lines.append("    return 0;")
    lines.append("}")

    return "\n".join(lines)


def detect_enclosing_namespace(source_path: Path, func_line: int) -> str:
    """Detect the namespace enclosing a function definition.

    Scans the file forward, tracking the namespace stack per line,
    and returns the namespace active at *func_line* (1-based).
    Nested namespaces are joined with ``::``.
    Returns '' if not in a namespace.
    """
    if not source_path.exists():
        return ""
    lines = source_path.read_text().split('\n')
    ns_stack: list[tuple[str, int]] = []
    depth = 0
    for i, raw in enumerate(lines):
        line = i + 1
        if line > func_line:
            break
        # Check for namespace declaration before counting braces on this line
        m = re.match(r'^\s*namespace\s+(\w+)\s*', raw)
        if m:
            remainder = raw[m.end():].strip()
            if remainder.startswith('{') or remainder == '':
                ns_stack.append((m.group(1), depth))
        depth += raw.count('{') - raw.count('}')
        # Pop namespaces whose scope has been exited
        while ns_stack and depth <= ns_stack[-1][1]:
            ns_stack.pop()
    return '::'.join(name for name, _ in ns_stack)


def find_function_at_line(source_path: Path, line: int) -> Optional[tuple[str, list[ParamInfo], str]]:
    """Find the function enclosing a given source line.

    Scans the source file for function definitions and determines which
    one contains *line* by tracking brace depth.

    Returns (func_name, params, return_type) or None.
    """
    if not source_path.exists():
        return None

    code = source_path.read_text()
    lines = code.split('\n')
    if line < 1 or line > len(lines):
        return None

    target_idx = line - 1

    # Collect all function definitions before or at the target line.
    func_pattern = re.compile(
        r'(?:^|\n)([\w:<>,\s\*&]+)\s+([\w:~]+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:\{|override|final)',
        re.DOTALL
    )

    candidates: list[tuple[int, str, list[ParamInfo], str]] = []

    # Search from the beginning of the file up to the target line
    # (plus a little buffer for multi-line signatures)
    search_end = min(len(lines), target_idx + 1)
    context = '\n'.join(lines[:search_end])

    for m in func_pattern.finditer(context):
        func_name = m.group(2).strip()
        if func_name in ("if", "while", "for", "switch", "catch", "else"):
            continue

        # Approximate start line from match position
        start_line = context[:m.start()].count('\n')  # 0-based

        ret_type = m.group(1).strip()
        param_text = m.group(3).strip()

        params = []
        for p in split_params(param_text):
            p = p.strip()
            if not p or p == "void":
                continue
            p_clean = re.sub(r"\s*=\s*[^,]+$", "", p).strip()
            parts = p_clean.split()
            if len(parts) >= 2:
                ptype = " ".join(parts[:-1])
                pname = parts[-1].lstrip("*&")
            elif len(parts) == 1:
                ptype = parts[0]
                pname = f"arg_{len(params)}"
            pname = pname.rstrip(",;")
            if pname:
                params.append(ParamInfo(pname, ptype, p_clean))

        candidates.append((start_line, func_name, params, ret_type))

    # Find which candidate's body contains the target line by tracking
    # brace depth from the function start.
    for func_start, func_name, params, ret_type in reversed(candidates):
        # Scan from function start forward, counting braces
        brace_depth = 0
        started = False
        for i in range(func_start, len(lines)):
            stripped = lines[i].strip()
            if not started:
                # Look for opening brace of function body
                if '{' in stripped:
                    started = True
                    brace_depth = stripped.count('{') - stripped.count('}')
                    if brace_depth < 0:
                        brace_depth = 0
                continue

            if brace_depth <= 0 and i > func_start:
                # Function body ended
                break

            if i >= target_idx:
                return (func_name, params, ret_type)

            brace_depth += stripped.count('{') - stripped.count('}')

        # If we didn't return above, the function ends before target

    return None
