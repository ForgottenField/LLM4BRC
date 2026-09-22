"""Clang CFG dump invocation, text parsing, and path enumeration.

Provides:
- ``build_cfg_command()`` — build a clang command that dumps CFG to stderr
- ``run_cfg_dump()`` — invoke clang and capture raw CFG text
- ``parse_cfg_text()`` — state-machine parser for ``debug.DumpCFG`` text output
- ``enumerate_paths()`` — DFS-based enumeration of all paths through a CFG

CFG dump format (from ``clang -cc1 -analyze -analyzer-checker=debug.DumpCFG``)::

    int test(int x)
     [B1]
       1: x
       2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
       3: 0
       4: [B1.3] (ImplicitCastExpr, LValueToRValue, int)
       5: [B1.2] > [B1.4]
       T: [B1.5] (ImplicitCastExpr, IntegralToBoolean, int)
       Preds (1): B2
       Succs (2): B3 B4

     [B2]
       Preds (1): B0
       Succs (1): B1
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.cfg_models import (
    BranchDecision,
    CFGBlock,
    CFGFunction,
    CFGPathVariant,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clang command building
# ---------------------------------------------------------------------------

# Standard include paths for the CFG dump invocation
_CFG_INCLUDE_PATHS = [
    "-I.",
    "-Ibuild",
    "-I/usr/include",
    "-I/usr/include/c++/11",
    "-I/usr/include/x86_64-linux-gnu/c++/11",
    "-I/usr/include/c++/11/backward",
    "-I/usr/lib/gcc/x86_64-linux-gnu/11/include",
    "-I/usr/local/include",
    "-I/usr/include/x86_64-linux-gnu",
    "-I/usr/include/c++/11",
    "-I/usr/src/googletest/googletest/include",
    "-I/usr/src/googletest/googlemock/include",
]

# Common defines needed for many projects
_CFG_COMMON_DEFINES = [
    "-DBOOST_ALL_NO_LIB",
    "-DFMT_LOCALE",
    "-DGFLAGS_IS_A_DLL=0",
    "-D_GNU_SOURCE",
    "-D_REENTRANT",
    "-DFOLLY_MOBILE=0",
    "-DFMT_SHARED",
]

# Project-specific include/config overrides
_CFG_PROJECT_CONFIGS: dict[str, dict] = {
    "aria2": {
        "include_paths": ["src", "."],
        "defines": [
            "HAVE_CONFIG_H",
            "WSLAY_VERSION=\"1.1.0\"",
        ],
        "force_includes": [
            "sys/socket.h",
            "netinet/in.h",
            "net/if.h",
        ],
    },
    "folly": {
        "include_paths": [
            ".",
            "folly",
            "folly/gen",
            "/tmp/folly_cfg",
            "_vendor/include",
        ],
        "defines": [
            "FOLLY_HAVE_INT128_T=1",
            "FOLLY_HAVE_MEMCHR=1",
            "FOLLY_HAVE_MEMRCHR=1",
            "FOLLY_HAVE_WCHAR_SUPPORT=1",
        ],
        "force_includes": [],
    },
    "faiss": {
        "include_paths": [
            ".",
            "/usr/lib/gcc/x86_64-linux-gnu/10/include",
        ],
        "defines": [
            "FINTEGER=int",
        ],
        "force_includes": [],
    },
}


def _detect_project_includes(source_root: str, project_name: str) -> list[str]:
    """Auto-detect include paths from the project source tree."""
    includes: list[str] = []
    root = Path(source_root)
    if project_name in ("aria2", "aria2-1.35.0", "aria2-1.36.0"):
        wslay_candidates = [
            root / "deps" / "wslay" / "lib" / "includes",
            root / "deps" / "wslay" / "lib" / "wslay",
        ]
        for c in wslay_candidates:
            if c.exists():
                includes.append(str(c))
                break
        # Also add the src and src/includes directories for aria2 headers
        src_dir = root / "src"
        if src_dir.exists():
            includes.append(str(src_dir))
        src_includes = root / "src" / "includes"
        if src_includes.exists():
            includes.append(str(src_includes))
    return includes


def build_cfg_command(
    source_file: str,
    project_name: str = "default",
    source_root: str | None = None,
) -> list[str]:
    """Build a clang command to dump CFG for *source_file*.

    Uses ``-Xclang -analyze -Xclang -analyzer-checker=debug.DumpCFG``
    through the clang driver (handles ``-resource-dir`` automatically).

    Returns a list of command-line arguments suitable for ``subprocess.run()``.
    """
    # Use clang for C files, clang++ for C++ files
    is_c_file = source_file.endswith(".c")
    compiler = "clang-14" if is_c_file else "clang++-14"
    std_flag = "-std=c17" if is_c_file else "-std=c++17"

    cmd = [
        compiler,
        std_flag,
        "-c",
        source_file,
        "-o",
        "/dev/null",
        "-Xclang",
        "-analyze",
        "-Xclang",
        "-analyzer-checker=debug.DumpCFG",
    ]

    # Add standard includes
    for inc in _CFG_INCLUDE_PATHS:
        cmd.append(inc)

    # Project-specific includes
    config = _CFG_PROJECT_CONFIGS.get(project_name, {})
    for rel_inc in config.get("include_paths", []):
        candidate = Path(source_root or ".") / rel_inc
        if candidate.exists():
            cmd.append(f"-I{candidate}")
        else:
            cmd.append(f"-I{rel_inc}")

    # Auto-detected includes from source_root or project/ directory
    used_source_root = source_root
    if not used_source_root:
        # Derive project root from the source file location
        src_path = Path(source_file)
        if "project/" in str(src_path):
            parts = str(src_path.resolve()).split("project/", 1)
            if len(parts) > 1:
                proj_part = parts[1].split("/", 1)[0]
                candidate = Path("project") / proj_part
                if candidate.exists():
                    used_source_root = str(candidate.resolve())

    if used_source_root:
        for inc in _detect_project_includes(used_source_root, project_name):
            inc_path = Path(inc)
            if inc_path.exists():
                cmd.append(f"-I{inc_path}")
            else:
                # Also try resolving relative to project/ directory
                alt = Path("project") / project_name / inc
                if alt.exists():
                    cmd.append(f"-I{alt}")

    # Defines
    cmd.extend(_CFG_COMMON_DEFINES)
    for d in config.get("defines", []):
        cmd.append(f"-D{d}")

    # Force-included system headers (for projects needing specific types)
    for inc in config.get("force_includes", []):
        cmd.append(f"-include")
        cmd.append(inc)

    return cmd


def run_cfg_dump(cmd: list[str], timeout: int = 30) -> str | None:
    """Run *cmd* and return CFG dump text from stderr.

    First tries the direct command.  If that fails with include errors,
    falls back to: preprocess the source file, strip ``#`` line markers,
    then dump CFG on the cleaned output.  This handles projects where
    certain type definitions (e.g., ``int64_t``) come from transitively
    included headers that ``-analyze`` mode struggles with.

    Returns ``None`` if no CFG content could be extracted.
    """
    # ---- Attempt 1: direct command ----
    result = _run_cfg_cmd(cmd, timeout)
    if result is not None:
        return result

    # ---- Attempt 2: preprocess + clean + re-analyze ----
    # Extract the source file from the command (last positional arg)
    source_file = None
    for i in range(len(cmd) - 1, -1, -1):
        if cmd[i].endswith((".cpp", ".cc", ".c", ".cxx", ".ii")):
            source_file = cmd[i]
            break
    if source_file and Path(source_file).exists():
        logger.info("CFG direct dump failed; trying preprocess+clean fallback ...")
        return _run_cfg_dump_via_preprocess(source_file, cmd, timeout)

    return None


def _run_cfg_cmd(cmd: list[str], timeout: int) -> str | None:
    """Run *cmd* and extract CFG text from stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("CFG dump failed (cmd=%s ...): %s", cmd[:3], e)
        return None

    stderr = result.stderr
    cfg_text = _extract_cfg_from_stderr(stderr)
    if cfg_text is not None:
        return cfg_text

    if result.returncode != 0:
        logger.warning(
            "CFG dump exited with code %d:\n%s",
            result.returncode,
            (stderr or "")[:500],
        )
        return None

    logger.warning("CFG dump returned empty output")
    return None


def _extract_cfg_from_stderr(stderr: str) -> str | None:
    """Extract CFG blocks from clang stderr output.

    Returns cleaned CFG text, or ``None`` if no ``[B`` blocks found.
    """
    if not stderr or "[B" not in stderr:
        return None
    lines = stderr.splitlines()
    cfg_lines: list[str] = []
    in_cfg = False
    for line in lines:
        if in_cfg:
            cfg_lines.append(line)
        elif line.strip() and not line.startswith("In file") and not line.startswith(" "):
            cfg_lines.append(line)  # possible function header
        elif "[B" in line:
            in_cfg = True
            if cfg_lines and not cfg_lines[-1].strip():
                cfg_lines.pop()
            cfg_lines.append(line)
    if not cfg_lines or "[B" not in "\n".join(cfg_lines):
        return None
    return "\n".join(cfg_lines)


def _run_cfg_dump_via_preprocess(
    source_file: str, original_cmd: list[str], timeout: int
) -> str | None:
    """Preprocess *source_file*, strip ``#`` markers, then dump CFG.

    Many C++ projects rely on configure-generated headers or transitive
    includes that clang's ``-analyze`` mode cannot resolve.  By preprocessing
    first, we resolve all includes, then strip the ``#`` line markers so
    the analysis pass does not try to re-read the original (unresolvable)
    headers.
    """
    import tempfile

    # Build the preprocess command using the same includes/defines
    # but replacing -c and the source with -E
    pp_cmd = [
        a for a in original_cmd
        if a not in ("-c", "-o", "/dev/null")
        and not a.startswith("-Xclang")
        and not a.startswith("-analyzer-checker")
        and not a.startswith("-analyze")
    ]
    pp_cmd[1:1] = ["-E"]  # replace -c with -E (keep -std=c++17 at index 1)

    try:
        pp_result = subprocess.run(
            pp_cmd + [source_file],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Preprocessing failed for CFG fallback: %s", e)
        return None

    if pp_result.returncode != 0:
        logger.warning("Preprocessor failed for CFG fallback:\n%s", pp_result.stderr[:300])
        return None

    # Strip # line markers from preprocessed output
    cleaned_lines = [
        line for line in pp_result.stdout.splitlines()
        if not line.startswith("#")
    ]
    cleaned = "\n".join(cleaned_lines)

    # Determine language based on source file extension
    is_c_file = source_file.endswith(".c")
    lang = "c" if is_c_file else "c++"
    compiler = "clang-14" if is_c_file else "clang++-14"
    std_flag = "-std=c17" if is_c_file else "-std=c++17"

    # Write cleaned output to temp file and run CFG dump on it
    with tempfile.NamedTemporaryFile(
        suffix=".cc", mode="w", delete=False
    ) as tmp:
        tmp.write(cleaned)
        tmp_path = tmp.name

    try:
        cfg_cmd = [
            compiler,
            std_flag,
            "-x", lang,
            "-c",
            tmp_path,
            "-o",
            "/dev/null",
            "-Xclang",
            "-analyze",
            "-Xclang",
            "-analyzer-checker=debug.DumpCFG",
        ]
        result = subprocess.run(
            cfg_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return _extract_cfg_from_stderr(result.stderr)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CFG text parser (state machine)
# ---------------------------------------------------------------------------

# Regex patterns for CFG block components
_BLOCK_HEADER_RE = re.compile(r"^\s+\[(B\d+)\s*(?:\((\w+)\))?\]\s*$")
_STATEMENT_RE = re.compile(r"^\s+(\d+):\s+(.+)$")
_TERMINATOR_RE = re.compile(r"^\s+T:\s+(.+)$")
_PREDS_RE = re.compile(r"^\s+Preds\s+\((\d+)\):\s+(.*)$")
_SUCCS_RE = re.compile(r"^\s+Succs\s+\((\d+)\):\s+(.*)$")
_FUNC_HEADER_RE = re.compile(r"^(\w[\w\s*&:<>,()]*)\s*\(.*\)\s*$")


def parse_cfg_text(raw_text: str) -> dict[str, CFGFunction]:
    """Parse raw ``debug.DumpCFG`` text into a dict of ``CFGFunction``.

    Handles multiple functions in a single dump (common when the source
    file defines several functions).  Each function's blocks are collected
    under its demangled name.

    Returns ``{}`` if nothing could be parsed.
    """
    lines = raw_text.splitlines()
    functions: dict[str, CFGFunction] = {}
    current_func: str | None = None
    current_blocks: dict[str, CFGBlock] = {}
    current_block: CFGBlock | None = None
    parsing = "IDLE"  # IDLE | FUNC_HEADER | BLOCK | DONE

    def flush_function():
        nonlocal current_func, current_blocks, current_block
        if current_func and current_blocks:
            entry = _find_entry_block(current_blocks)
            functions[current_func] = CFGFunction(
                function_name=current_func,
                blocks=current_blocks,
                entry_block_id=entry,
                raw_dump=raw_text,
            )
        current_func = None
        current_blocks = {}
        current_block = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Skip lines that are purely decorative (dashes, separators, tool name)
        if stripped in ("---", "---...", "---...---") or stripped.startswith(
            "ANALYZE"
        ):
            continue

        # Detect function header: a non-blank line with ( ) that has a [B
        # block header within the next few lines (looking past blank lines).
        # Must NOT start with CFG keywords (Succs, Preds, T:, case, etc.)
        if (
            "(" in stripped and ")" in stripped
            and not stripped.startswith("[B")
            and not stripped.startswith("Succs")
            and not stripped.startswith("Preds")
            and not stripped.startswith("T:")
            and not stripped.startswith("case")
            and not stripped.startswith("default")
            and not stripped[0].isdigit()
        ):
            is_func_header = False
            for j in range(1, min(4, len(lines) - idx)):
                peek = lines[idx + j].strip()
                if peek.startswith("[B"):
                    is_func_header = True
                    break
                if peek and not peek.startswith("#"):
                    break
            if is_func_header:
                if current_func is not None:
                    flush_function()
                current_func = stripped
                parsing = "FUNC_HEADER"
                continue

        # Block header:  [B<N>] or [B<N> (LABEL)]
        m = _BLOCK_HEADER_RE.match(line)
        if m:
            block_id = m.group(1)
            label = m.group(2) or ""
            current_block = CFGBlock(
                block_id=block_id,
                label=label,
                raw_text=line,
            )
            current_blocks[block_id] = current_block
            parsing = "BLOCK"
            continue

        if current_block is None:
            continue

        # Statement:  N: <expression>
        m = _STATEMENT_RE.match(line)
        if m:
            current_block.statements.append(m.group(2))
            continue

        # Terminator:  T: <expression>
        m = _TERMINATOR_RE.match(line)
        if m:
            current_block.terminator = m.group(1)
            continue

        # Predecessors:  Preds (N): Bx By ...
        m = _PREDS_RE.match(line)
        if m:
            count = int(m.group(1))
            preds = m.group(2).strip().split() if count > 0 else []
            current_block.predecessors = preds
            continue

        # Successors:  Succs (N): Bx By ...
        m = _SUCCS_RE.match(line)
        if m:
            count = int(m.group(1))
            succs = m.group(2).strip().split() if count > 0 else []
            current_block.successors = succs
            continue

    # Flush the last function
    flush_function()

    return functions


def _find_entry_block(blocks: dict[str, CFGBlock]) -> str:
    """Find the entry block (first executable block after ENTRY).

    The ENTRY block (B<N> with label 'ENTRY') has one successor which
    is the first executable block.  If no ENTRY block, return the block
    with the most predecessors (heuristic).
    """
    # Look for an ENTRY block
    for bid, blk in blocks.items():
        if blk.label == "ENTRY" and blk.successors:
            return blk.successors[0]

    # No ENTRY marker: return block with most predecessors (heuristic)
    if not blocks:
        return ""
    # Find block with most predecessors (typically B1)
    best = max(blocks.items(), key=lambda kv: len(kv[1].predecessors))
    return best[0]


# ---------------------------------------------------------------------------
# Path enumeration
# ---------------------------------------------------------------------------


def enumerate_paths(
    cfg: CFGFunction,
    max_variants: int = 16,
    max_visits_per_block: int = 3,
) -> list[CFGPathVariant]:
    """Enumerate all execution paths through a function's CFG.

    Uses DFS with loop detection (counts visits per block per path;
    if a block is visited *max_visits_per_block* times, the path
    is terminated to avoid infinite loops).

    Args:
        cfg: Parsed CFG of the function.
        max_variants: Cap on the number of paths to enumerate.
        max_visits_per_block: Loop detection threshold.

    Returns:
        List of ``CFGPathVariant``, each representing a distinct path
        through the CFG.
    """
    if not cfg.entry_block_id or cfg.entry_block_id not in cfg.blocks:
        return []

    variants: list[CFGPathVariant] = []

    # Stack entries: (block_id, block_sequence, branch_decisions, visit_counts)
    block: dict[str, int] = {}
    stack = [(cfg.entry_block_id, [cfg.entry_block_id], [], block)]

    while stack and len(variants) < max_variants:
        bid, seq, decisions, counts = stack.pop()
        block_obj = cfg.blocks.get(bid)
        if not block_obj:
            continue

        # Check if this is the EXIT block
        if bid == cfg.exit_block_id or block_obj.label == "EXIT":
            variant = CFGPathVariant(
                variant_id=len(variants),
                branch_decisions=decisions,
                block_sequence=seq,
            )
            # First path = the original (CSA's approximate path)
            if len(variants) == 0:
                variant.is_original_path = True
            variants.append(variant)
            continue

        succs = block_obj.successors
        if not succs:
            # Dead end — still record what we have
            variant = CFGPathVariant(
                variant_id=len(variants),
                branch_decisions=decisions,
                block_sequence=seq,
            )
            variants.append(variant)
            continue

        if len(succs) == 1:
            # Linear flow — follow the single successor
            new_c = _new_counts(counts, succs[0])
            if new_c is not None:
                stack.append(
                    (succs[0], seq + [succs[0]], list(decisions), new_c)
                )
        else:
            # Branch — push branches in reverse so true (first succ) pops first
            for i in range(len(succs) - 1, -1, -1):
                succ = succs[i]
                is_true = i == 0

                branch_dec = BranchDecision(
                    block_id=bid,
                    terminator_text=block_obj.terminator or "",
                    taken_successor=succ,
                    alternative_successor=succs[1 - i],
                    taken_is_true_branch=is_true,
                )

                new_counts = _new_counts(counts, succ)
                if new_counts is not None:
                    stack.append(
                        (succ, seq + [succ], decisions + [branch_dec], new_counts)
                    )

    return variants


def _new_counts(
    counts: dict[str, int], next_bid: str, max_visits: int = 3
) -> dict[str, int] | None:
    """Return updated visit counts for *next_bid*, or ``None`` if loop detected."""
    new = dict(counts)
    new[next_bid] = new.get(next_bid, 0) + 1
    if new[next_bid] > max_visits:
        return None
    return new
