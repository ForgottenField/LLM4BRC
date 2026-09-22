"""Static compilation checker for POC test harnesses.

Tries to compile generated C++ code and returns structured error
information that can be fed back to an LLM for iterative fixing.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default compilation flags for folly-based POCs
_DEFAULT_FLAGS = [
    "clang++-14",
    "-std=c++17",
    "-fsyntax-only",  # only check syntax, don't generate code
]

# Include paths for folly and dependencies
# These are relative to the project_root (folly checkout root)
_INCLUDE_PATHS = [
    "-I.",
    "-Ibuild",
    "-I/usr/include",
    "-I/usr/src/googletest/googletest/include",
    "-I/usr/src/googletest/googlemock/include",
]

# Common defines needed for folly
_COMMON_DEFINES = [
    "-DBOOST_ALL_NO_LIB",
    "-DFMT_LOCALE",
    "-DGFLAGS_IS_A_DLL=0",
    "-D_GNU_SOURCE",
    "-D_REENTRANT",
    "-DFOLLY_XLOG_STRIP_PREFIXES=\"\"",
    "-DFOLLY_MOBILE=0",
    "-DFMT_SHARED",
]


class CompileError:
    """A single compilation error with structured information."""

    def __init__(self, file: str, line: int, column: int, message: str, code: str) -> None:
        self.file = file
        self.line = line
        self.column = column
        self.message = message
        self.code = code

    def __repr__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.message}"

    def format_for_llm(self, index: int) -> str:
        """Format this error for LLM consumption with source context."""
        return (
            f"Error #{index}:\n"
            f"  Location: {self.file}:{self.line}:{self.column}\n"
            f"  Message: {self.message}\n"
            f"  Problematic code: {self.code}\n"
        )


class CompileResult:
    """Result of a compilation attempt."""

    def __init__(
        self,
        success: bool,
        errors: list[CompileError] | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.success = success
        self.errors = errors or []
        self.stdout = stdout
        self.stderr = stderr

    def format_errors_for_llm(self, max_errors: int = 5) -> str:
        """Format the first N errors for inclusion in an LLM fix prompt."""
        if self.success:
            return "No errors."

        lines = [f"Found {len(self.errors)} compilation error(s):"]
        for i, err in enumerate(self.errors[:max_errors]):
            lines.append("")
            lines.append(err.format_for_llm(i + 1))
        if len(self.errors) > max_errors:
            lines.append(f"... and {len(self.errors) - max_errors} more error(s)")
        return "\n".join(lines)


# Project-specific include paths and defines
_PROJECT_CONFIGS: dict[str, dict] = {
    "aria2": {
        "include_paths": [],
        "defines": [],
        "extra_flags": [],
    },
    "folly": {
        "include_paths": [
            ".",
            "build",
        ],
        "defines": [
            "BOOST_ALL_NO_LIB",
            "FMT_LOCALE",
            "GFLAGS_IS_A_DLL=0",
            "_GNU_SOURCE",
            "_REENTRANT",
            "FOLLY_XLOG_STRIP_PREFIXES=\"\"",
            "FOLLY_MOBILE=0",
            "FMT_SHARED",
        ],
        "extra_flags": [],
    },
    "faiss": {
        "include_paths": [
            ".",
            "/usr/lib/gcc/x86_64-linux-gnu/10/include",
        ],
        "defines": [
            "FINTEGER=int",
        ],
        "extra_flags": [],
    },
}


def _detect_project_includes(source_root: str, project_name: str) -> list[str]:
    """Auto-detect include paths from the project source tree.

    For aria2/wslay: finds the wslay include directory under deps/.
    """
    includes: list[str] = []
    root = Path(source_root)

    if project_name in ("aria2", "aria2-1.35.0"):
        # Auto-detect wslay include path
        wslay_candidates = [
            root / "deps" / "wslay" / "lib" / "includes",
            root / "deps" / "wslay" / "includes",
        ]
        for p in wslay_candidates:
            if (p / "wslay" / "wslay.h").exists():
                includes.append(str(p))
                break
        # Also look in project/ directory for local copies
        local_wslay = Path("project") / "aria2" / "deps" / "wslay" / "lib" / "includes"
        if local_wslay.exists() and str(local_wslay) not in includes:
            includes.append(str(local_wslay.resolve()))

    return includes


def _detect_project_defines(source_root: str, project_name: str) -> list[str]:
    """Auto-detect preprocessor defines needed for a project."""
    defines: list[str] = []

    if project_name in ("aria2", "aria2-1.35.0"):
        defines.append("WSLAY_VERSION=\"1.1.0\"")

    return defines


class CompileChecker:
    """Check C++ source code compiles correctly with folly-friendly flags.

    Supports project-specific include paths, defines, and flags.
    Automatically configures itself for known projects (aria2, folly).

    Usage::

        checker = CompileChecker(project_name="aria2", source_root="/path/to/aria2")
        result = checker.check(code)
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        project_name: str = "",
        source_root: str | Path | None = None,
        extra_flags: list[str] | None = None,
        extra_includes: list[str] | None = None,
        extra_defines: list[str] | None = None,
    ) -> None:
        self._project_root = str(Path(project_root).resolve()) if project_root else os.getcwd()
        self._project_name = project_name or ""
        self._source_root = str(Path(source_root).resolve()) if source_root else ""
        self._extra_flags = extra_flags or []
        self._extra_includes = extra_includes or []
        self._extra_defines = extra_defines or []

    def check(self, code: str, source_name: str = "poc.cpp") -> CompileResult:
        """Attempt to compile a C++ source snippet.

        Args:
            code: C++ source code to check.
            source_name: Name for the temporary source file.

        Returns:
            CompileResult with success/failure and structured errors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / source_name
            src_path.write_text(code)

            cmd = self._build_command(str(src_path))
            logger.debug("Compile command: %s", " ".join(cmd))

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self._project_root,
                )
            except subprocess.TimeoutExpired:
                return CompileResult(
                    success=False,
                    errors=[CompileError(source_name, 0, 0, "Compilation timed out", "")],
                    stderr="timeout",
                )

            if result.returncode == 0:
                return CompileResult(success=True, stdout=result.stdout, stderr=result.stderr)

            errors = self._parse_errors(result.stderr, source_name)
            return CompileResult(
                success=False,
                errors=errors,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    def _build_command(self, source_path: str) -> list[str]:
        """Build the full compilation command."""
        cmd = list(_DEFAULT_FLAGS)
        cmd.extend(self._extra_flags)

        # Standard system includes
        cmd.extend(_INCLUDE_PATHS)

        # Project-specific includes from config
        config = _PROJECT_CONFIGS.get(self._project_name, {})
        for rel_inc in config.get("include_paths", []):
            candidate = Path(self._project_root) / rel_inc
            if candidate.exists():
                cmd.append(f"-I{candidate}")
            else:
                cmd.append(f"-I{rel_inc}")

        # Extra user-provided includes
        cmd.extend(f"-I{p}" for p in self._extra_includes)

        # Auto-detected includes (source tree scanning)
        if self._source_root:
            for inc in _detect_project_includes(self._source_root, self._project_name):
                cmd.append(f"-I{inc}")

        # Standard system defines
        cmd.extend(_COMMON_DEFINES)
        # Project-specific defines from config
        for d in config.get("defines", []):
            cmd.append(f"-D{d}")
        # Auto-detected defines
        if self._source_root:
            for d in _detect_project_defines(self._source_root, self._project_name):
                cmd.append(f"-D{d}")
        # Extra user-provided defines
        cmd.extend(f"-D{d}" for d in self._extra_defines)

        cmd.append(source_path)
        return cmd

    @staticmethod
    def _parse_errors(stderr: str, source_name: str) -> list[CompileError]:
        """Parse clang/gcc error output into structured CompileError objects.

        Handles the standard clang error format:
            file:line:column: error: message
            problematic code line
            ^~~~~~~~~~~~~~~~
        """
        errors: list[CompileError] = []
        lines = stderr.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]
            # Match clang error format: file:line:column: error/warning: message
            m = re.match(
                r"^(.*?):(\d+):(\d+):\s+(?:error|fatal error|warning):\s+(.+)$",
                line,
            )
            if m:
                err_file = m.group(1)
                err_line = int(m.group(2))
                err_col = int(m.group(3))
                err_msg = m.group(4).strip()

                # Get the problematic code line (next line if it's part of the error)
                code_line = ""
                if i + 1 < len(lines) and not lines[i + 1].startswith(err_file):
                    code_candidate = lines[i + 1].strip()
                    if code_candidate and not code_candidate.startswith("^"):
                        code_line = code_candidate

                # Only report errors from our source file
                if source_name in err_file or not err_file.endswith((".h", ".cpp")):
                    errors.append(CompileError(
                        file=source_name if source_name in err_file else err_file,
                        line=err_line,
                        column=err_col,
                        message=err_msg,
                        code=code_line,
                    ))

            i += 1

        # Deduplicate (clang sometimes reports the same error twice)
        seen: set[str] = set()
        unique_errors: list[CompileError] = []
        for err in errors:
            key = f"{err.line}:{err.column}:{err.message}"
            if key not in seen:
                seen.add(key)
                unique_errors.append(err)

        return unique_errors
