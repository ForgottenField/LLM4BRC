"""Project-level CSA scanning — batch analysis of multi-file C/C++ projects."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from llm_client.csa_analysis.errors import SourceFileNotFoundError
from llm_client.csa_analysis.models import (
    AnalyzerMode,
    CSAConfig,
    CSAResult,
    CSAStatus,
    FileResult,
    ProjectAnalysisResult,
)
from llm_client.csa_analysis.parser import PlistParser
from llm_client.csa_analysis.runner import CSARunner

SOURCE_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".C", ".c++"}


class CompilationDatabase:
    """Parser for ``compile_commands.json`` (JSON Compilation Database).

    Provides per-file compiler flags extracted from the build system
    (CMake, Bear, etc.).
    """

    def __init__(self, json_path: str | Path) -> None:
        self._path = Path(json_path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Compilation database not found: {self._path}")
        with open(self._path) as f:
            self._entries: list[dict[str, Any]] = json.load(f)

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self._entries

    def get_flags(self, file_path: str | Path) -> list[str]:
        """Resolve the compiler flags for a source file.

        Matches by file path suffix (longest match wins) to handle
        relative path differences.
        """
        target = Path(file_path).resolve()
        best_entry: dict[str, Any] | None = None
        best_len = 0

        for entry in self._entries:
            entry_file = Path(entry.get("file", ""))
            if not entry_file.name:
                continue
            try:
                resolved = entry_file.resolve()
            except (OSError, RuntimeError):
                resolved = entry_file

            if resolved == target:
                best_entry = entry
                break

            # Match by name + directory suffix
            entry_str = str(entry_file)
            target_str = str(target)
            if target_str.endswith(entry_str) and len(entry_str) > best_len:
                best_entry = entry
                best_len = len(entry_str)

        if best_entry is None:
            return []

        return self._parse_command(best_entry)

    def all_source_files(self) -> list[dict[str, Any]]:
        """Return all source file entries with their flags."""
        return [
            {
                "file": str(Path(e["file"]).resolve() if not os.path.isabs(e["file"])
                           else Path(e["file"]).resolve()),
                "flags": self._parse_command(e),
                "directory": e.get("directory", ""),
            }
            for e in self._entries
            if _is_source_file(e.get("file", ""))
        ]

    @staticmethod
    def _parse_command(entry: dict[str, Any]) -> list[str]:
        """Extract compiler flags from a compilation database entry.

        Handles both ``command`` (string) and ``arguments`` (list) formats.
        """
        if "arguments" in entry:
            args = entry["arguments"]
            return _strip_compiler_invocation(args)

        cmd = entry.get("command", "")
        if not cmd:
            return []
        import shlex
        args = shlex.split(cmd)
        return _strip_compiler_invocation(args)


def _strip_compiler_invocation(args: list[str]) -> list[str]:
    """Remove compiler name, input/output file flags, leaving only flags."""
    result: list[str] = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        # Skip compiler name (first positional)
        if i == 0 and not arg.startswith("-"):
            continue
        # Skip input file
        if not arg.startswith("-") and i > 0 and args[i - 1].startswith("-"):
            result.append(arg)
            continue
        if arg == "-o":
            skip_next = True
            continue
        if arg.startswith("-o"):
            continue
        # Skip -c, -o
        if arg in ("-c",):
            continue
        result.append(arg)
    return result


def _is_source_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in SOURCE_EXTENSIONS


def discover_source_files(
    project_dir: str | Path,
    extensions: set[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> list[str]:
    """Recursively discover C/C++ source files in a project directory.

    Args:
        project_dir: Root directory of the project.
        extensions: Set of file extensions to include
                    (default: ``{".c", ".cpp", ".cc", ".cxx"}``).
        exclude_dirs: Directory names to exclude
                      (default: ``{"build", ".git", "node_modules", ".venv"}``).

    Returns:
        Sorted list of absolute source file paths.
    """
    exts = extensions or SOURCE_EXTENSIONS
    excluded = exclude_dirs or {"build", ".git", "node_modules", ".venv", "__pycache__"}
    root = Path(project_dir).resolve()
    sources: list[str] = []
    for f in root.rglob("*"):
        if f.suffix in exts and not any(p in excluded for p in f.parts):
            sources.append(str(f.resolve()))
    return sorted(sources)


class ProjectScanner:
    """Run CSA analysis across an entire C/C++ project.

    Supports three modes:
    1. **Directory scan** — discover all source files recursively
    2. **Compilation database** — use ``compile_commands.json`` for flags
    3. **scan-build** — delegate to ``scan-build`` with a build command
    """

    def __init__(
        self,
        config: CSAConfig,
        runner: Optional[CSARunner] = None,
        parser: Optional[PlistParser] = None,
        max_workers: int = 4,
    ) -> None:
        self._config = config
        self._runner = runner or CSARunner(config)
        self._parser = parser or PlistParser()
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_directory(
        self,
        project_dir: str | Path,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> ProjectAnalysisResult:
        """Scan all source files in a project directory.

        Each file is analyzed independently with the shared ``CSAConfig``
        compiler flags.

        Args:
            project_dir: Root directory of the project.
            extensions: File extensions to include.
            exclude_dirs: Directory names to exclude.

        Returns:
            Aggregated ``ProjectAnalysisResult``.
        """
        sources = discover_source_files(project_dir, extensions, exclude_dirs)
        if not sources:
            return ProjectAnalysisResult(
                project_dir=str(Path(project_dir).resolve()),
                total_files=0,
            )

        return self._analyze_files(
            project_dir=str(Path(project_dir).resolve()),
            files=sources,
        )

    def scan_from_compile_commands(
        self,
        json_path: str | Path,
    ) -> ProjectAnalysisResult:
        """Scan using a ``compile_commands.json`` compilation database.

        Each source file gets its own compiler flags from the database.

        Args:
            json_path: Path to ``compile_commands.json``.

        Returns:
            Aggregated ``ProjectAnalysisResult``.
        """
        db = CompilationDatabase(json_path)
        file_entries = db.all_source_files()

        project_dir = str(Path(json_path).resolve().parent)
        if not file_entries:
            return ProjectAnalysisResult(
                project_dir=project_dir,
                total_files=0,
            )

        # Build per-file configs with custom flags
        def analyze_entry(entry: dict[str, Any]) -> FileResult:
            file_config = self._config.model_copy(deep=True)
            file_config.compiler_flags = entry["flags"]
            file_runner = CSARunner(file_config)
            return self._analyze_single_file(
                file_path=entry["file"],
                runner=file_runner,
            )

        return self._run_batch(
            project_dir=project_dir,
            entries=file_entries,
            analyze_fn=analyze_entry,
            key_fn=lambda e: e["file"],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze_files(
        self,
        project_dir: str,
        files: list[str],
    ) -> ProjectAnalysisResult:
        def analyze(path: str) -> FileResult:
            return self._analyze_single_file(
                file_path=path,
                runner=self._runner,
            )

        return self._run_batch(
            project_dir=project_dir,
            entries=files,
            analyze_fn=analyze,
            key_fn=lambda f: f,
        )

    def _analyze_single_file(
        self,
        file_path: str,
        runner: CSARunner,
    ) -> FileResult:
        """Analyze a single file and return a FileResult."""
        start = time.monotonic()
        try:
            result: CSAResult = runner.run(file_path)
            elapsed = int((time.monotonic() - start) * 1000)

            if result.status == CSAStatus.ERROR or not result.plist_path:
                return FileResult(
                    file_path=file_path,
                    status=CSAStatus.ERROR,
                    duration_ms=elapsed,
                    error=result.stderr[:500] if result.stderr else "Unknown error",
                )

            bug_paths = self._parser.parse(result.plist_path)
            return FileResult(
                file_path=file_path,
                status=CSAStatus.SUCCESS,
                duration_ms=elapsed,
                bug_paths=bug_paths,
            )

        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return FileResult(
                file_path=file_path,
                status=CSAStatus.ERROR,
                duration_ms=elapsed,
                error=str(exc)[:500],
            )

    def _run_batch(
        self,
        project_dir: str,
        entries: list[Any],
        analyze_fn,
        key_fn,
    ) -> ProjectAnalysisResult:
        """Run batch analysis with a thread pool."""
        file_results: list[FileResult] = []
        total_duration = 0
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {
                pool.submit(analyze_fn, entry): key_fn(entry)
                for entry in entries
            }
            for future in as_completed(future_map):
                fpath = future_map[future]
                try:
                    fr = future.result()
                except Exception as exc:
                    fr = FileResult(
                        file_path=fpath,
                        status=CSAStatus.ERROR,
                        error=str(exc)[:500],
                    )
                file_results.append(fr)
                total_duration += fr.duration_ms
                if fr.status == CSAStatus.SUCCESS:
                    successful += 1
                else:
                    failed += 1

        all_bug_paths = [
            bp
            for fr in file_results
            for bp in fr.bug_paths
        ]

        return ProjectAnalysisResult(
            project_dir=str(Path(project_dir).resolve()),
            total_files=len(entries),
            successful_files=successful,
            failed_files=failed,
            total_duration_ms=total_duration,
            file_results=file_results,
            all_bug_paths=all_bug_paths,
        )
