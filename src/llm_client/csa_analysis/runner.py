"""CSA invocation — subprocess wrapper around clang --analyze / scan-build."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from llm_client.csa_analysis.errors import (
    CSANotFoundError,
    CSATimeoutError,
    SourceFileNotFoundError,
)
from llm_client.csa_analysis.models import (
    AnalyzerMode,
    CSAConfig,
    CSAResult,
    CSAStatus,
    ReachabilityTarget,
)

CLANG_BIN = "clang++-14"
SCAN_BUILD_BIN = "scan-build"
PLIST_FILENAME = "csa_output.plist"

# Path to the built reachability checker plugin .so
_DEFAULT_PLUGIN_PATH = Path(__file__).resolve().parent / "checker" / "build" / "libReachabilityPlugin.so"


class CSARunner:
    """Invoke Clang Static Analyzer on target source files.

    Args:
        config: Configuration for CSA invocation.
    """

    def __init__(self, config: CSAConfig) -> None:
        self._config = config

    @property
    def config(self) -> CSAConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, file_path: str) -> CSAResult:
        """Run CSA analysis on a single source file via ``clang --analyze``.

        Args:
            file_path: Path to the C/C++ source file to analyze.

        Returns:
            CSAResult with status, plist_path, command, stdout/stderr, duration.

        Raises:
            CSANotFoundError: If clang is not installed or not in PATH.
            SourceFileNotFoundError: If *file_path* does not exist.
            CSATimeoutError: If analysis exceeds configured timeout.
        """
        _check_clang()
        self._resolve_source(file_path)

        output_dir = self._resolve_output_dir()
        plist_path = os.path.join(output_dir, PLIST_FILENAME)
        cmd = self._build_clang_command(file_path, plist_path)

        result = self._execute(cmd)
        if result.status == CSAStatus.SUCCESS:
            if os.path.exists(plist_path):
                result.plist_path = plist_path
        return result

    def run_reachability(
        self,
        file_path: str,
        target: ReachabilityTarget,
        plugin_path: str | Path | None = None,
    ) -> CSAResult:
        """Run CSA with the custom reachability checker plugin.

        Instead of running bug-detection checkers, this loads a custom
        ``ReachabilityChecker`` plugin that traces *all* feasible execution
        paths reaching *target*.

        Args:
            file_path: Path to the C/C++ source file to analyze.
            target: Target location (file + line) to trace paths to.
            plugin_path: Path to ``libReachabilityPlugin.so``.
                         Defaults to the built artifact in the ``checker/`` tree.

        Returns:
            CSAResult with plist_path pointing to the generated plist.
            The plist contains one diagnostic per execution path reaching the
            target, with full branch-condition annotation.

        Raises:
            CSANotFoundError: If clang or the plugin is not available.
            SourceFileNotFoundError: If *file_path* does not exist.
            CSATimeoutError: If analysis exceeds configured timeout.
        """
        _check_clang()
        self._resolve_source(file_path)

        plugin = Path(plugin_path or _DEFAULT_PLUGIN_PATH).resolve()
        if not plugin.exists():
            raise SourceFileNotFoundError(
                f"Reachability plugin not found at {plugin}. "
                f"Build it first: cd {plugin.parent} && bash build.sh"
            )

        output_dir = self._resolve_output_dir()
        plist_path = os.path.join(output_dir, PLIST_FILENAME)

        cmd = [CLANG_BIN, "--analyze", "-w"]
        cmd.extend(self._config.compiler_flags)

        # Load the custom plugin
        cmd.extend(["-Xclang", "-load", "-Xclang", str(plugin)])

        # Enable only our checker (replaces all built-in checkers)
        cmd.extend([
            "-Xclang", "-analyzer-checker=reachability.ReachabilityChecker",
        ])

        # Configure the target location
        target_file = target.file_path
        cmd.extend([
            "-Xclang", "-analyzer-config",
            "-Xclang", f"reachability.ReachabilityChecker:TargetFile={target_file}",
        ])
        cmd.extend([
            "-Xclang", "-analyzer-config",
            "-Xclang", f"reachability.ReachabilityChecker:TargetLine={target.line_number}",
        ])

        # Bypass built-in checker bundles so only our checker runs
        cmd.extend([
            "-Xclang", "-analyzer-disable-checker=core",
            "-Xclang", "-analyzer-disable-checker=unix",
            "-Xclang", "-analyzer-disable-checker=deadcode",
            "-Xclang", "-analyzer-disable-checker=security",
            "-Xclang", "-analyzer-disable-checker=nullability",
            "-Xclang", "-analyzer-disable-checker=apiModeling",
            "-Xclang", "-analyzer-disable-checker=cplusplus",
            "-Xclang", "-analyzer-disable-checker=optin",
            "-Xclang", "-analyzer-disable-checker=osx",
        ])

        cmd.extend([
            "-Xclang", "-analyzer-output=plist-multi-file",
            "-o", plist_path,
            file_path,
        ])

        result = self._execute(cmd)
        if os.path.exists(plist_path):
            result.plist_path = plist_path
        # Exit code may be non-zero due to shutdown warnings; treat
        # plist existence as success.
        if result.plist_path:
            result.status = CSAStatus.SUCCESS
        return result

    def run_scan_build(self, build_command: str) -> CSAResult:
        """Run CSA analysis via scan-build with a build command.

        Args:
            build_command: Build command (e.g., ``"make"``, ``"cmake --build ."``).

        Returns:
            CSAResult with status, plist_paths, command, stdout/stderr, duration.

        Raises:
            CSANotFoundError: If scan-build is not installed.
            CSATimeoutError: If analysis exceeds configured timeout.
        """
        if not shutil.which(SCAN_BUILD_BIN):
            raise CSANotFoundError(
                f"'{SCAN_BUILD_BIN}' not found in PATH. "
                f"Install with: sudo apt install clang"
            )

        output_dir = self._resolve_output_dir()
        cmd = [
            SCAN_BUILD_BIN,
            "-o",
            output_dir,
            "--status-codes",
        ]
        cmd.extend(shlex.split(build_command))

        result = self._execute(cmd)
        if result.status == CSAStatus.SUCCESS:
            plists = sorted(Path(output_dir).rglob("*.plist"))
            result.plist_paths = [str(p) for p in plists]
        return result

    @staticmethod
    def check_available() -> bool:
        """Check if CSA (``clang --analyze``) is available in PATH."""
        return shutil.which(CLANG_BIN) is not None

    @staticmethod
    def list_checkers() -> list[str]:
        """List available CSA checkers.

        Returns:
            List of checker names.

        Raises:
            CSANotFoundError: If clang is not installed.
        """
        _check_clang()
        cmd = [
            CLANG_BIN,
            "--analyze",
            "-Xclang",
            "-analyzer-checker-help",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            lines = result.stdout.strip().splitlines()
            checkers = [line.strip() for line in lines if line.strip()]
            return checkers
        except subprocess.TimeoutExpired:
            raise CSATimeoutError("Listing CSA checkers timed out after 30s.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_clang_command(self, file_path: str, plist_path: str) -> list[str]:
        """Assemble the ``clang --analyze`` command."""
        cmd = [CLANG_BIN, "--analyze"]

        # Suppress non-diagnostic warnings for cleaner output
        cmd.append("-w")

        if self._config.language_standard:
            cmd.append(f"-std={self._config.language_standard}")

        cmd.extend(self._config.compiler_flags)

        # Enable specific checkers
        if self._config.checkers:
            for checker in self._config.checkers:
                cmd.extend(["-Xclang", f"-analyzer-checker={checker}"])

        cmd.extend([
            "-Xclang",
            "-analyzer-output=plist-multi-file",
            "-o",
            plist_path,
            file_path,
        ])
        return cmd

    def _resolve_output_dir(self) -> str:
        if self._config.output_dir:
            os.makedirs(self._config.output_dir, exist_ok=True)
            return self._config.output_dir
        return tempfile.mkdtemp(prefix="csa_analysis_")

    def _resolve_source(self, file_path: str) -> Path:
        path = Path(file_path).resolve()
        if not path.exists():
            raise SourceFileNotFoundError(
                f"Source file not found: {file_path}"
            )
        return path

    def _execute(self, cmd: list[str]) -> CSAResult:
        """Run a command and wrap the result in a CSAResult."""
        start = time.monotonic()
        command_str = " ".join(cmd)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )
            elapsed = int((time.monotonic() - start) * 1000)
            return CSAResult(
                status=CSAStatus.SUCCESS if proc.returncode == 0 else CSAStatus.ERROR,
                command=command_str,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_ms=elapsed,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.monotonic() - start) * 1000)
            raise CSATimeoutError(
                f"CSA analysis timed out after {self._config.timeout_seconds}s "
                f"(ran for {elapsed}ms)."
            )


def _check_clang() -> None:
    """Raise CSANotFoundError if clang is not available."""
    if not shutil.which(CLANG_BIN):
        raise CSANotFoundError(
            f"'{CLANG_BIN}' not found in PATH. "
            f"Install with: sudo apt install clang"
        )


def _check_scan_build() -> None:
    """Raise CSANotFoundError if scan-build is not available."""
    if not shutil.which(SCAN_BUILD_BIN):
        raise CSANotFoundError(
            f"'{SCAN_BUILD_BIN}' not found in PATH. "
            f"Install with: sudo apt install clang"
        )
