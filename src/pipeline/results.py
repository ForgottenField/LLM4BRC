"""Run-scoped result directory layout and console capture."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from pipeline.io import write_json_atomic
from pipeline.layout import REPO_ROOT


@dataclass(frozen=True)
class ResultsLayout:
    """Filesystem layout for one invocation."""

    run_id: str
    run_dir: Path
    project_dir: Path
    reports_dir: Path
    pocs_dir: Path
    log_path: Path
    manifest_path: Path
    summary_path: Path


def create_results_layout(
    results_root: str | Path | None,
    project_name: str,
) -> ResultsLayout:
    """Create a unique ``results/<run-time>/<project>/`` tree."""
    root = Path(results_root).resolve() if results_root else REPO_ROOT / "results"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{stamp}_{suffix:02d}"
        suffix += 1

    project_dir = run_dir / _safe_component(project_name or "unknown")
    layout = ResultsLayout(
        run_id=run_dir.name,
        run_dir=run_dir,
        project_dir=project_dir,
        reports_dir=project_dir / "reports",
        pocs_dir=project_dir / "pocs",
        log_path=project_dir / "run.log",
        manifest_path=project_dir / "run_manifest.json",
        summary_path=project_dir / "batch_summary.json",
    )
    for directory in (layout.reports_dir, layout.pocs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def configure_results_args(args, layout: ResultsLayout) -> None:
    """Attach implicit output destinations without overriding explicit paths."""
    if not getattr(args, "output", None):
        args.output = str(layout.reports_dir)
        args.output_is_dir = True
    if not getattr(args, "batch_summary", None):
        args.batch_summary = str(layout.summary_path)
    args.poc_output_dir = str(layout.pocs_dir)
    args.results_run_dir = str(layout.run_dir)
    args.results_project_dir = str(layout.project_dir)


def write_manifest(
    layout: ResultsLayout,
    payload: dict,
) -> None:
    """Write or update the run manifest atomically."""
    write_json_atomic(layout.manifest_path, payload)


class _Tee:
    """Mirror writes to the console and a run log."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def capture_console(log_path: Path):
    """Return a context manager-like object that tees stdout to *log_path*."""
    return _ConsoleCapture(log_path)


class _ConsoleCapture:
    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._stream: TextIO | None = None
        self._stdout = None

    def __enter__(self) -> "_ConsoleCapture":
        self._stream = self._log_path.open("a", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._stream)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stream is not None:
            self._stream.close()


def _safe_component(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return text.strip("._-") or "unknown"
