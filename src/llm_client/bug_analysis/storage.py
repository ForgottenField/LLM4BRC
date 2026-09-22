"""Result persistence helpers for bug report analysis."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from llm_client.bug_analysis.schemas import (
    AnalysisResult,
    BatchAnalysisResult,
)

_SUPPORTED_FMTS = ("json", "yaml")


def _format_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def save_analysis(
    result: AnalysisResult,
    output_dir: str | Path,
    fmt: str = "json",
) -> str:
    """Save an *AnalysisResult* to disk.

    Parameters
    ----------
    result:
        The analysis result to persist.
    output_dir:
        Directory to write into (created if it does not exist).
    fmt:
        Output format: ``"json"`` (default) or ``"yaml"``.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    if fmt not in _SUPPORTED_FMTS:
        raise ValueError(f"Unsupported format '{fmt}'; choose from {_SUPPORTED_FMTS}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = _format_timestamp()
    fname = out / f"analysis_{timestamp}_{result.metadata.input_format.value}.{fmt}"

    _write(result, fname, fmt)  # type: ignore[arg-type]
    return str(fname.resolve())


def save_batch(
    result: BatchAnalysisResult,
    output_dir: str | Path,
    fmt: str = "json",
) -> str:
    """Save a *BatchAnalysisResult* to disk.

    Also creates individual per-report files inside a ``batch_<id>/``
    subdirectory.

    Parameters
    ----------
    result:
        The batch analysis result.
    output_dir:
        Directory to write into.
    fmt:
        Output format: ``"json"`` (default) or ``"yaml"``.

    Returns
    -------
    str
        Absolute path of the main batch file.
    """
    if fmt not in _SUPPORTED_FMTS:
        raise ValueError(f"Unsupported format '{fmt}'; choose from {_SUPPORTED_FMTS}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batch_dir = out / f"batch_{result.batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Write individual per-report files
    for entry in result.results:
        if entry.result is not None:
            _write(
                entry.result,
                batch_dir / f"report_{entry.report_index:04d}.{fmt}",
                fmt,
            )

    # Main batch file
    fname = out / f"batch_{result.batch_id}.{fmt}"
    _write(result, fname, fmt)
    return str(fname.resolve())


def load_analysis(file_path: str | Path) -> AnalysisResult:
    """Load an *AnalysisResult* from a JSON or YAML file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".yaml":
        import yaml
        data = yaml.safe_load(raw)
        return AnalysisResult.model_validate(data)

    return AnalysisResult.model_validate_json(raw)


def load_batch(file_path: str | Path) -> BatchAnalysisResult:
    """Load a *BatchAnalysisResult* from a JSON or YAML file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".yaml":
        import yaml
        data = yaml.safe_load(raw)
        return BatchAnalysisResult.model_validate(data)

    return BatchAnalysisResult.model_validate_json(raw)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _write(data: object, path: Path, fmt: str) -> None:
    if fmt == "json":
        path.write_text(
            data.model_dump_json(indent=2)  # type: ignore[union-attr]
        )
    else:
        import yaml
        path.write_text(
            yaml.dump(
                data.model_dump(mode="python"),  # type: ignore[union-attr]
                default_flow_style=False,
                sort_keys=False,
            )
        )
