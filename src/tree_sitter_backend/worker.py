"""Crash-isolated execution for the native Tree-sitter backend."""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path
from typing import Iterable

from tree_sitter_backend.backend import TreeSitterBackend
from tree_sitter_backend.models import CodeIndex


def build_index_isolated(
    source_files: Iterable[str | Path],
    source_root: str | Path,
    *,
    compile_commands: str | Path | None = None,
    timeout: float = 30.0,
) -> tuple[CodeIndex | None, str]:
    """Build an index in a child process and contain native parser failures."""
    # A fresh fork is safe here: the worker is started before LLM networking or
    # other background threads, and it avoids re-importing the caller's __main__
    # on POSIX.  Windows has no fork and uses spawn.
    method = "fork" if sys.platform != "win32" else "spawn"
    context = multiprocessing.get_context(method)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(
            [str(path) for path in source_files],
            str(source_root),
            str(compile_commands) if compile_commands is not None else None,
            sender,
        ),
        daemon=True,
    )
    process.start()
    sender.close()
    payload = None
    readable = receiver.poll(timeout)
    if readable:
        try:
            payload = receiver.recv()
        except (EOFError, OSError):
            payload = None
    process.join(2)
    if process.is_alive():
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(1)
        receiver.close()
        return None, f"Tree-sitter worker timed out after {timeout:.1f}s"
    receiver.close()

    if process.exitcode != 0:
        return None, f"Tree-sitter worker exited with code {process.exitcode}"
    if not payload:
        return None, "Tree-sitter worker returned no result"
    status, value = payload
    if status != "ok":
        return None, str(value)
    return value, ""


def _worker(
    files: list[str],
    root: str,
    compile_commands: str | None,
    sender,
) -> None:
    try:
        index = TreeSitterBackend().build_index(
            files,
            root,
            compile_commands=compile_commands,
        )
    except BaseException as exc:  # noqa: BLE001 - report child failures to parent
        sender.send(("error", f"{type(exc).__name__}: {exc}"))
    else:
        sender.send(("ok", index))
    finally:
        sender.close()
