"""Compilation database loading and preprocessing for the syntax backend."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPILE_COMMANDS_NAME = "compile_commands.json"


class CompileDatabaseError(RuntimeError):
    """A compile database or one of its entries cannot be used."""


@dataclass(frozen=True)
class SourcePosition:
    file: str
    line: int
    column: int = 0


@dataclass(frozen=True)
class SourceMap:
    """Map byte offsets in preprocessed output back to original source lines."""

    line_starts: tuple[int, ...]
    origins: tuple[SourcePosition | None, ...]

    def locate(self, byte_offset: int) -> SourcePosition | None:
        if not self.line_starts:
            return None
        clamped = max(0, byte_offset)
        line_index = bisect_right(self.line_starts, clamped) - 1
        if not 0 <= line_index < len(self.origins):
            return None
        origin = self.origins[line_index]
        if origin is None:
            return None
        column = clamped - self.line_starts[line_index] + 1
        return SourcePosition(origin.file, origin.line, column)


@dataclass(frozen=True)
class PreprocessedSource:
    source_file: Path
    text: str
    source_map: SourceMap
    compiler: str
    language: str


@dataclass(frozen=True)
class CompileCommand:
    directory: Path
    file: Path
    arguments: tuple[str, ...]


class CompileDatabase:
    """A small JSON compilation database with exact source-file lookup."""

    def __init__(self, path: Path, commands: tuple[CompileCommand, ...]) -> None:
        self.path = path
        self.commands = commands
        self._by_file: dict[Path, CompileCommand] = {}
        for command in commands:
            self._by_file.setdefault(command.file, command)

    @classmethod
    def load(cls, path: str | Path) -> CompileDatabase:
        db_path = Path(path).expanduser().resolve()
        if not db_path.is_file():
            raise CompileDatabaseError(
                f"compile database not found: {db_path}. Place it at "
                "<project-root>/compile_commands.json or pass "
                "--compile-commands <path>."
            )
        try:
            payload = json.loads(db_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompileDatabaseError(
                f"cannot read compile database {db_path}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise CompileDatabaseError(
                f"compile database {db_path} must contain a JSON array"
            )

        commands: list[CompileCommand] = []
        for index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                continue
            command = _parse_entry(entry, db_path.parent, index)
            if command is not None:
                commands.append(command)
        if not commands:
            raise CompileDatabaseError(
                f"compile database {db_path} has no usable entries"
            )
        return cls(db_path, tuple(commands))

    def command_for(self, source_file: str | Path) -> CompileCommand:
        requested = Path(source_file).expanduser().resolve()
        command = self._by_file.get(requested)
        if command is None:
            raise CompileDatabaseError(
                f"no compile database entry for {requested}; add an entry for "
                "every report source file, including headers, to "
                f"{self.path}"
            )
        return command

    def preprocess(
        self,
        source_file: str | Path,
        *,
        timeout: float = 30.0,
        max_output_bytes: int = 64 * 1024 * 1024,
    ) -> PreprocessedSource:
        command = self.command_for(source_file)
        invocation = _preprocess_invocation(command)
        try:
            result = subprocess.run(
                invocation,
                cwd=command.directory,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CompileDatabaseError(
                f"preprocessing failed for {command.file}: {exc}"
            ) from exc
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            detail = " ".join(stderr.split())[-800:]
            raise CompileDatabaseError(
                f"preprocessor exited with {result.returncode} for "
                f"{command.file}: {detail}"
            )
        if not result.stdout:
            raise CompileDatabaseError(
                f"preprocessor produced no output for {command.file}"
            )
        if len(result.stdout) > max_output_bytes:
            raise CompileDatabaseError(
                f"preprocessed output for {command.file} exceeds "
                f"{max_output_bytes} bytes"
            )
        text = result.stdout.decode("utf-8", errors="replace")
        source_map = build_source_map(text, command.directory)
        text, source_map = strip_line_markers(text, source_map)
        return PreprocessedSource(
            source_file=command.file,
            text=text,
            source_map=source_map,
            compiler=Path(invocation[0]).name,
            language=_command_language(command),
        )


def default_compile_commands(source_root: str | Path) -> Path:
    """Return the required project-local compile database path."""
    return Path(source_root).expanduser().resolve() / COMPILE_COMMANDS_NAME


def build_source_map(text: str, working_dir: Path) -> SourceMap:
    marker = re.compile(r'^\s*#\s+(\d+)\s+"((?:\\.|[^"])*)"')
    line_starts: list[int] = []
    origins: list[SourcePosition | None] = []
    current_file = ""
    current_line = 1
    output_offset = 0

    for raw_line in text.splitlines(keepends=True):
        line_starts.append(output_offset)
        output_offset += len(raw_line.encode("utf-8", errors="replace"))
        line = raw_line.rstrip("\r\n")
        match = marker.match(line)
        if match:
            current_line = int(match.group(1))
            current_file = _decode_marker_path(match.group(2))
            origins.append(None)
            continue
        if current_file:
            resolved = _resolve_marker_path(current_file, working_dir)
            origins.append(SourcePosition(resolved, current_line))
            current_line += 1
        else:
            origins.append(None)
    return SourceMap(tuple(line_starts), tuple(origins))


def raw_source_map(source_file: str | Path, text: str) -> SourceMap:
    path = str(Path(source_file).expanduser().resolve())
    line_starts: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_starts.append(offset)
        offset += len(line.encode("utf-8", errors="replace"))
    return SourceMap(
        tuple(line_starts),
        tuple(SourcePosition(path, index + 1) for index in range(len(line_starts))),
    )


def strip_line_markers(text: str, source_map: SourceMap) -> tuple[str, SourceMap]:
    """Remove compiler line directives while retaining line-origin metadata."""
    marker = re.compile(r'^\s*#\s+(\d+)\s+"((?:\\.|[^"])*)"')
    sanitized_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = (
            "\r\n" if line.endswith("\r\n")
            else "\n" if line.endswith("\n")
            else ""
        )
        body = line[:-len(newline)] if newline else line
        sanitized_lines.append(
            (" " * len(body) + newline) if marker.match(line) else line
        )
    sanitized = "".join(sanitized_lines)
    line_starts: list[int] = []
    offset = 0
    for line in sanitized.splitlines(keepends=True):
        line_starts.append(offset)
        offset += len(line.encode("utf-8", errors="replace"))
    return sanitized, SourceMap(tuple(line_starts), source_map.origins)


def _parse_entry(
    entry: dict[str, Any],
    fallback_dir: Path,
    index: int,
) -> CompileCommand | None:
    raw_file = entry.get("file")
    if not isinstance(raw_file, str) or not raw_file:
        return None
    directory = Path(str(entry.get("directory") or fallback_dir)).expanduser()
    if not directory.is_absolute():
        directory = (fallback_dir / directory).resolve()
    else:
        directory = directory.resolve()
    file_path = Path(raw_file).expanduser()
    if not file_path.is_absolute():
        file_path = (directory / file_path).resolve()
    else:
        file_path = file_path.resolve()

    arguments = entry.get("arguments")
    if arguments is None:
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        try:
            arguments = shlex.split(command)
        except ValueError as exc:
            raise CompileDatabaseError(
                f"invalid command in compile database entry {index}: {exc}"
            ) from exc
    if not isinstance(arguments, list) or not all(
        isinstance(arg, str) for arg in arguments
    ):
        raise CompileDatabaseError(
            f"compile database entry {index} has invalid arguments"
        )
    return CompileCommand(directory, file_path, tuple(arguments))


def _preprocess_invocation(command: CompileCommand) -> list[str]:
    args = list(command.arguments)
    if not args:
        raise CompileDatabaseError(f"empty compile command for {command.file}")

    while args and Path(args[0]).name in {"ccache", "sccache", "distcc"}:
        args.pop(0)
    compiler = args.pop(0)
    flags: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-o", "-MF", "-MT", "-MQ", "-MJ"}:
            skip_next = True
            continue
        if arg.startswith(("-o", "-MF", "-MT", "-MQ", "-MJ")):
            continue
        if arg in {
            "-c", "-S", "-E", "-MD", "-MMD", "-MP", "-M", "-MM",
            "-fsyntax-only",
        }:
            continue
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = (command.directory / candidate)
        try:
            is_source = candidate.resolve() == command.file
        except OSError:
            is_source = False
        if is_source:
            continue
        flags.append(arg)
    return [compiler, "-E", *flags, str(command.file)]


def _decode_marker_path(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r"\\", "\\").replace(r"\"", '"')


def _resolve_marker_path(value: str, working_dir: Path) -> str:
    if value.startswith("<") and value.endswith(">"):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = working_dir / path
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _command_language(command: CompileCommand) -> str:
    args = list(command.arguments)
    for index, argument in enumerate(args[:-1]):
        if argument == "-x":
            value = args[index + 1].lower()
            if value.startswith("c++"):
                return "cpp"
            if value == "c" or value.startswith("c-"):
                return "c"
    extension = command.file.suffix.lower()
    return "c" if extension == ".c" else "cpp"
