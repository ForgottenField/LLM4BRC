"""The other half of the model: the code a report points at.

A report names files with the paths of the machine that produced it —

* CSA writes the build machine's absolute path
  (``/home/me/projects/cpp-project/faiss/faiss/impl/index_read.cpp``),
* CodeQL writes a URI relative to the database's source root
  (``faiss/impl/index_read.cpp``, ``uriBaseId: %SRCROOT%``),
* Cppcheck and Infer write whatever they were invoked with.

Three different spellings of the same file.  Everything downstream —
segmentation, PDG lookup, snippet extraction, the dependency builder's seed
set — needs the *local* file, so the mapping has to happen exactly once, in
one place, with its outcome recorded rather than assumed.

That is what :class:`SourceCodeModel` is.  It replaces three near-duplicate
resolvers that had grown up separately (``html_parser.resolve_report_source_file``,
``source_context.SourceContextExtractor._resolve_to_local``,
``build_project_deps.resolve_source``) and it makes the two defects they shared
impossible:
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: Build roots CSA reports were generated under.  Recognising one is what turns
#: an absolute build path into a project-relative one; they are matched longest
#: first so the more specific `fbcode_builder` root wins over a prefix of it.
KNOWN_BUILD_PREFIXES: tuple[str, ...] = (
    "/tmp/fbcode_builder_getdeps-ZhomeZmeZprojectsZcpp-projectZfollyZbuildZfbcode_builder/",
    "/home/me/projects/cpp-project/",
)

#: Directories that never hold report-relevant source.
_SKIP_DIRS = frozenset({".git", "__pycache__", ".mypy_cache", ".pytest_cache"})

#: Extensions a definition search will open.
_SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inl")

#: Resolution strategies, weakest last.  ``embedded`` means the report itself
#: carries that file's source, so no local file is needed for a snippet — but
#: a local file may still be resolved for the PDG/CFG side of the pipeline.
STRATEGY_DIRECT = "direct"
STRATEGY_BUILD_PREFIX = "build-prefix"
STRATEGY_RELATIVE = "relative"
STRATEGY_SUFFIX = "suffix"
STRATEGY_BASENAME = "basename"
STRATEGY_EMBEDDED = "embedded"
STRATEGY_UNRESOLVED = "unresolved"

#: The strategies a caller may treat as *evidence a path is right*.  ``suffix``
#: is in and ``basename`` is out on purpose: a suffix match pins the file's
#: position in the tree, a basename match only pins its name.
STRONG_STRATEGIES = frozenset({
    STRATEGY_DIRECT, STRATEGY_BUILD_PREFIX, STRATEGY_RELATIVE, STRATEGY_SUFFIX,
})

#: How many report-vs-tree disagreements one model reports in full.  The record
#: is a *signal* ("this checkout has moved on"), not a diff: a stale tree
#: disagrees on every line of the window, and the result JSON is not the place
#: for thousands of them.  What the cap drops is counted in
#: :attr:`SourceCodeModel.drift_truncated`.
_MAX_DRIFT_RECORDS = 200

#: How many files one report's materialisation reports in full, and how many
#: unanticipated reads it names.  Both records are *inventories* — the cap
#: bounds the result JSON, and every record it drops is counted, so a truncated
#: list cannot read as a short one.
_MAX_CODE_SET_RECORDS = 64
_MAX_LAZY_READ_RECORDS = 32

#: Stem-relative siblings of the file a report names.  A C++ declaration and its
#: template body routinely live in ``X.h``/``X-inl.h`` next to ``X.cpp``, and the
#: pipeline reads them; materialising them costs one ``is_file`` each and no tree
#: walk, which is why they are included while whole-tree scans are not.
_COMPANION_SUFFIXES = (".h", "-inl.h", ".cc", ".cpp", ".hh", ".hpp")


@dataclass(frozen=True)
class ResolvedFile:
    """Outcome of resolving one reported path against the local tree."""

    reported: str
    local: Optional[Path] = None
    strategy: str = STRATEGY_UNRESOLVED
    #: Every local file that shares the reported path's name, **the chosen one
    #: first**.  Non-empty only for a basename match (the one rung that cannot
    #: tell same-named files apart); recorded so a name-only guess is visible
    #: rather than silently accepted.  Use :attr:`rivals` for the ones it *did
    #: not* choose.
    candidates: tuple[Path, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.local is not None

    @property
    def is_strong(self) -> bool:
        """True when the resolution pins a position, not just a filename."""
        return self.strategy in STRONG_STRATEGIES

    @property
    def is_ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def rivals(self) -> tuple[Path, ...]:
        """The candidates the resolution passed over — empty when unambiguous.

        The chosen file is the first element of ``candidates``, so anything
        reporting the ambiguity must drop it: a message reading
        ``X → X (also: X, Y)`` says the opposite of what it means.
        """
        return self.candidates[1:]

    def to_dict(self) -> dict:
        return {
            "reported": self.reported,
            "local": str(self.local) if self.local else None,
            "strategy": self.strategy,
            "ambiguous_with": [str(p) for p in self.rivals],
        }


@dataclass
class CodeReference:
    """One place a report points at, and what it resolved to.

    ``role`` says why the report mentions it, so a consumer can decide how much
    the mention matters: ``bug_file``/``related`` are claims, ``step`` is
    context, and ``event_file`` is a file the reported path physically crosses
    (the seed set for dependency building).
    """

    reported: str
    line: int = 0
    role: str = "step"
    resolution: ResolvedFile = field(
        default_factory=lambda: ResolvedFile(reported="")
    )
    function: str = ""

    def to_dict(self) -> dict:
        return {
            "reported": self.reported,
            "line": self.line,
            "role": self.role,
            "function": self.function,
            **self.resolution.to_dict(),
        }


# ---------------------------------------------------------------------------
# Source-tree index
# ---------------------------------------------------------------------------

#: ``source_root -> index``, built once per tree per process.  Read-only, so
#: sharing it across the reports of a batch run is safe (unlike the CFG cache
#: set, which is mutated per report and is therefore built fresh each time).
_INDEX_CACHE: dict[str, "SourceTreeIndex"] = {}


class SourceTreeIndex:
    """``basename`` and ``relative-path`` lookups over one source tree.

    One :func:`os.walk` per tree: the alternative — an ``rglob`` per event —
    is minutes of wall clock on a protobuf-sized checkout, and it is what made
    the old resolver pick a different file depending on call order.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.by_basename: dict[str, list[Path]] = {}
        self.relative: set[str] = set()
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                full = Path(dirpath) / name
                self.by_basename.setdefault(name, []).append(full)
                try:
                    self.relative.add(full.relative_to(self.root).as_posix())
                except ValueError:  # pragma: no cover - walk stays under root
                    continue
        for paths in self.by_basename.values():
            paths.sort()
        self._built = True
        logger.debug(
            "source index built for %s: %d files", self.root, len(self.relative),
        )

    def basenames(self, name: str) -> list[Path]:
        self.build()
        return list(self.by_basename.get(name, ()))

    def has_relative(self, rel: str) -> bool:
        self.build()
        return rel in self.relative

    def relative_paths(self) -> list[str]:
        """Every file under the root, root-relative, sorted.

        A definition search walks *this* rather than ``rglob``-ing the tree
        again per call: one ``os.walk``, a stable order, and the same list
        every caller sees.
        """
        self.build()
        return sorted(self.relative)


def _index_for(root: Path) -> SourceTreeIndex:
    key = str(root)
    index = _INDEX_CACHE.get(key)
    if index is None:
        index = SourceTreeIndex(root)
        _INDEX_CACHE[key] = index
    return index


def clear_index_cache() -> None:
    """Drop the cached tree indexes (tests, and long-lived processes)."""
    _INDEX_CACHE.clear()


def _scan_function_spans(lines: list[str]) -> dict[str, tuple[int, int]]:
    """``{function name: (start, end)}`` for one file, by brace matching.

    Ported from the CFG path's scanner so a project *without* a PDG artifact
    still gets function extents from the one model.  It is deliberately a
    heuristic — the PDG is authoritative where it exists — and it keys on the
    bare name before ``(``, which is the convention the CFG-cache lookup uses.
    """
    ranges: dict[str, tuple[int, int]] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (
            line
            and not line.startswith(("#", "//", "/*", "*"))
            and "(" in line
            and ")" in line
            and ";" not in line.rstrip(" {")
            and not line.startswith((
                "if ", "while ", "for ", "switch ", "case ", "return ",
                "else ", "typedef ", "enum ", "struct ", "union ",
            ))
        ):
            depth = 0
            for j in range(i, len(lines)):
                stripped = lines[j].strip()
                if stripped.startswith(("//", "/*", "*")):
                    continue
                for ch in stripped:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            name_match = re.search(r"(\w+)\s*\(", line.rstrip("{"))
                            if name_match:
                                ranges[name_match.group(1)] = (i + 1, j + 1)
                                i = j
                            break
                if depth == 0:
                    break
        i += 1
    return ranges


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class SourceCodeModel:
    """Resolves a report's paths against a local tree and reads the code.

    One instance per report run.  It is cheap until something needs a file
    list, at which point it builds (and caches) the tree index.
    """

    def __init__(
        self,
        source_root: str | Path | None = None,
        prefixes: Iterable[str] = KNOWN_BUILD_PREFIXES,
        uri_bases: Iterable[str] = (),
    ) -> None:
        self.source_root = Path(source_root).resolve() if source_root else None
        self.prefixes = tuple(prefixes)
        self.uri_bases = tuple(uri_bases)
        #: True for a model from :func:`shared_model`, i.e. one every report in
        #: the process shares.  Such a model may hold only what is a fact about
        #: the *tree* — the resolution memo, the tree index — never per-report
        #: state, so the writers that carry a report's own facts
        #: (:meth:`set_function_index`) refuse it.  The distinction is silent
        #: otherwise: a shared model that accepted one report's SDG would answer
        #: the next report's ``function_body`` from the wrong graph.
        self._is_shared = False
        #: ``reported path -> {line: code}`` for source the report embedded.
        self._embedded: dict[str, dict[int, str]] = {}
        self._resolved: dict[str, ResolvedFile] = {}
        self._lines: dict[Path, list[str]] = {}
        #: ``local file -> bytes read`` for everything :meth:`materialize` read,
        #: and ``-1`` for a file it tried and could not read.  Membership is
        #: "already attempted", which is what makes a read *lazy* rather than
        #: anticipated.
        self._materialized: dict[Path, int] = {}
        #: Files :meth:`lines` had to read that materialisation did not
        #: anticipate — the signal that the pre-load missed something.  Capped;
        #: what the cap drops is counted in :attr:`lazy_reads_truncated`.
        self._lazy_reads: list[str] = []
        self._lazy_read_overflow = 0
        self._code_set: Optional[dict] = None
        #: ``function base name -> [(local file, start, end)]``, from the PDG.
        #: Authoritative where present; :meth:`function_body` falls back to a
        #: brace scan when a project has no PDG artifact.
        self._functions: dict[str, list[tuple[Path, int, int]]] = {}
        #: ``local file -> {function name: (start, end)}`` for the brace scan.
        self._fn_spans: dict[Path, dict[str, tuple[int, int]]] = {}
        #: Lines where the report's own copy and the local tree disagree.
        #: Deduped by ``(reported, line)`` and capped (see :meth:`_record_drift`).
        self._drift: list[dict] = []
        self._drift_seen: set[tuple[str, int]] = set()
        self._drift_overflow = 0

    # -- embedded source ---------------------------------------------------

    def add_embedded(self, reported: str, lines: dict[int, str]) -> None:
        """Register source under the local file shared by every path alias.

        CSA may title a section ``./header.h`` or ``file.c`` while ``BUGFILE``
        uses the analyzer machine's absolute path.  Resolving first prevents
        those spellings from creating separate embedded-source entries.
        """
        if reported and lines:
            self._embedded.setdefault(
                self._embedded_key(reported), {},
            ).update(lines)
            # ``_embedded_key`` resolves the alias and may have cached an
            # unresolved answer before the source was registered.
            self._resolved.pop(reported, None)

    def has_embedded(self, reported: str) -> bool:
        """True when the report carries source for *reported* or its alias."""
        return bool(self._embedded_for(reported))

    def _embedded_key(self, reported: str) -> str:
        """Canonical key shared by BUGFILE and section-path spellings."""
        local = self.resolve(reported).local
        return str(local) if local is not None else reported

    def _embedded_for(self, reported: str) -> dict[int, str]:
        return self._embedded.get(self._embedded_key(reported), {})

    @property
    def embedded_files(self) -> tuple[str, ...]:
        return tuple(self._embedded)

    # -- materialisation ---------------------------------------------------

    def materialize(self, finding, *, companions: bool = True) -> dict:
        """Read the code *finding* points at into the model, whole files.

        The report is an index: it says which files and which lines, and this is
        where that index is turned into code.  Everything the report names — the
        bug file, the path's event files, the call targets — is resolved through
        the one ladder and read in full, so the stages below take their code
        from the model instead of deciding for themselves which file to open.

        Whole files rather than windows, because a window is a guess about what
        the reader will want: the pipeline asks for the bug line's neighbourhood,
        then an event line past the end of that window, then a callee's body.
        Reading the file once answers all three.

        This is also the *only* place a same-stem companion is materialised
        (:data:`_COMPANION_SUFFIXES`, one ``is_file`` each) — the shape the
        ``stem + '-inl.h'`` guess was reaching for, but checked against the
        directory the file actually resolved to.

        Returns the inventory (also held as :attr:`code_set`), which is *visible
        only*: it is reported in the result JSON next to ``drift`` and never
        influences a verdict.  Files that could not be read and reported paths
        that resolved to nothing are listed rather than counted away.

        Note for callers: materialise the **per-report** model
        (:class:`ReportFrontend` builds one per ``load``), never
        :func:`shared_model` — that one outlives the report, so filling it would
        accumulate every report's code across a ``--scan-dir`` batch.
        """
        files: list[dict] = []
        unresolved: list[dict] = []
        seen: set[Path] = set()
        reported_seen: set[str] = set()
        truncated = 0
        total_lines = total_bytes = 0

        for ref in self.references(finding):
            if ref.reported in reported_seen:
                # One record per reported path: the index's own unit.  Two
                # events on the same file are one file to read.
                continue
            reported_seen.add(ref.reported)
            local = ref.resolution.local
            if local is None:
                if self.has_embedded(ref.reported):
                    # No local file, but the report carries this one's source:
                    # readable without the tree, which is why it is not a gap.
                    files.append({
                        "reported": ref.reported,
                        "local": None,
                        "strategy": ref.resolution.strategy,
                        "source": STRATEGY_EMBEDDED,
                        "lines": len(self._embedded_for(ref.reported)),
                        "bytes": 0,
                    })
                else:
                    unresolved.append({
                        "reported": ref.reported,
                        "strategy": ref.resolution.strategy,
                    })
                continue

            key = Path(local)
            first = key not in seen
            seen.add(key)
            nbytes = self._read_file(key)
            record = {
                "reported": ref.reported,
                "local": str(key),
                "strategy": ref.resolution.strategy,
                "source": "tree" if nbytes is not None else "unreadable",
                "lines": len(self._lines.get(key, [])),
                "bytes": nbytes if nbytes is not None else 0,
            }
            if first:
                total_lines += record["lines"]
                total_bytes += record["bytes"]
            if len(files) >= _MAX_CODE_SET_RECORDS:
                truncated += 1
            else:
                files.append(record)

        companions_out: list[dict] = []
        companions_truncated = 0
        if companions:
            for local in sorted(seen):
                for cand in self._companion_files(local):
                    if cand in seen:
                        continue
                    seen.add(cand)
                    nbytes = self._read_file(cand)
                    loaded = len(self._lines.get(cand, []))
                    total_lines += loaded
                    total_bytes += nbytes or 0
                    if len(companions_out) >= _MAX_CODE_SET_RECORDS:
                        companions_truncated += 1
                    else:
                        companions_out.append({
                            "local": str(cand), "lines": loaded,
                            "bytes": nbytes or 0,
                        })

        self._code_set = {
            "files": files,
            "files_truncated": truncated,
            "companions": companions_out,
            "companions_truncated": companions_truncated,
            "unresolved": unresolved,
            "reported_paths": len(files) + truncated + len(unresolved),
            "lines": total_lines,
            "bytes": total_bytes,
        }
        logger.debug(
            "code model: materialised %d file(s) + %d companion(s) (%d lines, "
            "%d KB) for %s; %d reported path(s) unresolved",
            len(files), len(companions_out), total_lines, total_bytes // 1024,
            getattr(finding, "finding_id", "?"), len(unresolved),
        )
        return self._code_set

    def _companion_files(self, local: Path) -> list[Path]:
        """Same-stem siblings of *local* that exist, by the model's own rule.

        Naming a file is not finding it: each candidate is a real path beside
        the file that *resolved*, so ``X-inl.h`` is materialised when it is
        there and is not invented when it is not.
        """
        primary = Path(local)
        out: list[Path] = []
        for suffix in _COMPANION_SUFFIXES:
            candidate = primary.parent / f"{primary.stem}{suffix}"
            if candidate != primary and candidate.is_file():
                out.append(candidate)
        return out

    @property
    def code_set(self) -> Optional[dict]:
        """What :meth:`materialize` read — ``None`` before it runs."""
        return dict(self._code_set) if self._code_set is not None else None

    @property
    def lazy_reads(self) -> list[str]:
        """Files read *after* materialisation, i.e. the pre-load's blind spot."""
        return list(self._lazy_reads)

    @property
    def lazy_reads_truncated(self) -> int:
        """Lazy reads the :data:`_MAX_LAZY_READ_RECORDS` cap dropped."""
        return self._lazy_read_overflow

    # -- resolution --------------------------------------------------------

    def resolve(self, reported: str) -> ResolvedFile:
        """Map a reported path to a local file (memoised per instance)."""
        if reported in self._resolved:
            return self._resolved[reported]
        result = self._resolve_uncached(reported)
        self._resolved[reported] = result
        if not result.resolved:
            logger.debug("unresolved report path: %s", reported)
        elif result.is_ambiguous:
            logger.warning(
                "ambiguous report path %s → %s (also: %s; matched by basename)",
                reported, result.local,
                ", ".join(str(p) for p in result.rivals[:3]),
            )
        return result

    def resolve_strong(self, reported: str) -> Optional[Path]:
        """Resolve only when the match pins a position, not just a basename."""
        hit = self.resolve(reported)
        return hit.local if hit.is_strong else None

    def _resolve_uncached(self, reported: str) -> ResolvedFile:
        """The resolution ladder.  Each rung is a strictly weaker claim."""
        raw = (reported or "").strip()
        if not raw:
            return ResolvedFile(reported=reported, strategy=STRATEGY_UNRESOLVED)

        # 0. A report that carries the file's own source has, in the only sense
        #    that matters for snippets, resolved it already.  A local file may
        #    still be found below; both answers are useful and both are kept.
        embedded = raw in self._embedded

        candidate = Path(raw)
        # 1. The build machine's path is this machine's path.
        if candidate.is_absolute() and candidate.is_file():
            return ResolvedFile(raw, candidate.resolve(), STRATEGY_DIRECT)

        if self.source_root is not None:
            # 2. Strip a known build root and join the remainder onto ours,
            #    trimming leading components until something exists — reports
            #    spell the project directory once or twice depending on how
            #    the build was laid out.
            for prefix in sorted(self.prefixes, key=len, reverse=True):
                if raw.startswith(prefix):
                    hit = self._join_trimmed(
                        raw[len(prefix):].lstrip("/"), STRATEGY_BUILD_PREFIX, raw,
                    )
                    if hit:
                        return hit

            # 3. A relative path (SARIF, Cppcheck) is relative to the root.
            if not candidate.is_absolute():
                hit = self._join_trimmed(raw, STRATEGY_RELATIVE, raw)
                if hit:
                    return hit
                # ... and possibly to one of the URI bases the document named.
                for base in self.uri_bases:
                    hit = self._join_trimmed(
                        raw, STRATEGY_RELATIVE, raw, base=Path(base),
                    )
                    if hit:
                        return hit

            # 4. Longest suffix present in the tree: pins the position, so it
            #    survives an arbitrary (build-machine) prefix and a tree whose
            #    top-level directory is named differently.
            parts = [p for p in raw.split("/") if p not in ("", ".")]
            for cut in range(len(parts)):
                rel = "/".join(parts[cut:])
                index = _index_for(self.source_root)
                if index.has_relative(rel):
                    return ResolvedFile(
                        raw, (self.source_root / rel).resolve(), STRATEGY_SUFFIX,
                    )

        # 5. Basename: weakest, and the only one that can be ambiguous.
        hits = (
            _index_for(self.source_root).basenames(candidate.name)
            if self.source_root is not None else []
        )
        if hits:
            return ResolvedFile(raw, hits[0], STRATEGY_BASENAME, tuple(hits))

        if embedded:
            return ResolvedFile(raw, None, STRATEGY_EMBEDDED)
        return ResolvedFile(raw, None, STRATEGY_UNRESOLVED)

    def _join_trimmed(
        self, rel: str, strategy: str, original: str, base: Optional[Path] = None,
    ) -> Optional[ResolvedFile]:
        """Join *rel* under the root, dropping leading components until it hits.

        ``.../faiss/faiss/impl/io.cpp`` → try ``faiss/faiss/impl/io.cpp``, then
        ``faiss/impl/io.cpp``, then ``impl/io.cpp``.  Without the trimming, a
        report from a checkout whose root is one level deeper than the local
        one resolves to nothing and falls through to the basename guess.
        """
        root = base or self.source_root
        if root is None:
            return None
        parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
        for cut in range(len(parts)):
            local = root.joinpath(*parts[cut:])
            if local.is_file():
                return ResolvedFile(
                    reported=original, local=local.resolve(), strategy=strategy,
                )
        return None

    # -- reading -----------------------------------------------------------

    def lines(self, local: Path) -> list[str]:
        """All lines of a local file (cached per instance; ``[]`` if unreadable).

        The one disk entry point of the model.  A file :meth:`materialize` did
        not read is still readable — a callee body discovered mid-run has to be
        — but the read is recorded as *lazy* so the pre-load's coverage is
        measurable rather than assumed.
        """
        key = Path(local)
        if key not in self._lines:
            if key not in self._materialized:
                self._note_lazy_read(key)
            self._read_file(key)
        return self._lines.get(key, [])

    def _read_file(self, local: Path) -> Optional[int]:
        """Read *local* into the line cache; its byte size, or ``None``.

        Records the attempt in ``_materialized`` either way: a file that cannot
        be read is read *once*, not once per caller.
        """
        key = Path(local)
        try:
            text = key.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("cannot read %s: %s", key, exc)
            self._lines[key] = []
            self._materialized[key] = -1
            return None
        self._lines[key] = text.splitlines()
        self._materialized[key] = len(text.encode("utf-8", "replace"))
        return self._materialized[key]

    def _note_lazy_read(self, local: Path) -> str:
        """Record a read materialisation did not anticipate."""
        path = str(local)
        if len(self._lazy_reads) >= _MAX_LAZY_READ_RECORDS:
            self._lazy_read_overflow += 1
        else:
            self._lazy_reads.append(path)
        logger.debug("code model: lazy read (not materialised): %s", path)
        return path

    def read_range(
        self, local: Path, start: int, end: int, keep_blank: bool = False,
    ) -> dict[int, str]:
        """Lines ``start..end`` (1-based, inclusive) of a local file.

        ``keep_blank`` defaults to False — blank lines are dropped, which is
        what every caller has always got.  A *rendered* window (the text an LLM
        reads, where a gap in the numbering is a lie about the file) wants
        ``keep_blank=True``; changing the default would silently alter every
        window already produced.
        """
        lines = self.lines(local)
        if not lines:
            return {}
        lo = max(1, start)
        hi = min(len(lines), end)
        if keep_blank:
            return {n: lines[n - 1] for n in range(lo, hi + 1)}
        return {n: lines[n - 1] for n in range(lo, hi + 1) if lines[n - 1].strip()}


    def snippet(
        self, reported: str, line: int, radius: int = 30, embedded_first: bool = True,
    ) -> dict[int, str]:
        """Source lines around *line* for a reported file.

        Embedded source wins by default: a CSA report's own copy of the flagged
        region is what ``build_project_deps.verify_line_alignment`` compares the
        local tree against, so reading it keeps the pipeline's line-alignment
        signal meaningful.  A SARIF report embeds nothing and the local file is
        then the only source.
        """
        embedded = self._embedded_for(reported)
        if embedded_first and embedded:
            window = {
                n: text for n, text in embedded.items()
                if abs(n - line) <= radius
            }
            if window:
                window = dict(sorted(window.items()))
                self._record_drift(reported, window)
                return window

        resolved = self.resolve(reported)
        if not resolved.local:
            return {}
        window = self.read_range(resolved.local, line - radius, line + radius)
        # Tie each line to the *file* it came from so a multi-file report's
        # window cannot mix files that share a line number.
        return window

    # -- drift: the report's copy vs the tree -------------------------------

    def _record_drift(self, reported: str, window: dict[int, str]) -> None:
        """Compare a served embedded window against the local tree, line by line.

        The report is the revision the analyzer ran on and is what the line
        numbers refer to, so it stays authoritative — but a tree that has moved
        on is a fact the pipeline should be able to *see* rather than infer.
        Only lines that both copies have are compared, and blank-vs-blank is not
        a disagreement.

        Recorded once per ``(reported, line)``: ``snippet()`` is called once per
        rendered block and the simulation renders one per path step, so the same
        disagreement would otherwise be appended hundreds of times.  The list is
        also capped — it is a signal to report, not a diff — and every record
        the cap dropped is counted in :attr:`drift_truncated` so a truncated
        list cannot read as a small one.
        """
        resolved = self.resolve(reported)
        if not resolved.local:
            return
        local = self.read_range(
            resolved.local, min(window), max(window), keep_blank=True,
        )
        for n, embedded_text in window.items():
            local_text = local.get(n)
            if local_text is None:
                continue
            if embedded_text.strip() != local_text.strip():
                key = (reported, n)
                if key in self._drift_seen:
                    continue
                self._drift_seen.add(key)
                if len(self._drift) >= _MAX_DRIFT_RECORDS:
                    self._drift_overflow += 1
                    continue
                self._drift.append({
                    "reported": reported,
                    "line": n,
                    "embedded": embedded_text.strip(),
                    "local": local_text.strip(),
                    # *How* that local file was found is part of the record: a
                    # disagreement against a ``basename`` match may be a
                    # different file of that name rather than a moved checkout,
                    # and the two call for different actions.
                    "strategy": resolved.strategy,
                })

    @property
    def drift(self) -> list[dict]:
        """``[{reported, line, embedded, local, strategy}]`` — report vs tree."""
        return list(self._drift)

    @property
    def drift_truncated(self) -> int:
        """Records the :data:`_MAX_DRIFT_RECORDS` cap dropped, never served as 0."""
        return self._drift_overflow

    # -- function bodies ----------------------------------------------------

    def set_function_index(
        self, index: dict[str, list[tuple[Path, int, int]]],
    ) -> None:
        """Register where functions are defined, keyed by base name.

        The PDG is the accurate source of a function's extent — a brace scan
        misses one whose body is nested inside another's, and a name-only
        lookup lands on same-named functions from other classes.  Whoever holds
        the SDG builds this once; the model then answers
        :meth:`function_body` from a fact instead of a guess.

        Refused on a shared model: the index is one report's SDG, so keeping it
        there would answer another report's lookup from the wrong graph (and
        the refusal is loud rather than silent, since the caller cannot see
        which model it was handed).
        """
        if self._is_shared:
            logger.warning(
                "set_function_index on the shared model ignored: the index "
                "belongs to one report's SDG and would leak across reports",
            )
            return
        self._functions = {
            name: list(entries) for name, entries in (index or {}).items()
        }

    def function_body(
        self,
        name: str,
        *,
        file_hint: Optional[Path] = None,
        max_lines: Optional[int] = None,
    ) -> Optional[tuple[Path, int, int, str]]:
        """Locate *name* and return ``(local, start, end, text)``, or ``None``.

        The PDG index wins where it has the function.  Otherwise a brace scan
        runs over this model's own tree index, starting with the files beside
        *file_hint* (where a callee usually sits) and widening to the whole
        tree — one indexed walk, not an ``rglob`` per call.
        """
        base = (name or "").split("::")[-1].strip()
        if not base:
            return None

        entry = self._pick_function_entry(base, file_hint)
        if entry is not None:
            local, start, end = entry
            text = "\n".join(self.lines(local)[start - 1: end])
            return local, start, end, text

        for local in self._definition_candidates(file_hint):
            span = self.function_span(local, base)
            if span is None:
                continue
            start, end = span
            if max_lines is not None and end - start + 1 > max_lines:
                end = start + max_lines - 1
            text = "\n".join(self.lines(local)[start - 1: end])
            return local, start, end, text
        return None

    def _pick_function_entry(
        self, base: str, file_hint: Optional[Path],
    ) -> Optional[tuple[Path, int, int]]:
        """Choose among the PDG's definitions of *base* — nearest *file_hint*.

        The hint is a file when the caller knows one (an event's own file) and
        a directory when it only knows where to look; either way the entry in
        that place wins, then one in the same directory, then the PDG's first.
        """
        entries = self._functions.get(base)
        if not entries:
            return None
        if file_hint is not None:
            hint = Path(file_hint)
            wanted_dir = hint if hint.is_dir() else hint.parent
            for local, _start, _end in entries:
                if local == hint:
                    return local, _start, _end
            same_dir = [e for e in entries if e[0].parent == wanted_dir]
            if same_dir:
                return same_dir[0]
        return entries[0]

    def function_files(
        self, name: str, file_hint: Optional[Path] = None,
    ) -> list[Path]:
        """Files that may define *name*, most likely first, deduplicated.

        The *locating* half of :meth:`function_body`, for a caller that parses
        the body itself.  The PDG's answer comes first — it knows where a
        function is even when its body is nested inside another's — and the
        tree index supplies the rest, again with no ``rglob`` per call.

        Returning files rather than a span is deliberate: the extent the PDG
        records is its nodes' line range, which starts at the first *statement*
        — a caller rendering a function for a model wants the signature line
        too (it carries the parameter types), and that is a parsing question,
        not a location one.
        """
        base = (name or "").split("::")[-1].strip()
        if not base:
            return []
        out: list[Path] = []
        seen: set[Path] = set()

        def add(candidate: Optional[Path]) -> None:
            if candidate is None:
                return
            candidate = Path(candidate)
            if candidate not in seen and candidate.is_file():
                seen.add(candidate)
                out.append(candidate)

        entry = self._pick_function_entry(base, file_hint)
        if entry is not None:
            add(entry[0])
        for local in self._definition_candidates(file_hint):
            add(local)
        return out

    def _definition_candidates(self, file_hint: Optional[Path]) -> list[Path]:
        """Files worth scanning for a definition, nearest first, no duplicates.

        A hint may be a file or a *directory* — the call sites name both (the
        event's own file; the directory a callee is expected beside) — and the
        two mean the same thing to the caller: start here.  Reading a directory
        as if it were a file skipped it and searched its *parent*, which is not
        near the hint at all, so it is resolved explicitly.
        """
        seen: set[Path] = set()
        out: list[Path] = []

        def add(p: Path) -> None:
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)

        if file_hint is not None:
            hint = Path(file_hint)
            if hint.is_dir():
                for member in self._sibling_sources(hint):
                    add(member)
            else:
                add(hint)
                for sibling in self._sibling_sources(hint.parent):
                    add(sibling)
        if self.source_root is not None:
            for rel in _index_for(self.source_root).relative_paths():
                if rel.endswith(_SOURCE_SUFFIXES):
                    add(self.source_root / rel)
        return out

    def _sibling_sources(self, directory: Path) -> list[Path]:
        """Files in *directory*, from the tree index (not a fresh listing)."""
        if self.source_root is None:
            return []
        return [
            self.source_root / rel
            for rel in _index_for(self.source_root).relative_paths()
            if Path(rel).parent == directory.relative_to(self.source_root)
            if directory.is_relative_to(self.source_root)
        ]

    def function_span(self, local: Path, name: str) -> Optional[tuple[int, int]]:
        """``(start, end)`` of *name*'s definition in *local*, by brace matching.

        The fallback for a project with no PDG artifact.  Scanned once per file
        and cached: the scan is over the file's lines, not the tree.
        """
        spans = self._fn_spans.get(Path(local))
        if spans is None:
            spans = _scan_function_spans(self.lines(local))
            self._fn_spans[Path(local)] = spans
        return spans.get((name or "").split("::")[-1])

    def companion_window(
        self,
        local: Path,
        line: int,
        radius: int = 15,
        suffixes: tuple[str, ...] = (".h", "-inl.h", ".cc", ".cpp"),
    ) -> dict[str, dict[int, str]]:
        """Code around *line* in a sibling of *local* that shares its stem.

        CSA/folly layouts put a template's body in ``X-inl.h`` beside
        ``X.cpp``, and an event line past the primary file's end is the signal
        that it lives there.  The companion is looked for *beside the resolved
        primary file* and must exist — a name that does not is not invented.
        """
        primary = Path(local)
        parent, stem = primary.parent, primary.stem
        out: dict[str, dict[int, str]] = {}
        for suffix in suffixes:
            candidate = parent / f"{stem}{suffix}"
            if candidate == primary or not candidate.is_file():
                continue
            total = len(self.lines(candidate))
            if not total or not 1 <= line <= total:
                continue
            window = self.read_range(
                candidate, line - radius, line + radius, keep_blank=True,
            )
            if window:
                out[candidate.name] = window
        return out

    # -- the report ↔ code relation ---------------------------------------

    def references(self, finding) -> list[CodeReference]:
        """Every file/line a finding mentions, with its resolution.

        Order is the report's: the bug file, then the path's files in path
        order, then related locations.  Duplicates are kept — the *line* makes
        them distinct and a caller that wants files can key on ``reported``.
        """
        refs: list[CodeReference] = []
        seen: set[tuple[str, int]] = set()

        def add(reported: str, line: int, role: str) -> None:
            key = (reported, line)
            if not reported or key in seen:
                return
            seen.add(key)
            refs.append(
                CodeReference(
                    reported=reported, line=line, role=role,
                    resolution=self.resolve(reported),
                )
            )

        add(finding.location.file, finding.location.line, "bug_file")
        for step in finding.steps:
            if not step.location.file:
                continue
            role = {
                "call": "call_target",
                "error_start": "error_start",
            }.get(step.kind.value, "event_file")
            add(step.location.file, step.location.line, role)
        for loc in finding.related:
            add(loc.file, loc.line, "related")
        return refs

    def unresolved(self, finding) -> list[CodeReference]:
        """References that did not resolve — a report the pipeline cannot place."""
        return [r for r in self.references(finding) if not r.resolution.resolved]

    def summary(self, finding) -> dict:
        """A small, serialisable description of "report ↔ code" for one finding."""
        refs = self.references(finding)
        files = {r.reported for r in refs}
        resolved_files = {r.reported for r in refs if r.resolution.resolved}
        return {
            "source_root": str(self.source_root) if self.source_root else None,
            "reported_files": len(files),
            "resolved_files": len(resolved_files),
            "unresolved": [r.reported for r in refs if not r.resolution.resolved],
            "weak": [
                r.reported for r in refs
                if r.resolution.resolved
                and r.resolution.strategy == STRATEGY_BASENAME
            ],
            "embedded": sorted(self._embedded),
            "references": [r.to_dict() for r in refs],
        }


_MODEL_CACHE: dict[tuple, SourceCodeModel] = {}


def shared_model(
    source_root: str | Path | None,
    prefixes: Iterable[str] = KNOWN_BUILD_PREFIXES,
    uri_bases: Iterable[str] = (),
) -> SourceCodeModel:
    """A process-wide :class:`SourceCodeModel` for *source_root*.

    For callers that resolve many paths against one tree in a loop — the
    pipeline resolves one per report path event and one per SDG function — and
    need the memoisation a fresh model would throw away each iteration.  The
    model holds no report state, so sharing it is safe; the only state it keeps
    is the resolution cache and the (already module-cached) tree index — and
    the report-scoped writers (``set_function_index``, ``add_embedded``) must
    not be pointed at it, which ``set_function_index`` enforces.
    """
    key = (str(Path(source_root).resolve()) if source_root else None,
           tuple(prefixes), tuple(uri_bases))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = SourceCodeModel(source_root, prefixes=prefixes, uri_bases=uri_bases)
        model._is_shared = True
        _MODEL_CACHE[key] = model
    return model


def clear_model_cache() -> None:
    """Drop the shared models (tests, and long-lived processes)."""
    _MODEL_CACHE.clear()


__all__ = [
    "CodeReference",
    "KNOWN_BUILD_PREFIXES",
    "ResolvedFile",
    "STRONG_STRATEGIES",
    "SourceCodeModel",
    "SourceTreeIndex",
    "clear_index_cache",
    "clear_model_cache",
    "shared_model",
]
