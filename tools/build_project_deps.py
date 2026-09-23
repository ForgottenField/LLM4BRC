#!/usr/bin/env python3.10
"""On-demand dependency-graph builder (PDG + CFG cache) for a CSA target project.

Why on-demand: later targets (protobuf, grpc, duckdb, ...) are large enough that
building a PDG for a whole source tree is wasteful — the pipeline only ever looks
at the files a CSA report walks and the functions those files may call. So the
seed set is derived from the reports themselves, then closed over the callees the
reports mention, and only that set of translation units is fed to the PDG builder.

What it produces (see ``docs/dependency-graph-build-guide.md`` §"按需构建"):

1. ``pdg_<project>.json``            — repo root, ``load_sdg`` reads this
2. ``cfg_cache/<project>/*.json``    — one full-TU CFG per seed TU
3. a compile database at ``/tmp/<project>_cc/compile_commands.json``

Both (2) and (3) are rebuildable artifacts; only the seed selection is knowledge.

Usage::

    # on-demand: seeds come from the reports' own event files
    python3.10 tools/build_project_deps.py --project protobuf --on-demand \
        ~/csa_reports/project/protobuf/reports

    # explicit TU list (no reports involved)
    python3.10 tools/build_project_deps.py --project foo --files a.cc b.cc

    # dry run: show the seed set / flags / compile-db, build nothing
    python3.10 tools/build_project_deps.py --project protobuf --on-demand <dir> --dry-run

Design notes / invariants:

* **Line alignment is verified, not assumed.** Report HTML embeds the source
  listing of the bug file, and the parser exposes it as ``source_snippets``. A
  fetched third-party tree (abseil, googletest) whose lines do not match the
  report's snapshot makes every later ``(file, line)`` resolution wrong — so the
  build refuses to include a seed TU below ``--min-line-match``.
* **A seed that cannot be resolved is reported, never substituted.** Same rule as
  ``run_path_selection._resolve_paths``: a report whose file we cannot find is
  listed as a gap, not silently pointed at another file.
* **Seeds are TUs, not functions.** The PDG builder walks whole translation
  units, so a header host (``repeated_ptr_field.h``) is built from the ``.cc``
  that the report names, and the cache lands under that ``.cc``'s hash.
* **One builder binary, and the artifact must say so.** ``build.sh`` produces
  ``build/pdg_builder`` only; the build refuses to start if that binary or the
  version declared in ``PDGBuilder.cpp`` disagrees with
  ``pdg_models.PDG_ARTIFACT_VERSION``, and the artifact is removed (not left
  behind) when the version it declares does not match.  A second, older binary
  (``pdg_builder_local``, v1.0) used to be named here and silently produced
  ``pdg_protobuf.json`` for an entire evaluation round.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_client.fp_analysis import cfg_feasibility as CF  # noqa: E402
from llm_client.fp_analysis import cfg_parser as CP  # noqa: E402
from llm_client.fp_analysis.html_parser import FPReportParser  # noqa: E402
from llm_client.fp_analysis.pdg_loader import read_pdg_artifact_version  # noqa: E402
from llm_client.fp_analysis.pdg_models import PDG_ARTIFACT_VERSION  # noqa: E402

#: The ONE builder binary — ``build.sh`` writes exactly this path, and the
#: artifact's declared version is checked against it below.  A second, older
#: copy (``pdg_builder_local``, built 8-21, version 1.0) used to be named here;
#: it produced ``pdg_protobuf.json`` for a whole 21-report evaluation.
PDG_BUILDER = REPO_ROOT / "src/llm_client/csa_analysis/pdg/build/pdg_builder"
PDG_BUILDER_SRC = REPO_ROOT / "src/llm_client/csa_analysis/pdg/PDGBuilder.cpp"
CXX = "clang++-14"
CXX_STD = "-std=c++17"

#: Prefixes CSA reports were generated under; stripped to get a project-relative path.
REPORT_PATH_PREFIXES = (
    "/home/me/projects/cpp-project/",
    "/tmp/fbcode_builder_getdeps-ZhomeZmeZprojectsZcpp-projectZfollyZbuildZfbcode_builder/",
)

#: Per-project build spec. ``include_paths`` are relative to the project root and
#: are the same values registered in ``cfg_parser._CFG_PROJECT_CONFIGS`` — keep the
#: two in sync (``--check-config`` warns when they drift).
PROJECT_SPECS: dict[str, dict] = {
    "folly": {
        # Mirrors cfg_parser._CFG_PROJECT_CONFIGS["folly"] — --check-config
        # compares the two and warns on drift.
        "root": "~/csa_reports/project/folly",
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
    "protobuf": {
        "root": "~/csa_reports/project/protobuf",
        "include_paths": [
            ".",
            "src",
            "third_party/abseil-cpp",
            "third_party/googletest/googletest/include",
            "third_party/googletest/googlemock/include",
        ],
        "defines": [],
        "force_includes": [],
    },
}

_HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".inc", ".tcc"}
_SOURCE_SUFFIXES = {".cc", ".cpp", ".cxx", ".c"}

#: ``Calling 'Foo'`` / ``Returning to 'Foo'`` steps name the callee on the path.
_CALLEE_RE = re.compile(r"(?:Calling|Entered call from|Returning to) '([^']+)'")

#: basename -> paths, per source root (see ``basename_index``).
_BASENAME_CACHE: dict[Path, dict[str, list[Path]]] = {}

#: Lines either side of the bug line compared when checking checkout alignment.
_ALIGN_WINDOW = 8

#: Shortest line text treated as evidence of alignment (``}`` proves nothing).
_ALIGN_MIN_LINE = 8

#: Fewer comparable lines than this and the report is not judged either way.
_ALIGN_MIN_PROBES = 3


# ─── data ─────────────────────────────────────────────────────────────────────


@dataclass
class ReportSeeds:
    """What one CSA report contributes to the seed set."""

    report: Path
    bug_file: str = ""
    bug_line: int = 0
    bug_function: str = ""
    local_file: Path | None = None
    event_files: set[Path] = field(default_factory=set)
    callees: set[str] = field(default_factory=set)
    line_match: float = 0.0
    line_offset: int = 0
    line_probe: int = 0
    problems: list[str] = field(default_factory=list)


# ─── path resolution ──────────────────────────────────────────────────────────


def resolve_source(raw: str, root: Path) -> Path | None:
    """Map a report's source path to a real file under *root*.

    Order matters: the prefix-stripped relative path is authoritative, the
    basename search is a last resort because protobuf/abseil both define
    same-named files (``numbers.cc``, ``time.cc``) in several directories.
    """
    if not raw:
        return None
    direct = Path(raw)
    if direct.is_absolute() and direct.exists():
        return direct.resolve()

    for prefix in REPORT_PATH_PREFIXES:
        if raw.startswith(prefix):
            tail = raw[len(prefix):]
            parts = tail.split("/")
            for cut in range(len(parts)):
                candidate = root / "/".join(parts[cut:])
                if candidate.exists():
                    return candidate.resolve()

    hits = basename_index(root).get(Path(raw).name, [])
    return hits[0] if hits else None


def basename_index(root: Path) -> dict[str, list[Path]]:
    """Lazy ``basename -> [paths]`` index of the source tree.

    Built once per root: the basename fallback runs per report event, and a
    filesystem walk per event over protobuf + abseil is minutes of wall clock.
    """
    cached = _BASENAME_CACHE.get(root)
    if cached is not None:
        return cached
    index: dict[str, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in filenames:
            index.setdefault(name, []).append(Path(dirpath) / name)
    for paths in index.values():
        paths.sort()
    _BASENAME_CACHE[root] = index
    return index


def verify_corpus_alignment(snippets: dict, files: list[Path], bug_line: int,
                            min_probes: int = _ALIGN_MIN_PROBES,
                            window: int = _ALIGN_WINDOW):
    """Best alignment across the files a report embeds.

    Report snippets are keyed by *section-local* line numbers, and a report
    shows one section per file — so key N must be read as "line N of whichever
    file that section came from", never as "line N of ``metadata.bug_file``"
    (on ``time_test.cc-TestBody-4-2`` the bug line 236 lands on ``time.cc``'s
    ``int month = 1;``, while ``bug_file`` says ``time_test.cc``). Checking every
    embedded file and keeping the best evidence is what makes this a revision
    check instead of a path-metadata check.

    Returns ``(file, ratio, offset, probes)``.
    """
    best = (None, 0.0, 0, 0)
    for path in dict.fromkeys(files):
        if path is None:
            continue
        ratio, offset, probes = verify_line_alignment(snippets, path, bug_line, window)
        if probes > best[3] or (probes == best[3] and ratio > best[1]):
            best = (path, ratio, offset, probes)
    file, ratio, offset, probes = best
    if probes < min_probes:
        return file, 1.0, 0, probes  # too little evidence to judge: do not reject
    return file, ratio, offset, probes


def verify_line_alignment(snippets: dict, local_file: Path, bug_line: int,
                          window: int = _ALIGN_WINDOW) -> tuple[float, int, int]:
    """How well the report's embedded source lines agree with *local_file*.

    Returns ``(best_ratio, offset, probes)`` where *offset* maximizes agreement
    over ``-WINDOW..+WINDOW``. **A non-zero offset is a failure, not a match**:
    the report's line numbers come from the revision CSA ran on, and the pipeline
    pairs reported lines with PDG nodes *by line number*, so a revision drift of
    even two lines mis-attributes every branch from that point on. The offset is
    reported so a wrong checkout shows up as "shift +2" rather than as noise.

    Only the window around the bug line is compared: report HTML shows one
    section per file, and a wide radius mixes lines from neighbouring files
    (which is why a whole-file comparison reads as a puzzling 85%).
    """
    if not snippets:
        return 1.0, 0, 0
    try:
        lines = local_file.read_text(errors="replace").splitlines()
    except OSError:
        return 0.0, 0, 0
    file_texts = {ln.strip() for ln in lines}

    want = {}
    for key, text in snippets.items():
        try:
            lineno = int(key)
        except (TypeError, ValueError):
            continue
        if abs(lineno - bug_line) > window:
            continue
        text = html_mod.unescape(str(text)).strip()
        # A report embeds one section per file, and sections are numbered
        # independently — the same ``data-linenumber`` can come from a
        # neighbouring file's section. Keep a line only when its text occurs
        # somewhere in *this* file; trivial lines (``}``, ``return;``) are
        # dropped because they prove nothing either way.
        if len(text) < _ALIGN_MIN_LINE or text not in file_texts:
            continue
        want[lineno] = text
    if not want:
        return 1.0, 0, 0

    best = (0.0, 0)
    for offset in range(-window, window + 1):
        hits = 0
        for lineno, text in want.items():
            local = lineno - offset
            if 1 <= local <= len(lines) and lines[local - 1].strip() == text:
                hits += 1
        ratio = hits / len(want)
        if ratio > best[0]:
            best = (ratio, offset)
    return best[0], best[1], len(want)


# ─── seed collection ──────────────────────────────────────────────────────────


def collect_seeds(report_paths: list[Path], root: Path) -> list[ReportSeeds]:
    parser = FPReportParser()
    seeds: list[ReportSeeds] = []
    for path in report_paths:
        parsed = parser.parse(str(path))
        rs = ReportSeeds(
            report=path,
            bug_file=parsed.metadata.bug_file or "",
            bug_line=int(parsed.metadata.bug_line or 0),
            bug_function=parsed.metadata.function_name or "",
        )
        if not rs.bug_file:
            rs.problems.append("report has no bug_file")
        for evt in parsed.path_events:
            raw = evt.get("file_path") or ""
            resolved = resolve_source(raw, root)
            if resolved:
                rs.event_files.add(resolved)
            elif raw:
                rs.problems.append(f"unresolved event file: {raw}")
            desc = evt.get("description") or ""
            rs.callees.update(_CALLEE_RE.findall(desc))
        rs.local_file = resolve_source(rs.bug_file, root)
        if rs.local_file is None and rs.bug_file:
            rs.problems.append(f"unresolved bug file: {rs.bug_file}")
        else:
            candidates = [rs.local_file, *sorted(rs.event_files)]
            _f, ratio, offset, probe = verify_corpus_alignment(
                parsed.source_snippets, candidates, rs.bug_line)
            rs.line_match, rs.line_offset, rs.line_probe = ratio, offset, probe
        seeds.append(rs)
    return seeds


def host_tu_for(rs: ReportSeeds, root: Path) -> Path | None:
    """The translation unit to build for a report.

    A header host (``gtest.h``, ``repeated_ptr_field.h``) carries no TU of its
    own; the report's own filename names the ``.cc`` that instantiates it, which
    is also what the CFG cache must be keyed on.
    """
    if rs.local_file and rs.local_file.suffix in _SOURCE_SUFFIXES:
        return rs.local_file
    stem = rs.report.name
    stem = stem[len("report-"):] if stem.startswith("report-") else stem
    named = stem.split("-")[0]
    if named.endswith(tuple(_SOURCE_SUFFIXES)):
        found = resolve_source(named, root)
        if found:
            return found
    # Fall back to any .cc event file: it is, by construction, the TU that
    # instantiates the header the report is about.
    for ev in sorted(rs.event_files):
        if ev.suffix in _SOURCE_SUFFIXES:
            return ev
    return None


#: Names that match a definition line in dozens of files; a match on one of
#: these says nothing about which TU owns the callee the report stepped into.
_CLOSURE_STOPWORDS = {
    "empty", "size", "begin", "end", "clear", "append", "swap", "delete",
    "copy", "new", "reset", "move", "insert", "erase", "find", "at", "data",
    "return", "release", "acquire", "init", "destroy", "close", "open", "read",
    "write", "resize", "reserve", "push", "pop", "front", "back", "get", "set",
}


def ctags_definitions(dirs: list[Path], log) -> dict[str, list[tuple[str, Path, int]]]:
    """``name -> [(qualified_name, file, line)]`` over *dirs*, via ctags.

    A definition index is what makes the closure precise: grepping for ``Name(``
    matches call sites, comments and same-named functions in unrelated TUs, which
    is how a "callee closure" turns into half the project (127 TUs on protobuf
    for 29 names). ctags reports real declarations, and the qualified name lets a
    ``Foo::Bar`` callee skip the dozen unrelated ``Bar``s.
    """
    index: dict[str, list[tuple[str, Path, int]]] = {}
    cmd = ["ctags", "-x", "--c++-kinds=f", "-f", "-", "-R",
           *[str(d) for d in dirs]]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log(f"    closure: ctags unavailable ({exc.__class__.__name__}) — skipped")
        return index
    for line in proc.stdout.splitlines():
        # ``-x`` columns: NAME KIND LINE FILE SIGNATURE...
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        qualified, _kind, lnum_s, tail = parts
        try:
            lnum = int(lnum_s)
            path = Path(tail.split()[0])
        except (IndexError, ValueError):
            continue
        if not path.is_file():
            continue
        index.setdefault(qualified.split("::")[-1], []).append(
            (qualified, path.resolve(), lnum))
    return index


def close_over_callees(tus: set[Path], callees: set[str], root: Path,
                       rounds: int, log, max_tus: int = 25,
                       per_name: int = 2) -> set[Path]:
    """Add the TUs that *define* callees the reports name but no seed TU covers.

    A callee defined in another TU (``time.cc`` called from ``time_test.cc``) has
    no PDG unless its own TU is built, and stage 5 expands exactly such callees.

    Conservative by construction: only real definitions (ctags), a stop-list for
    names defined everywhere, at most *per_name* files per callee and *max_tus*
    in total. An over-eager closure is worse than none — it grows the PDG and
    dilutes function attribution without adding reported-path coverage.
    """
    if rounds <= 0 or not callees:
        return set()
    candidates = {n for n in callees
                  if len(n.split("::")[-1]) >= 4
                  and n.split("::")[-1].lower() not in _CLOSURE_STOPWORDS}
    if not candidates:
        return set()

    search_dirs = [d for d in (root / "src", root / "third_party") if d.is_dir()]
    index = ctags_definitions(search_dirs, log)
    if not index:
        log("    closure: ctags produced no definitions — skipped")
        return set()
    log(f"    closure: {len(candidates)} callee name(s) against "
        f"{sum(len(v) for v in index.values())} definitions")

    known = {p.resolve() for p in tus}
    added: dict[Path, str] = {}
    for name in sorted(candidates):
        short = name.split("::")[-1]
        hits = index.get(short, [])
        if "::" in name:
            want = name.split("::")[-2]
            qualified = [h for h in hits if want in h[0]]
            hits = qualified or hits
        for qualified, path, lnum in hits[:per_name * 4]:
            if path.suffix not in _SOURCE_SUFFIXES:
                continue  # header definition — the including TU covers it
            if path in known or path in added:
                continue
            added[path] = f"{qualified}@L{lnum}"
            break
        if len(added) >= max_tus:
            log(f"    closure: cap reached ({max_tus} TU(s)) — stopping")
            break

    for path, why in added.items():
        rel = path.relative_to(root) if root in path.parents else path
        log(f"    + closure {rel}  ({why})")
    return set(added)


# ─── compile database ─────────────────────────────────────────────────────────


def spec_flags(spec: dict, root: Path) -> list[str]:
    flags = [CXX_STD]
    for rel in spec.get("include_paths", []):
        cand = (root / rel) if rel != "." else root
        if cand.exists():
            flags.append(f"-I{cand}")
    flags += [f"-D{d}" for d in spec.get("defines", [])]
    flags += [f"-include{fi}" for fi in spec.get("force_includes", [])]
    return flags


def validate_tu(path: Path, flags: list[str], timeout: int = 300) -> tuple[bool, str]:
    cmd = [CXX, *flags, "-fsyntax-only", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode == 0:
        return True, ""
    first = next((l for l in proc.stderr.splitlines() if "error" in l), "")
    return False, (first or proc.stderr.splitlines()[-1] if proc.stderr else "unknown")


def write_compile_db(tus: list[Path], flags: list[str], out_path: Path) -> int:
    entries = [
        {
            "directory": str(tu.parent),
            "file": str(tu),
            "command": " ".join([CXX, *flags, "-c", str(tu), "-o", "/dev/null"]),
        }
        for tu in tus
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    return len(entries)


# ─── build steps ──────────────────────────────────────────────────────────────


def declared_builder_version() -> str | None:
    """``Metadata["version"]`` as *declared in the source* — the wanted value.

    Anchored to the start of a code line on purpose: an unanchored search matches
    the *first* mention anywhere in the file, so a comment that merely quotes the
    assignment (``// see Metadata["version"] = "1.3" below``) becomes the answer —
    a check reading prose instead of the value it is supposed to verify.
    """
    try:
        src = PDG_BUILDER_SRC.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r'^\s*Metadata\["version"\]\s*=\s*"([^"]+)"', src, re.M)
    return m.group(1) if m else None


def check_builder(log) -> bool:
    """Refuse to build with a missing or stale builder binary.

    A stale binary is invisible otherwise: it runs, exits 0 and writes a
    perfectly parseable artifact.  ``pdg_protobuf.json`` was built that way
    (binary 8-21 / version 1.0 vs source 9-15 / version 1.2), so the protobuf
    corpus lost the macro-branch annotation and the postdominator-derived
    control dependencies without a single warning.  Two signals are checked
    here — the binary's mtime against the source's, and the version the source
    declares against the version this code expects — and the artifact's own
    declaration is checked after the build (``run_pdg_builder``).
    """
    ok = True
    if not PDG_BUILDER.exists():
        log(f"ERROR: PDG builder missing: {PDG_BUILDER}")
        log("       build it: bash src/llm_client/csa_analysis/pdg/build.sh")
        return False

    declared = declared_builder_version()
    if declared is None:
        log(f"ERROR: cannot read Metadata[\"version\"] from {PDG_BUILDER_SRC}")
        ok = False
    elif declared != PDG_ARTIFACT_VERSION:
        log(f"ERROR: version drift — {PDG_BUILDER_SRC.name} declares {declared!r}, "
            f"this code expects PDG_ARTIFACT_VERSION={PDG_ARTIFACT_VERSION!r}")
        log("       update llm_client/fp_analysis/pdg_models.PDG_ARTIFACT_VERSION "
            "in the same change")
        ok = False

    try:
        if PDG_BUILDER.stat().st_mtime < PDG_BUILDER_SRC.stat().st_mtime:
            log(f"WARNING: {PDG_BUILDER.name} is older than {PDG_BUILDER_SRC.name} "
                "— rebuild: bash src/llm_client/csa_analysis/pdg/build.sh "
                "(the declared version below will not catch a same-version source edit)")
    except OSError:
        pass

    log(f"   builder: {PDG_BUILDER.name} — source declares version {declared!r}, "
        f"this code expects {PDG_ARTIFACT_VERSION!r}")
    return ok


def verify_pdg_version(pdg_path: Path, log) -> bool:
    """Check the artifact the builder just wrote declares the expected version.

    Without this the tool could happily install an artifact produced by an
    older binary that happened to be on PATH — the failure mode this repo has
    already paid for once.  A mismatched artifact is *removed*: leaving it in
    place is exactly the "one wrong file, reused later" trap.
    """
    version = read_pdg_artifact_version(pdg_path)
    if version == PDG_ARTIFACT_VERSION:
        log(f"   artifact declares version {version!r} — OK")
        return True
    log(f"ERROR: {pdg_path.name} declares version {version!r}, expected "
        f"{PDG_ARTIFACT_VERSION!r} — the builder binary does not match its source")
    log("       rebuild: bash src/llm_client/csa_analysis/pdg/build.sh")
    try:
        pdg_path.unlink()
        log(f"       removed the unusable artifact: {pdg_path}")
    except OSError as exc:
        log(f"       WARNING: could not remove it: {exc}")
    return False


def check_artifacts(log) -> bool:
    """Report every ``pdg_<project>.json`` in the repo root against the constant.

    One artifact per project is the rule: a stale file left beside a current one
    is indistinguishable to every consumer except by this declaration, and the
    pipeline only refuses it at analysis time (``run_path_selection``), after a
    report has already been chosen.  Printing the inventory here makes the whole
    set visible in one command — a project with no rebuildable source (no compile
    db) still shows up as STALE rather than silently disappearing from view.
    """
    ok = True
    artifacts = sorted(REPO_ROOT.glob("pdg_*.json"))
    if not artifacts:
        log("   artifacts: none in the repo root")
        return ok
    for path in artifacts:
        project = path.stem[len("pdg_"):]
        version = read_pdg_artifact_version(path)
        if version == PDG_ARTIFACT_VERSION:
            state = "OK"
        elif version is None:
            state = "NONE (no metadata.version — cannot verify)"
            ok = False
        else:
            state = "STALE"
            ok = False
        size_mb = path.stat().st_size / 1e6
        log(f"   {path.name:<24s} declares {str(version):<6s} "
            f"({size_mb:6.1f} MB) — {state}")
        if state != "OK":
            registered = project in PROJECT_SPECS
            log(f"      rebuild: python3 tools/build_project_deps.py --project "
                f"{project} --on-demand ~/csa_reports/project/{project}/reports"
                + ("" if registered else "   [not in PROJECT_SPECS — needs a spec/compiledb]"))
    return ok


def _check_config(project: str | None) -> int:
    """``--check-config``: project pins, builder revision, and artifact versions.

    With no ``--project`` every registered spec is checked (and the artifact
    inventory is always printed), so the tooling half of a stale-data incident
    can be verified before any report is analysed.
    """
    log = print
    rc = 0
    for name in ([project] if project else sorted(PROJECT_SPECS)):
        spec = PROJECT_SPECS.get(name, {})
        registered = CP._CFG_PROJECT_CONFIGS.get(name, {})
        want = {k: spec.get(k, []) for k in ("include_paths", "defines", "force_includes")}
        if registered == want:
            log(f"{name}: PROJECT_SPECS matches cfg_parser._CFG_PROJECT_CONFIGS")
        else:
            if not spec:
                log(f"{name}: not in PROJECT_SPECS and not registered in cfg_parser")
            else:
                log(f"{name}: DRIFT\n  spec       = {want}\n  registered = {registered}")
            rc = 1
    if not check_builder(log):
        rc = 1
    if not check_artifacts(log):
        rc = 1
    return rc


def run_pdg_builder(compile_db: Path, out_path: Path, log) -> bool:
    cmd = [str(PDG_BUILDER), "--compile-db", str(compile_db),
           "--output", str(out_path), "--verbose"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    for line in tail:
        log(f"      {line}")
    if proc.returncode != 0 or not out_path.exists():
        return False
    return verify_pdg_version(out_path, log)


def preheat_cfg_cache(tus: list[Path], project: str, root: Path, log) -> tuple[int, list[str]]:
    ok = 0
    failures: list[str] = []
    for tu in tus:
        cache = CF._cfg_cache_path(str(tu), project)
        if cache.exists():
            ok += 1
            continue
        cmd = CP.build_cfg_command(str(tu), project, str(root))
        raw = CP.run_cfg_dump(cmd)
        if not raw:
            failures.append(f"{tu}: CFG dump produced nothing")
            continue
        fns = CP.parse_cfg_text(raw)
        if not fns:
            failures.append(f"{tu}: CFG dump parsed to 0 functions")
            continue
        CF._save_cfg_cache(cache, fns)
        ok += 1
    log(f"   CFG cache: {ok}/{len(tus)} present")
    for f in failures:
        log(f"   WARNING {f}")
    return ok, failures


def verify_pdg(pdg_path: Path, seeds: list[ReportSeeds], log) -> dict:
    data = json.loads(pdg_path.read_text())
    names = {f["function_name"] for f in data["functions"]}
    short = {n.split("::")[-1] for n in names}
    missing = [rs.bug_function for rs in seeds
               if rs.bug_function and rs.bug_function not in short]
    log(f"   PDG: {len(names)} functions, {pdg_path.stat().st_size / 1e6:.1f} MB")
    if missing:
        log(f"   WARNING {len(missing)} report host function(s) absent from PDG: "
            f"{sorted(set(missing))[:5]}")
    return {"functions": len(names), "missing_host_functions": sorted(set(missing))}


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project",
                    help="project name (required unless --check-config, which "
                         "checks every registered project when omitted)")
    ap.add_argument("--root", help="project source root (default: PROJECT_SPECS entry)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--on-demand", metavar="REPORTS_DIR", nargs="+",
                     help="derive the seed set from these reports (dirs or files)")
    src.add_argument("--files", nargs="+", help="explicit TU list (relative to --root)")
    ap.add_argument("--compile-db", help="where to write compile_commands.json")
    ap.add_argument("--output", help="PDG output path (default pdg_<project>.json)")
    ap.add_argument("--closure-rounds", type=int, default=1,
                    help="rounds of callee-TU closure to add (0 disables, default 1)")
    ap.add_argument("--closure-max-tus", type=int, default=25,
                    help="hard cap on TUs the callee closure may add (default 25)")
    ap.add_argument("--min-line-match", type=float, default=0.9,
                    help="refuse a seed whose report/checkout lines agree less (default 0.9)")
    ap.add_argument("--no-cfg", action="store_true", help="skip CFG cache preheat")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-config", action="store_true",
                    help="only check PROJECT_SPECS vs cfg_parser._CFG_PROJECT_CONFIGS "
                         "and the builder binary vs PDG_ARTIFACT_VERSION")
    args = ap.parse_args()

    if args.check_config:
        return _check_config(args.project)

    if not args.project:
        print("ERROR: --project is required (or use --check-config)", file=sys.stderr)
        return 2
    if not args.on_demand and not args.files:
        print("ERROR: one of --on-demand / --files is required (unless --check-config)")
        return 2

    spec = PROJECT_SPECS.get(args.project, {})
    root = Path(args.root or spec.get("root", "")).expanduser()
    flags = spec_flags(spec, root)

    def log(msg: str = "") -> None:
        print(msg, flush=True)

    if not root.is_dir():
        log(f"ERROR: source root not found: {root} (pass --root)")
        return 2

    # 1 — seeds
    seeds: list[ReportSeeds] = []
    if args.on_demand:
        reports: list[Path] = []
        for item in args.on_demand:
            p = Path(item).expanduser()
            reports.extend(sorted(p.rglob("*.html")) if p.is_dir() else [p])
        log(f"[1] seeds from {len(reports)} report(s)")
        seeds = collect_seeds(reports, root)
    else:
        tus = [resolve_source(f, root) for f in args.files]
        return _finish(args, spec, root, flags, [t for t in tus if t], [], log)

    tus: set[Path] = set()
    gaps: list[ReportSeeds] = []
    for rs in seeds:
        if rs.line_probe and rs.line_offset:
            rs.problems.append(
                f"checkout drift: report lines map onto the local file at "
                f"{rs.line_offset:+d} (best {rs.line_match:.0%} over "
                f"{rs.line_probe} lines)")
        elif rs.line_probe and rs.line_match < args.min_line_match:
            rs.problems.append(
                f"line mismatch {rs.line_match:.0%} over {rs.line_probe} lines "
                f"(< --min-line-match {args.min_line_match:.0%})")
        if rs.problems:
            gaps.append(rs)
            log(f"    GAP {rs.report.name}: {'; '.join(rs.problems)}")
            continue
        tu = host_tu_for(rs, root)
        if tu:
            tus.add(tu)
        else:
            rs.problems.append("no host TU")
            gaps.append(rs)
        tus |= {f for f in rs.event_files if f.suffix in _SOURCE_SUFFIXES}
    log(f"    {len(tus)} seed TU(s) from {len(seeds) - len(gaps)} usable report(s)")

    # 2 — callee closure
    callees: set[str] = set()
    for rs in seeds:
        callees |= rs.callees
    added = close_over_callees(tus, callees, root, args.closure_rounds, log,
                               max_tus=args.closure_max_tus)
    if added:
        log(f"[2] closure added {len(added)} TU(s) for {len(callees)} named callee(s)")
        tus |= added

    return _finish(args, spec, root, flags, sorted(tus), seeds, log, gaps)


def _finish(args, spec, root: Path, flags: list[str], tus: list[Path],
            seeds: list[ReportSeeds], log, gaps: list | None = None) -> int:
    summary: dict = {"project": args.project, "root": str(root),
                     "tus": [str(t) for t in tus],
                     "gaps": [{"report": g.report.name, "problems": g.problems}
                              for g in (gaps or [])]}
    log(f"[3] flags: {' '.join(flags)}")
    if not check_builder(log):
        log("ERROR: refusing to build with a mismatched builder binary")
        return 6
    for tu in tus:
        log(f"      TU {tu.relative_to(root) if root in tu.parents else tu}")

    if not tus:
        log("ERROR: no seed translation units — nothing to build")
        return 3

    # syntax gate: the PDG builder only sees TUs that actually parse
    usable: list[Path] = []
    for tu in tus:
        ok, err = validate_tu(tu, flags)
        if ok:
            usable.append(tu)
        else:
            log(f"      SKIP (does not parse): {tu} — {err[:160]}")
            summary.setdefault("unparseable", []).append(str(tu))
    log(f"[4] {len(usable)}/{len(tus)} TU(s) parse clean")
    if not usable:
        return 4
    summary["tus_parsed"] = [str(t) for t in usable]

    if args.dry_run:
        log("[dry-run] stopping before build")
        print(json.dumps(summary, indent=2))
        return 0

    compile_db = Path(args.compile_db or f"/tmp/{args.project}_cc/compile_commands.json")
    pdg_out = Path(args.output or REPO_ROOT / f"pdg_{args.project}.json")
    n = write_compile_db(usable, flags, compile_db)
    log(f"[5] compile db: {compile_db} ({n} entries)")

    if not run_pdg_builder(compile_db, pdg_out, log):
        log("ERROR: PDG build failed")
        return 5
    log(f"[6] PDG written: {pdg_out}")
    summary["pdg"] = verify_pdg(pdg_out, seeds, log)

    if not args.no_cfg:
        log("[7] CFG cache preheat")
        ok, failures = preheat_cfg_cache(usable, args.project, root, log)
        summary["cfg_cache"] = {"ok": ok, "total": len(usable), "failures": failures}

    log("[8] done")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
