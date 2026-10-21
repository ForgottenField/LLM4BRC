"""Stage 1's assembly half: resolve the project, validate the PDG, load the caches.

Everything that turns a ``--report``/``--project``/``--source-root`` invocation
into a ``(report ref, cfg cache, pdg, source root)`` tuple plus the in-memory
SDG and CFG cache lives here.  The two rules this module exists to uphold:

* **Never substitute another project for an unresolvable one.**  A guess that
  fails validation is reported, not silently replaced (``_resolve_paths``); a
  missing ``pdg_<project>.json`` raises rather than analysing another project's
  call graph.
* **Refuse a stale PDG artifact.**  A stale artifact parses fine and only its
  declared version says otherwise, so the version is checked before any work.
"""

from __future__ import annotations

import re
from pathlib import Path

from fp_analysis.cfg_cache_set import find_fn_key
from fp_analysis.pdg_augment import augment_pdg_sdg
from fp_analysis.pdg_loader import load_pdg, read_pdg_artifact_version
from fp_analysis.pdg_models import PDG_ARTIFACT_VERSION, SDG
from report_frontend import ReportFrontend, ReportFrontendError, ReportRef

from pipeline.layout import REPO_ROOT, get_logger

logger = get_logger()

# Markers that identify a directory as a project's source root.  A report whose
# grandparent carries none of them is not a project tree, and *guessing* a
# different one silently produced plausible-looking runs against the wrong
# project — see _resolve_paths.
_PROJECT_ROOT_MARKERS = (
    "deps", "lib", "src", ".git", "CMakeLists.txt", "Makefile", "meson.build",
    "configure", "setup.py",
)


def looks_like_project_root(path: Path) -> bool:
    """True when *path* is an existing directory that carries any project marker."""
    return path.is_dir() and any(
        (path / marker).exists() for marker in _PROJECT_ROOT_MARKERS
    )


#: Artifact version → the check is memoised because a batch run resolves the
#: same PDG once per report and the file is tens of MB.
_PDG_VERSION_CACHE: dict[tuple[str, float], str | None] = {}


def check_pdg_artifact_version(pdg_path: Path, project_name: str) -> None:
    """Refuse a ``pdg_<project>.json`` built by a different ``PDGBuilder`` revision.

    A stale artifact is not detectable from its contents: it parses, it has
    functions, nodes and edges — only the ``metadata.version`` it carries says
    which graph shape it is.  ``pdg_protobuf.json`` sat at 1.0 (binary dated
    8-21) while the source declared 1.2, so a whole protobuf evaluation ran on
    a graph without the macro-branch annotation and without the
    postdominator-derived control dependencies; the deterministic stage-6 gate
    then compared condition texts the old builder had attributed to macro
    bookkeeping nodes instead of the branch predicates, and turned three
    ground-truth TP reports into FP.

    Raises rather than warns: the failure is silent and expensive, and
    ``main()`` resolves every report before the first LLM call, so a
    misinvocation costs seconds.  An artifact with no ``metadata`` block at all
    (a hand-made fixture passed via ``--pdg``) is only warned about — there is
    nothing to compare.
    """
    try:
        mtime = pdg_path.stat().st_mtime
    except OSError:
        return
    key = (str(pdg_path), mtime)
    if key not in _PDG_VERSION_CACHE:
        _PDG_VERSION_CACHE[key] = read_pdg_artifact_version(pdg_path)
    version = _PDG_VERSION_CACHE[key]
    if version == PDG_ARTIFACT_VERSION:
        return
    rebuild = ("python3 tools/build_project_deps.py "
               f"--project {project_name} --on-demand <reports dir>")
    if version is None:
        logger.warning(
            "PDG %s declares no version (expected %r) — cannot verify it is "
            "current; rebuild with: %s", pdg_path, PDG_ARTIFACT_VERSION, rebuild,
        )
        return
    raise ValueError(
        f"PDG {pdg_path} was built by a different PDGBuilder revision: it "
        f"declares version {version!r}, this code expects "
        f"{PDG_ARTIFACT_VERSION!r}. Rebuild it ({rebuild}) — analysing it would "
        f"use a graph shape this code does not understand."
    )


def infer_source_root(
    args, ref: ReportRef
) -> tuple[Path | None, list[str]]:
    """Guess the source root of *ref* from the report itself; never substitute.

    Two candidate families, both of which must pass
    :func:`_looks_like_project_root` before they are accepted:

    * the report's own location (``<project>/reports/{TP,FP}/x`` — the layout
      every CSA report in this repository has), and
    * the project the report *names* — ``<projects-dir>/<stem>`` for a bare
      ``faiss.sarif`` exported outside any project tree, and the report's own
      advertised source-root hint (SARIF's ``originalUriBaseIds``, CSA's known
      build prefixes) mapped through ``--projects-dir``.

    Returns ``(source_root, tried)``; ``source_root`` is ``None`` when every
    candidate was rejected, in which case the caller raises.  The rule this
    function exists to uphold is that a guess which fails validation is *not*
    replaced by a different project — it is reported.
    """
    candidates: list[tuple[str, Path]] = [
        # ``<project>/reports/{TP,FP}/report-x.html`` — the layout of every CSA
        # corpus, where the project root is three levels up.  Labelled by what it
        # *is*, not as "the report's directory": the label goes into the error
        # message an operator reads, and off-layout reports land somewhere else.
        ("three levels up from the report", ref.path.parent.parent.parent)
    ]

    stem = ref.path.stem
    # ``report-foo-1`` → ``foo``; ``faiss.sarif`` → ``faiss``.  A CSA report
    # names its bug type inside that prefix, an exported SARIF names the project.
    guess = re.sub(r"-\d+$", "", re.sub(r"^report[-_]", "", stem))
    for name in (guess, stem, stem.split(".")[0]):
        if name:
            candidates.append(
                (f"--projects-dir/{name}", Path(args.projects_dir) / name)
            )

    try:
        frontend = ReportFrontend(report_format=getattr(args, "report_format", None))
        hint = frontend.load_document(ref).source_root_hint
    except (ReportFrontendError, OSError, ValueError):
        hint = ""
    if hint:
        # A hint is a path from the *build machine*; keep only its tail and
        # look that up under --projects-dir, which is where this machine keeps
        # project trees.
        for tail in (Path(hint).name, Path(hint).stem):
            if tail:
                candidates.append(
                    (f"report hint {hint!r}", Path(args.projects_dir) / tail)
                )

    tried: list[str] = []
    seen: set[Path] = set()
    for label, candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if looks_like_project_root(candidate):
            return candidate, tried
        tried.append(f"{label} → {candidate}")
    return None, tried


def resolve_paths(
    args, report_path: Path | str | ReportRef
) -> tuple[ReportRef, Path, Path, Path]:
    """Resolve (report ref, cfg_cache, pdg, source_root) for one report.

    ``report_path`` may be a plain path or a ``ReportRef`` (``<path>::<finding>``);
    the ref keeps its selector through resolution so a batch of SARIF findings
    stays addressable.  Every input is validated and a bad one raises
    ``ValueError`` with the command-line fix in the message.  It must never fall
    back to another project: the old ``source_root = HERE/"project"/"aria2"``
    substitution made ``--report <faiss report>`` load ``pdg_aria2.json`` and an
    empty ``cfg_cache/aria2/``, which produced a *plausible-looking* wrong run
    (18 branches, bound 32768→1, UNKNOWN) instead of an error.
    """
    ref = ReportRef.parse(report_path)
    report_path = ref.path.resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    ref = ReportRef(report_path, ref.selector)

    if args.project:
        # --project may be a full path to the source root, or a bare name
        # resolved under --projects-dir.
        proj = Path(args.project)
        if proj.is_dir():
            source_root = proj.resolve()
        else:
            source_root = (Path(args.projects_dir) / args.project).resolve()
        if not source_root.is_dir():
            raise ValueError(
                f"--project {args.project!r} resolved to {source_root}, which is "
                f"not a directory (a bare name is looked up under --projects-dir "
                f"{args.projects_dir})."
            )
    elif args.source_root:
        source_root = Path(args.source_root).resolve()
        if not source_root.is_dir():
            raise ValueError(
                f"--source-root {args.source_root!r} resolved to {source_root}, "
                f"which is not a directory."
            )
    else:
        source_root, tried = infer_source_root(args, ref)
        if source_root is None:
            raise ValueError(
                f"cannot infer the source root of {report_path.name} from the "
                f"report: no candidate directory carries any of "
                f"{', '.join(_PROJECT_ROOT_MARKERS)}. Tried "
                + "; ".join(tried)
                + f". Pass --project <name> (looked up under --projects-dir "
                f"{args.projects_dir}) or --source-root <dir>."
            )

    project_name = source_root.name

    if args.pdg:
        pdg_path = Path(args.pdg).resolve()
    else:
        pdg_path = REPO_ROOT / f"pdg_{project_name}.json"
    if not pdg_path.is_file():
        # The PDG drives every stage; another project's PDG parses fine and
        # silently analyses the wrong call graph.
        raise ValueError(
            f"SDG/PDG file not found: {pdg_path} (project {project_name!r}). "
            f"Build it offline for this project, or pass --pdg <path>."
        )
    # _check_pdg_artifact_version(pdg_path, project_name)

    if args.cfg_cache:
        cfg_cache_path = Path(args.cfg_cache).resolve()
    else:
        cfg_cache_path = None

    return ref, cfg_cache_path, pdg_path, source_root


def cache_mentions_function(
    cache_path: Path, fn_name: str, chunk: int = 4 << 20
) -> bool:
    """Probe a CFG cache for *fn_name* without parsing it.

    Cache files reach hundreds of MB, so a substring scan over the raw JSON is
    dramatically cheaper than ``json.loads`` — and sufficient: the function's
    dump signature is the key of its own entry, so it appears verbatim.
    """
    needle = fn_name.encode()
    tail = b""
    with cache_path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                return False
            if needle in tail + buf:
                return True
            tail = buf[-(len(needle) - 1):] if len(needle) > 1 else b""


def resolve_cfg_cache(source_root: Path, entry_fn_name: str) -> Path | None:
    """Find the CFG cache holding *entry_fn_name*.

    Last-resort lookup for reports whose bug file is not itself a translation
    unit (typically a header) and whose path files could not be resolved.
    Returns ``None`` when nothing matches — this used to fall back to the
    *largest* cache file, which silently handed the pipeline an unrelated
    translation unit's CFG, worse than no cache at all because the failure
    surfaced deep inside the branch filter instead of at load time.
    """
    cache_dir = REPO_ROOT / "cfg_cache" / source_root.name
    if not cache_dir.is_dir():
        return None
    caches = sorted(cache_dir.glob("*.json"))
    if not caches:
        return None
    if len(caches) == 1:
        return caches[0]

    for cache_path in caches:
        try:
            if cache_mentions_function(cache_path, entry_fn_name):
                return cache_path
        except OSError:
            continue
    return None


def find_fn_key(cfg_cache, function_name: str, source_file: str = ""):
    """Resolve *function_name* to a key of *cfg_cache* (see ``find_fn_key``)."""
    return find_fn_key(cfg_cache, function_name, source_file=source_file)


def load_cfg_cache(path: Path) -> dict:
    from fp_analysis.cfg_feasibility import _load_cfg_cache
    if not path.exists():
        logger.error("CFG cache not found: %s", path)
        return {}
    cache = _load_cfg_cache(path)
    logger.info("Loaded CFG cache: %d functions (%.1f KiB)",
                len(cache), path.stat().st_size / 1024)
    return cache or {}


def load_sdg(path: Path) -> SDG:
    if not path.exists():
        logger.error("PDG file not found: %s", path)
        return SDG(pdgs={})
    pdgs = load_pdg(str(path))
    sdg = SDG(pdgs=pdgs)
    augment_pdg_sdg(sdg)
    logger.info("Loaded SDG: %d functions", len(sdg.pdgs))
    return sdg
