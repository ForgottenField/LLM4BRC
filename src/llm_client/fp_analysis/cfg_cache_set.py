"""Multi-translation-unit view over the per-file CFG caches.

The pipeline's CFG cache is keyed per *translation unit*: one JSON file per
source file that was dumped (``cfg_cache/<project>/<md5(absolute source)>.json``).
A CSA bug path, however, routinely crosses files, so loading a single TU is not
enough — functions living in any other file fail to resolve and their segments
are dropped before branch extraction ever runs.

:class:`CFGCacheSet` loads exactly the TUs a path needs, on demand, and presents
them as one merged ``{signature: CFGFunction}`` mapping — the shape
``CFGBranchExtractor`` / ``ConflictChecker`` already consume (their lookups are
plain dict scans).

Notes
-----
* Merge order is deterministic (first loaded wins) and driven by the caller:
  bug-host TU first, then each segment's file in path order.
* ``raw_dump`` is dropped at load time; see ``_load_cfg_cache``.
* Same-named functions in different TUs are a real hazard in a merged dict, so
  each key keeps its owning file (:meth:`CFGCacheSet.key_source_file`) and
  :func:`find_fn_key` prefers the key belonging to the requested file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

from llm_client.fp_analysis.cfg_feasibility import (
    _cfg_cache_path,
    _load_cfg_cache,
    _resolve_local_path,
)
from llm_client.fp_analysis.cfg_models import CFGFunction

logger = logging.getLogger(__name__)

# Repo root — the cache dir is relative to CWD in ``_cfg_cache_path``; anchor
# lookups there too so behaviour does not depend on the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Extensions that can be a translation unit on their own.  Everything else
# (headers, system includes) can appear in a report's file sections but can
# never have its own cache, so those are summarized instead of warned about
# one by one.
_TU_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".c++", ".cppm"})


def _strip_template_args(spec: str) -> str:
    """Remove balanced ``<...>`` groups from a declarator tail.

    ``"heap_push<faiss::CMax<float, long>>"`` → ``"heap_push"``

    A template argument list can contain spaces, commas and nested angle
    brackets, so naive whitespace splitting reports the *last* template
    argument as the function name (``"long>>"``).  Groups are removed from the
    right until none is left; unbalanced text is returned unchanged.
    """
    out = spec
    while True:
        end = out.rfind(">")
        if end == -1:
            return out
        depth = 0
        start = -1
        for i in range(end, -1, -1):
            if out[i] == ">":
                depth += 1
            elif out[i] == "<":
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start == -1:                 # unbalanced → not a template list
            return out
        out = out[:start] + out[end + 1:]


def _method_component(name: str) -> str:
    """Reduce a C++ signature to its bare method name.

    ``"void SocketAddress::setFromLocalAddr(int)"`` → ``"setFromLocalAddr"``
    ``"template<> inline void heap_push<T>(...)"``  → ``"heap_push"``
    ``"faiss::Index *read_index(IOReader *, int)"`` → ``"read_index"``

    The pointer/reference marker of the *return type* must be stripped too:
    without that, every ``T *f(...)`` overload keeps the star (``"*read_index"``),
    the exact-match stage of :func:`find_fn_key` finds nothing, and the
    substring fallback silently picks an unrelated function whose name merely
    starts with the query (``read_index`` → ``read_index_header``).
    """
    base = name.split("(")[0].strip() if "(" in name else name.strip()
    base = _strip_template_args(base)
    tokens = base.split()
    last = tokens[-1] if tokens else base
    last = last.split("::")[-1]
    return last.lstrip("*&").strip()


def find_fn_key(
    cfg_cache: dict,
    function_name: str,
    source_file: str = "",
    key_source: Callable[[str], str] | None = None,
):
    """Resolve *function_name* to a key of *cfg_cache*.

    Prefers an EXACT method-component match over a plain substring: a bare
    substring such as ``setFromLocalAddr`` would otherwise match
    ``void SocketAddress::setFromLocalAddress(...)`` (``setFromLocalAddr`` ⊂
    ``setFromLocalAddress``) and extract the wrong overload's branches.

    When *source_file* is given and *key_source* can tell which file owns each
    key, keys from that file are preferred — the merged cache can hold the same
    function from several TUs.
    """
    if not function_name:
        return None
    if function_name in cfg_cache and _key_matches_file(
        function_name, source_file, key_source
    ):
        return function_name

    short = _method_component(function_name)

    def _prefer_file(keys: Iterable[str]):
        keys = list(keys)
        if len(keys) <= 1 or not source_file or key_source is None:
            return keys[0] if keys else None
        want = str(Path(source_file).resolve())
        for k in keys:
            if str(Path(key_source(k) or "").resolve()) == want:
                return k
        return keys[0]

    exact = [k for k in cfg_cache if _method_component(k) == short]
    if exact:
        return _prefer_file(exact)

    # Substring fallback (overloads / signature variants without an exact
    # method-component match).  Rank the candidates instead of taking the first
    # dict hit: dict order is insertion order, so ``read_index`` used to land on
    # ``read_index_header`` purely because the header function was dumped first.
    # Prefer, in order: same file, shortest method component (closest to the
    # query), then lexicographic — deterministic and never silently arbitrary.
    substr = [k for k in cfg_cache if function_name in k]
    if substr:
        want = str(Path(source_file).resolve()) if source_file else ""
        substr.sort(key=lambda k: (
            0 if (key_source and want
                  and str(Path(key_source(k) or "").resolve()) == want) else 1,
            len(_method_component(k)),
            k,
        ))
        pick = substr[0]
        logger.warning(
            "no CFG key with method name %r — falling back to %r (method %r; "
            "%d candidate(s)); the segment's own PDG (file+lines) is the "
            "authoritative source",
            short, pick, _method_component(pick), len(substr),
        )
        return pick
    return None


def _key_matches_file(
    key: str, source_file: str, key_source: Callable[[str], str] | None
) -> bool:
    if not source_file or key_source is None:
        return True
    owner = key_source(key)
    if not owner:
        return True
    return str(Path(owner).resolve()) == str(Path(source_file).resolve())


class CFGCacheSet:
    """Lazily loaded, merged view over the CFG caches of one project."""

    def __init__(self, project_name: str, source_root: str | None = None):
        self.project_name = project_name
        self.source_root = source_root
        self._merged: dict[str, CFGFunction] = {}
        self._key_source: dict[str, str] = {}
        self._loaded: list[str] = []            # local files successfully loaded
        self._attempted: set[str] = set()       # local files already looked up
        self._missing: list[str] = []           # looked up, no cache on disk

    # ── Loading ──────────────────────────────────────────────────────────

    def ensure(self, source_file: str | Path | None) -> bool:
        """Load *source_file*'s CFG cache if it has not been loaded yet.

        Returns True when that file's functions are in the merged view.
        """
        if not source_file:
            return False
        local = _resolve_local_path(source_file, self.source_root)
        if local is None:
            local = Path(source_file)
        local_key = str(local)
        if local_key in self._attempted:
            return local_key in self._loaded
        self._attempted.add(local_key)

        cache_path = self._find_cache_path(local)
        if cache_path is None:
            self._missing.append(local_key)
            return False

        data = _load_cfg_cache(cache_path)
        if not data:
            self._missing.append(local_key)
            return False

        added = 0
        for signature, fn_cfg in data.items():
            if signature in self._merged:
                continue                    # first TU wins — deterministic
            self._merged[signature] = fn_cfg
            self._key_source[signature] = local_key
            added += 1
        self._loaded.append(local_key)
        logger.debug(
            "CFG cache %s: +%d functions (%d merged)",
            local.name, added, len(self._merged),
        )
        return True

    def add_cache_file(self, cache_path: str | Path) -> int:
        """Load an explicitly specified cache file (``--cfg-cache``).

        Returns the number of functions added.
        """
        cache_path = Path(cache_path)
        data = _load_cfg_cache(cache_path)
        if not data:
            logger.error("CFG cache not found or empty: %s", cache_path)
            return 0
        added = 0
        # Attribute the entries to the cache's own source file, when that can
        # be recovered from the cache directory layout (md5 filename).
        owner = ""
        for signature, fn_cfg in data.items():
            if signature in self._merged:
                continue
            self._merged[signature] = fn_cfg
            self._key_source[signature] = owner
            added += 1
        logger.info(
            "Loaded CFG cache %s: %d functions (%d kB)",
            cache_path, added, cache_path.stat().st_size // 1024,
        )
        return added

    def _find_cache_path(self, local_source: Path) -> Path | None:
        relative = _cfg_cache_path(str(local_source), self.project_name)
        for candidate in (relative, _REPO_ROOT / relative):
            if candidate.exists():
                return candidate
        return None

    # ── Views ────────────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, CFGFunction]:
        """The live merged mapping (mutations from later ``ensure`` calls are
        visible, so callers may hold on to it)."""
        return self._merged

    def find_key(self, function_name: str, source_file: str = ""):
        """Resolve a function name to a merged-cache key."""
        return find_fn_key(
            self._merged, function_name,
            source_file=source_file,
            key_source=self._key_source.get,
        )

    def key_source_file(self, key: str) -> str:
        return self._key_source.get(key, "")

    @property
    def loaded_files(self) -> list[str]:
        return list(self._loaded)

    @property
    def missing_files(self) -> list[str]:
        return list(self._missing)

    def warn_missing(self) -> None:
        """Report files that have no CFG cache, without flooding the log."""
        by_suffix: dict[str, list[str]] = {}
        for path in self._missing:
            suffix = Path(path).suffix.lower()
            by_suffix.setdefault(suffix, []).append(path)
        for suffix, paths in sorted(by_suffix.items()):
            if suffix in _TU_SUFFIXES:
                for path in paths:
                    logger.warning(
                        "no CFG cache for %s — its segments fall back to "
                        "linear (preheat the cache offline to cover it)", path,
                    )
            elif suffix:
                logger.debug(
                    "%d non-TU file(s) (%s) have no CFG cache — expected for "
                    "headers/includes", len(paths), suffix,
                )
