"""Where the pipeline's files live, and the one logger every module shares.

The pipeline is an *application*, not a library: it reads and writes files at
well-known locations relative to the repository root (``pdg_<project>.json``,
``cfg_cache/<project>/``, ``output/``).  Those paths were computed inline in
three different functions, each carrying its own ``Path(__file__).parents[N]``
arithmetic — and the arithmetic changes when a module moves, silently.  So it is
computed exactly once, here, and asserted at import.

**The shared logger is deliberate, not an oversight.**  The repo convention is
``logging.getLogger(__name__)`` per module.  This package keeps one name,
``path_selection``, because the console prefix is part of the tool's output
(``logging.basicConfig(format="%(levelname)-5s %(name)s: %(message)s")`` is
configured by the entry script), and every existing log line reads
``path_selection: ...``.  Naming the logger after the module would rewrite every
prefix on the console.  If this is ever changed, it is a *visible* change to the
tool's output and should be made on purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path

#: The repository root.  ``src/llm_client/fp_analysis/pipeline/layout.py`` sits
#: four levels down (pipeline → fp_analysis → llm_client → src → root); the
#: older modules one level up use ``parents[3]``, so a module that moves into
#: this subpackage must not keep their index.
REPO_ROOT = Path(__file__).resolve().parents[4]

# A wrong index does not fail loudly on its own: it makes ``REPO_ROOT`` point at
# ``src/``, so ``pdg_<project>.json``, ``cfg_cache/`` and the default ``output/``
# all silently resolve to non-existent paths and the run degrades into "no CFG
# cache" warnings instead of an error.  Fail at import instead.
assert (REPO_ROOT / "run_path_selection.py").is_file(), (
    f"the repository root was computed as {REPO_ROOT}, which has no "
    f"run_path_selection.py — the parents[N] index in layout.py is wrong"
)

# Default base directory containing the project source trees + reports.  The
# reports/sources live under ~/csa_reports/project/<name>/ (NOT ./project, which
# is not present in this checkout).  Override with --projects-dir when needed.
DEFAULT_PROJECTS_DIR = Path.home() / "csa_reports" / "project"

#: The console logger's name — see the module docstring for why it is shared.
_LOGGER_NAME = "path_selection"


def get_logger() -> logging.Logger:
    """The pipeline's shared application logger (see the module docstring)."""
    return logging.getLogger(_LOGGER_NAME)
