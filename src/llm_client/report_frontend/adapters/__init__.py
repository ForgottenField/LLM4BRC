"""Report adapters, one module per container format.

**Adding an analyzer is a one-file change**: write a
:class:`~..registry.ReportAdapter` subclass, decorate it with
:class:`~..registry.register`, and import it below.  Nothing else in the
pipeline, and no other file in this package, needs to know the format exists —
``--report-format`` picks up the new registry key automatically and
``--list-findings``/``--scan-dir`` work through the canonical model.

Formats and the analyzers they serve:

======================  ==================================================
``csa-html``            Clang Static Analyzer (HTML output)
``sarif``               CodeQL, Infer (``--sarif``), Cppcheck
                        (``--output-format=sarif``) — anything emitting
                        SARIF 2.1.0; the analyzer is read off
                        ``tool.driver.name``
======================  ==================================================

Still native-only (add an adapter when they are needed): Infer's own
``report.json`` and Cppcheck's XML v2.  Both are narrower than the SARIF they
also emit, so the SARIF adapter usually covers them already.
"""

from __future__ import annotations

from .base import AdapterInfo  # noqa: F401  (re-exported for convenience)
from .csa_html import CsaHtmlAdapter  # noqa: F401
from .sarif import SarifAdapter  # noqa: F401


def load_adapters() -> None:
    """Ensure every adapter module has been imported (idempotent).

    Importing this package is what registers them, so this exists for callers
    that want to be explicit rather than relying on the import above — and to
    give a single place to mention the next adapter's module name.
    """
    return None


__all__ = ["AdapterInfo", "CsaHtmlAdapter", "SarifAdapter", "load_adapters"]
