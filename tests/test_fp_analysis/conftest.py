"""Shared fixtures for FP analysis tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Source corpora + their CSA reports live OUTSIDE the repo (see CLAUDE.md
# "Data artifacts"); ``run_path_selection.DEFAULT_PROJECTS_DIR`` is the same
# default.  Tests that need a corpus skip — rather than fail — without it, so
# the suite stays green on a machine that only has the small in-repo fixtures.
PROJECTS_DIR = Path(
    os.environ.get("CSA_PROJECTS_DIR") or Path.home() / "csa_reports" / "project"
)


@pytest.fixture(scope="session")
def projects_dir() -> Path:
    """Root of the source corpora (``CSA_PROJECTS_DIR`` overrides the default)."""
    return PROJECTS_DIR


@pytest.fixture(scope="session")
def wslay_lib(projects_dir: Path) -> Path:
    """aria2's bundled wslay sources — the corpus the source tests read."""
    path = projects_dir / "aria2" / "deps" / "wslay" / "lib"
    if not path.is_dir():
        pytest.skip(f"corpus not present: {path}")
    return path


@pytest.fixture
def memory_test_report() -> str:
    """Path to MemoryTest FP report (crash in gtest)."""
    return str(FIXTURES_DIR / "report-MemoryTest.cpp-Destroy-2-1.html")


@pytest.fixture
def socket_address_report() -> str:
    """Path to SocketAddress FP report (crash in folly code)."""
    path = (
        PROJECTS_DIR / "folly" / "reports" / "FP"
        / "report-SocketAddress.cpp-setFromLocalAddr-11-1.html"
    )
    if not path.exists():
        pytest.skip(f"corpus not present: {path}")
    return str(path)
