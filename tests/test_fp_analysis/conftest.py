"""Shared fixtures for FP analysis tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def memory_test_report() -> str:
    """Path to MemoryTest FP report (crash in gtest)."""
    return str(FIXTURES_DIR / "report-MemoryTest.cpp-Destroy-2-1.html")


@pytest.fixture
def socket_address_report() -> str:
    """Path to SocketAddress FP report (crash in folly code)."""
    return str(
        Path(__file__).resolve().parent.parent.parent
        / "project"
        / "folly"
        / "reports"
        / "FP"
        / "report-SocketAddress.cpp-setFromLocalAddr-11-1.html"
    )
