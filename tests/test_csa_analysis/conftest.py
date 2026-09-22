"""Shared fixtures for CSA analysis tests."""

from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def buffer_overflow_plist() -> Path:
    return FIXTURES_DIR / "buffer_overflow.plist"


@pytest.fixture
def use_after_free_plist() -> Path:
    return FIXTURES_DIR / "use_after_free.plist"


@pytest.fixture
def unix_api_plist() -> Path:
    return FIXTURES_DIR / "unix_api.plist"


@pytest.fixture
def buffer_overflow_c() -> Path:
    return FIXTURES_DIR / "buffer_overflow.c"
