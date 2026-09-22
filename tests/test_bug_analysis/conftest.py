"""Shared fixtures for bug analysis tests."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sarif_report() -> str:
    return (FIXTURES / "sample_sarif.json").read_text()


@pytest.fixture
def plaintext_report() -> str:
    return (FIXTURES / "sample_plaintext.txt").read_text()


@pytest.fixture
def json_report() -> str:
    return (FIXTURES / "sample_json.json").read_text()
