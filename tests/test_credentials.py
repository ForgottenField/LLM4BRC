"""Credential file parsing tests."""

from __future__ import annotations

import json

import pytest

from llm_client.credentials import CredentialConfigError, load_api_key


def test_loads_provider_environment_key(tmp_path) -> None:
    path = tmp_path / "api_key.json"
    path.write_text(json.dumps({"DEEPSEEK_API_KEY": " test-key "}))
    assert load_api_key("deepseek", path) == "test-key"


def test_loads_nested_provider_key(tmp_path) -> None:
    path = tmp_path / "api_key.json"
    path.write_text(json.dumps({"claude": {"api_key": "claude-key"}}))
    assert load_api_key("claude", path) == "claude-key"


def test_missing_provider_key_does_not_echo_other_secrets(tmp_path) -> None:
    path = tmp_path / "api_key.json"
    path.write_text(json.dumps({"OPENAI_API_KEY": "openai-secret"}))

    with pytest.raises(CredentialConfigError) as exc:
        load_api_key("deepseek", path)

    message = str(exc.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "openai-secret" not in message


def test_invalid_json_has_actionable_error(tmp_path) -> None:
    path = tmp_path / "api_key.json"
    path.write_text("{invalid")

    with pytest.raises(CredentialConfigError, match="cannot read API key file"):
        load_api_key("deepseek", path)
