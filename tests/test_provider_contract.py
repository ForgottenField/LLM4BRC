"""Cross-provider contract tests for the canonical chat-message request."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_client.anthropic_provider import ClaudeProvider
from llm_client.base import LLMConfig, LLMRequest
from llm_client.config import provider_spec
from llm_client.openai_provider import OpenAIProvider


MESSAGES = [
    {"role": "system", "content": "system instructions"},
    {"role": "user", "content": "user prompt"},
]


def _openai_response():
    response = MagicMock()
    response.model = "gpt-test"
    response.usage = None
    response.choices = [MagicMock()]
    response.choices[0].message.content = "ok"
    response.choices[0].finish_reason = "stop"
    return response


def _anthropic_response():
    response = MagicMock()
    response.model = "claude-test"
    response.usage = None
    response.content = [MagicMock(text="ok")]
    response.stop_reason = "end_turn"
    return response


def test_llm_request_normalizes_legacy_prompt() -> None:
    request = LLMRequest(prompt="hello")
    assert request.normalized_messages() == [{"role": "user", "content": "hello"}]


def test_llm_request_prefers_messages() -> None:
    request = LLMRequest(prompt="ignored", messages=MESSAGES)
    assert request.normalized_messages() == MESSAGES


def test_provider_specs_have_distinct_environment_keys() -> None:
    keys = {
        provider_spec(name).api_key_env for name in ("deepseek", "gpt", "claude")
    }
    assert keys == {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}


@patch("llm_client.openai_provider.OpenAI")
def test_openai_provider_forwards_canonical_messages(mock_openai) -> None:
    mock_openai.return_value.chat.completions.create.return_value = _openai_response()
    provider = OpenAIProvider(LLMConfig(provider_name="gpt", api_key="test"))

    provider.send(LLMRequest(messages=MESSAGES, model="gpt-test"))

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == MESSAGES


@patch("llm_client.anthropic_provider.Anthropic")
def test_anthropic_provider_splits_system_and_chat_messages(mock_anthropic) -> None:
    mock_anthropic.return_value.messages.create.return_value = _anthropic_response()
    provider = ClaudeProvider(LLMConfig(provider_name="claude", api_key="test"))

    provider.send(LLMRequest(messages=MESSAGES, model="claude-test"))

    kwargs = mock_anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["system"] == "system instructions"
    assert kwargs["messages"] == [{"role": "user", "content": "user prompt"}]
