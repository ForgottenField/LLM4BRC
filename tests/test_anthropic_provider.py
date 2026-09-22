"""Tests for ClaudeProvider (Anthropic)."""

from unittest.mock import patch

import pytest

from llm_client.base import LLMRequest
from llm_client.errors import AuthenticationError, RateLimitError, TimeoutError
from llm_client.factory import ProviderFactory
from tests.conftest import (
    mock_anthropic_response,
    make_anthropic_auth_error,
    make_anthropic_rate_limit_error,
    make_anthropic_timeout_error,
)


class TestClaudeProviderHappyPath:
    """Happy-path tests for ClaudeProvider."""

    @patch("llm_client.anthropic_provider.Anthropic")
    def test_send_returns_response(self, MockAnthropic, llm_request, claude_config):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = mock_anthropic_response(
            content="This is a Claude response.", model="claude-sonnet-4-20250514"
        )

        provider = ProviderFactory.create(claude_config)
        response = provider.send(llm_request)

        assert response.content == "This is a Claude response."
        assert response.provider == "claude"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.usage is not None

    @patch("llm_client.anthropic_provider.Anthropic")
    def test_send_with_custom_params(self, MockAnthropic, claude_config):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = mock_anthropic_response(
            model="claude-opus-4-20250514"
        )

        provider = ProviderFactory.create(claude_config)
        req = LLMRequest(prompt="Analyze this", model="claude-opus-4-20250514", max_tokens=256)
        response = provider.send(req)

        assert response.content == "Mock Claude response"
        assert response.model == "claude-opus-4-20250514"


class TestClaudeProviderErrors:
    """Error-handling tests for ClaudeProvider."""

    @patch("llm_client.anthropic_provider.Anthropic")
    def test_authentication_error(self, MockAnthropic, llm_request, claude_config):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = make_anthropic_auth_error()

        provider = ProviderFactory.create(claude_config)
        with pytest.raises(AuthenticationError):
            provider.send(llm_request)

    @patch("llm_client.anthropic_provider.Anthropic")
    def test_rate_limit_error(self, MockAnthropic, llm_request, claude_config):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = make_anthropic_rate_limit_error()

        provider = ProviderFactory.create(claude_config)
        with pytest.raises(RateLimitError):
            provider.send(llm_request)

    @patch("llm_client.anthropic_provider.Anthropic")
    def test_timeout_error(self, MockAnthropic, llm_request, claude_config):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = make_anthropic_timeout_error()

        provider = ProviderFactory.create(claude_config)
        with pytest.raises(TimeoutError):
            provider.send(llm_request)
