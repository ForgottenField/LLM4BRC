"""Tests for DeepSeekProvider."""

from unittest.mock import patch

import pytest

from llm_client.base import LLMRequest
from llm_client.errors import AuthenticationError, RateLimitError, TimeoutError
from llm_client.factory import ProviderFactory
from tests.conftest import (
    mock_openai_response,
    make_openai_auth_error,
    make_openai_rate_limit_error,
    make_openai_timeout_error,
)


class TestDeepSeekProviderHappyPath:
    """Happy-path tests for DeepSeekProvider."""

    @patch("llm_client.deepseek_provider.OpenAI")
    def test_send_returns_response(self, MockOpenAI, llm_request, deepseek_config):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_openai_response(
            content="This is a DeepSeek response.", model="deepseek-chat"
        )

        provider = ProviderFactory.create(deepseek_config)
        response = provider.send(llm_request)

        assert response.content == "This is a DeepSeek response."
        assert response.provider == "deepseek"
        assert response.model == "deepseek-chat"
        assert response.usage is not None

    @patch("llm_client.deepseek_provider.OpenAI")
    def test_send_custom_model(self, MockOpenAI, deepseek_config):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_openai_response(
            model="deepseek-reasoner"
        )

        provider = ProviderFactory.create(deepseek_config)
        req = LLMRequest(prompt="Hello", model="deepseek-reasoner", max_tokens=128)
        response = provider.send(req)

        assert response.content == "Mock GPT response"
        assert response.model == "deepseek-reasoner"


class TestDeepSeekProviderErrors:
    """Error-handling tests for DeepSeekProvider."""

    @patch("llm_client.deepseek_provider.OpenAI")
    def test_authentication_error(self, MockOpenAI, llm_request, deepseek_config):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = make_openai_auth_error()

        provider = ProviderFactory.create(deepseek_config)
        with pytest.raises(AuthenticationError):
            provider.send(llm_request)

    @patch("llm_client.deepseek_provider.OpenAI")
    def test_rate_limit_error(self, MockOpenAI, llm_request, deepseek_config):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = make_openai_rate_limit_error()

        provider = ProviderFactory.create(deepseek_config)
        with pytest.raises(RateLimitError):
            provider.send(llm_request)

    @patch("llm_client.deepseek_provider.OpenAI")
    def test_timeout_error(self, MockOpenAI, llm_request, deepseek_config):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = make_openai_timeout_error()

        provider = ProviderFactory.create(deepseek_config)
        with pytest.raises(TimeoutError):
            provider.send(llm_request)
