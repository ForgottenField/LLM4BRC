"""Tests for ProviderFactory."""

import pytest
from llm_client.base import LLMConfig, LLMProvider, LLMRequest, LLMResponse
from llm_client.factory import ProviderFactory
from llm_client.openai_provider import OpenAIProvider
from llm_client.anthropic_provider import ClaudeProvider
from llm_client.deepseek_provider import DeepSeekProvider


class MockProvider(LLMProvider):
    """A minimal mock provider used for extensibility tests."""

    def send(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=f"Mock response to: {request.prompt}",
            model="mock-model",
            provider="mock",
        )


class TestProviderFactory:
    """ProviderFactory correctly maps configs to provider classes."""

    def test_create_gpt_provider(self, gpt_config):
        provider = ProviderFactory.create(gpt_config)
        assert isinstance(provider, OpenAIProvider)

    def test_create_claude_provider(self, claude_config):
        provider = ProviderFactory.create(claude_config)
        assert isinstance(provider, ClaudeProvider)

    def test_create_deepseek_provider(self, deepseek_config):
        provider = ProviderFactory.create(deepseek_config)
        assert isinstance(provider, DeepSeekProvider)

    def test_unknown_provider_raises(self, unknown_config):
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.create(unknown_config)

    def test_missing_api_key_raises(self):
        cfg = LLMConfig(provider_name="gpt", api_key="")
        with pytest.raises(ValueError, match="api_key is required"):
            ProviderFactory.create(cfg)

    def test_register_and_create_mock_provider(self):
        """Verify a new provider can be registered and used without modifying
        existing factory or provider code."""
        ProviderFactory.register_provider("mock", MockProvider)
        cfg = LLMConfig(provider_name="mock", api_key="sk-mock")
        provider = ProviderFactory.create(cfg)

        assert isinstance(provider, MockProvider)
        response = provider.send(LLMRequest(prompt="Hello"))
        assert response.content == "Mock response to: Hello"
        assert response.provider == "mock"
