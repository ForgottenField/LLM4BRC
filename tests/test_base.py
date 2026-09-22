"""Contract compliance tests for the LLMProvider ABC."""

import pytest

from llm_client.base import LLMRequest, LLMResponse, LLMConfig, LLMProvider


class TestDataModels:
    """LLMRequest / LLMResponse / LLMConfig basic validation."""

    def test_llm_request_defaults(self):
        req = LLMRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.model == ""
        assert req.temperature == 0.0
        assert req.max_tokens == 1024
        assert req.timeout == 60.0

    def test_llm_response_creation(self):
        resp = LLMResponse(
            content="Hello back",
            model="gpt-4o",
            provider="gpt",
            usage={"prompt_tokens": 5, "completion_tokens": 5},
            response_time=0.42,
        )
        assert resp.content == "Hello back"
        assert resp.provider == "gpt"
        assert resp.usage["prompt_tokens"] == 5

    def test_llm_config_defaults(self):
        cfg = LLMConfig(provider_name="claude", api_key="sk-test")
        assert cfg.default_model == ""
        assert cfg.default_max_tokens == 1024
        assert cfg.default_timeout == 60.0


class TestProviderABC:
    """LLMProvider ABC enforces the correct interface."""

    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            LLMProvider(LLMConfig(provider_name="x", api_key="k"))  # type: ignore

    def test_concrete_provider_must_implement_send(self):
        class Incomplete(LLMProvider):
            pass

        with pytest.raises(TypeError):
            Incomplete(LLMConfig(provider_name="x", api_key="k"))  # type: ignore
