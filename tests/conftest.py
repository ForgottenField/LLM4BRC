"""Shared test fixtures for LLM client tests."""

from unittest.mock import MagicMock

import pytest

from llm_client.base import LLMConfig, LLMRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_request() -> LLMRequest:
    return LLMRequest(prompt="Test prompt", max_tokens=64)


@pytest.fixture
def gpt_config() -> LLMConfig:
    return LLMConfig(provider_name="gpt", api_key="sk-gpt-test")


@pytest.fixture
def claude_config() -> LLMConfig:
    return LLMConfig(provider_name="claude", api_key="sk-ant-test")


@pytest.fixture
def deepseek_config() -> LLMConfig:
    return LLMConfig(provider_name="deepseek", api_key="sk-deepseek-test")


@pytest.fixture
def unknown_config() -> LLMConfig:
    return LLMConfig(provider_name="unknown", api_key="sk-test")


# ---------------------------------------------------------------------------
# Mock response helpers
# ---------------------------------------------------------------------------


def mock_openai_response(content: str = "Mock GPT response", model: str = "gpt-4o"):
    """Build a MagicMock that mimics an OpenAI chat completion response."""
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    choice = MagicMock()
    choice.message.content = content

    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


def mock_anthropic_response(
    content: str = "Mock Claude response", model: str = "claude-sonnet-4-20250514"
):
    """Build a MagicMock that mimics an Anthropic message response."""
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = content

    resp = MagicMock()
    resp.content = [text_block]
    resp.model = model
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Mock SDK exception helpers for error tests
# ---------------------------------------------------------------------------


def make_openai_auth_error():
    """Create a mock OpenAI authentication error."""
    import openai

    return openai.AuthenticationError(
        "Incorrect API key provided",
        response=MagicMock(status_code=401),
        body=None,
    )


def make_openai_rate_limit_error():
    """Create a mock OpenAI rate-limit error."""
    import openai

    return openai.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )


def make_openai_timeout_error():
    """Create a mock OpenAI timeout error."""
    import openai

    return openai.APITimeoutError("Request timed out")


def make_anthropic_auth_error():
    """Create a mock Anthropic authentication error."""
    from anthropic import AuthenticationError as AnthAuthError

    return AnthAuthError(
        "Invalid API key",
        response=MagicMock(status_code=401),
        body=None,
    )


def make_anthropic_rate_limit_error():
    """Create a mock Anthropic rate-limit error."""
    from anthropic import RateLimitError as AnthRateLimitError

    return AnthRateLimitError(
        "Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )


def make_anthropic_timeout_error():
    """Create a mock Anthropic timeout error."""
    import anthropic

    return anthropic.APITimeoutError("Request timed out")
