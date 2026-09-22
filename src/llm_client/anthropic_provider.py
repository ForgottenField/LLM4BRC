"""Anthropic (Claude) provider implementation."""

from anthropic import Anthropic
from anthropic import (
    AuthenticationError as _AnthropicAuthError,
    RateLimitError as _AnthropicRateLimitError,
)

from llm_client.base import LLMProvider, LLMRequest, LLMResponse, LLMConfig
from llm_client.errors import (
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    LLMProviderError,
)


class ClaudeProvider(LLMProvider):
    """LLM provider for Anthropic Claude models."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = Anthropic(
            api_key=config.api_key,
            timeout=config.default_timeout,
        )

    def send(self, request: LLMRequest) -> LLMResponse:
        import time

        start = time.monotonic()
        model = request.model or self._config.default_model or "claude-sonnet-4-20250514"
        max_tokens = request.max_tokens or self._config.default_max_tokens
        timeout = request.timeout or self._config.default_timeout

        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                timeout=timeout,
            )
        except Exception as exc:
            self._raise_from_sdk_error(exc)

        elapsed = time.monotonic() - start
        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
        return LLMResponse(
            content=content,
            model=response.model,
            provider="claude",
            usage=usage,
            response_time=elapsed,
            finish_reason=response.stop_reason,
        )

    @staticmethod
    def _raise_from_sdk_error(exc: Exception) -> None:
        """Map Anthropic SDK exceptions to custom error hierarchy."""
        if isinstance(exc, _AnthropicAuthError):
            raise AuthenticationError(str(exc)) from exc
        if isinstance(exc, _AnthropicRateLimitError):
            raise RateLimitError(str(exc)) from exc
        import anthropic

        if isinstance(exc, anthropic.APITimeoutError):
            raise TimeoutError(str(exc)) from exc
        if isinstance(exc, anthropic.APIStatusError):
            raise LLMProviderError(str(exc)) from exc
        raise LLMProviderError(str(exc)) from exc
