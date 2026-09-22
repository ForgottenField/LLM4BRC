"""DeepSeek provider implementation.

DeepSeek offers an OpenAI-compatible API, so this provider uses the ``openai``
SDK with a custom ``base_url`` pointing to ``https://api.deepseek.com``.
"""

from openai import OpenAI

from llm_client.base import LLMProvider, LLMRequest, LLMResponse, LLMConfig
from llm_client.errors import (
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    LLMProviderError,
)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(LLMProvider):
    """LLM provider for DeepSeek models."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or _DEEPSEEK_BASE_URL,
            timeout=config.default_timeout,
        )

    def send(self, request: LLMRequest) -> LLMResponse:
        import time

        start = time.monotonic()
        model = request.model or self._config.default_model or "deepseek-chat"
        max_tokens = request.max_tokens or self._config.default_max_tokens
        timeout = request.timeout or self._config.default_timeout

        # Build messages list: use explicit messages when provided, otherwise
        # wrap the single prompt string as a user message.
        messages = request.messages
        if not messages:
            messages = [{"role": "user", "content": request.prompt}]

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            self._raise_from_sdk_error(exc)

        elapsed = time.monotonic() - start
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider="deepseek",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            }
            if response.usage
            else None,
            response_time=elapsed,
            finish_reason=choice.finish_reason,
        )

    @staticmethod
    def _raise_from_sdk_error(exc: Exception) -> None:
        """Map SDK exceptions to custom error hierarchy."""
        import openai

        if isinstance(exc, openai.AuthenticationError):
            raise AuthenticationError(str(exc)) from exc
        if isinstance(exc, openai.RateLimitError):
            raise RateLimitError(str(exc)) from exc
        if isinstance(exc, openai.APITimeoutError):
            raise TimeoutError(str(exc)) from exc
        if isinstance(exc, openai.APIStatusError):
            raise LLMProviderError(str(exc)) from exc
        raise LLMProviderError(str(exc)) from exc
