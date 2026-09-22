"""Base abstractions for the LLM client."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMRequest:
    """A prompt to send to an LLM provider.

    Use *prompt* for simple single-user-message calls.  For conversations
    that need system instructions, few-shot examples, or multi-turn
    dialogue, pass a list of role/dict entries via *messages* instead::

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        req = LLMRequest(messages=messages, model="gpt-4o")

    When *messages* is provided it takes precedence over *prompt*.
    """

    prompt: str = ""
    messages: Optional[list[dict[str, str]]] = None
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 60.0


@dataclass
class LLMResponse:
    """The result of an LLM API call."""

    content: str
    model: str
    provider: str
    usage: Optional[dict[str, int]] = None
    response_time: float = 0.0
    # Provider stop reason: ``"stop"`` for a complete answer, ``"length"`` (OpenAI /
    # DeepSeek) or ``"max_tokens"`` (Anthropic) when the completion was CUT OFF at
    # the requested budget.  A truncated reply is not malformed JSON produced by
    # the model — it is an incomplete document, so callers must be able to see
    # this before blaming their parser.
    finish_reason: Optional[str] = None


@dataclass
class LLMConfig:
    """Per-provider configuration."""

    provider_name: str
    api_key: str
    base_url: str = ""
    default_model: str = ""
    default_max_tokens: int = 1024
    default_timeout: float = 60.0


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Each concrete provider implements :meth:`send` to communicate with a
    specific LLM API.  Providers MUST raise the custom exceptions defined
    in :mod:`llm_client.errors` rather than raw SDK errors.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def config(self) -> LLMConfig:
        return self._config

    @abstractmethod
    def send(self, request: LLMRequest) -> LLMResponse:
        """Send a prompt and return the response."""
        ...
