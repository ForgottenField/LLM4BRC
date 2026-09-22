"""Provider factory — creates LLMProvider instances from config."""

from llm_client.base import LLMProvider, LLMConfig
from llm_client.openai_provider import OpenAIProvider
from llm_client.anthropic_provider import ClaudeProvider
from llm_client.deepseek_provider import DeepSeekProvider


class ProviderFactory:
    """Creates the appropriate :class:`LLMProvider` from an :class:`LLMConfig`.

    Usage::

        config = LLMConfig(provider_name="gpt", api_key="sk-...")
        provider = ProviderFactory.create(config)
        response = provider.send(LLMRequest(prompt="Hello"))
    """

    _PROVIDER_MAP: dict[str, type[LLMProvider]] = {
        "claude": ClaudeProvider,
        "gpt": OpenAIProvider,
        "deepseek": DeepSeekProvider,
    }

    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        """Create and return a provider matching *config.provider_name*."""
        if not config.api_key:
            raise ValueError("api_key is required")
        provider_cls = cls._PROVIDER_MAP.get(config.provider_name)
        if provider_cls is None:
            raise ValueError(
                f"Unknown provider: '{config.provider_name}'. "
                f"Available: {list(cls._PROVIDER_MAP)}"
            )
        return provider_cls(config)

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        """Register a new provider class under *name*."""
        cls._PROVIDER_MAP[name] = provider_cls
