"""Provider factory — creates LLMProvider instances from config."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from llm_client.base import LLMProvider, LLMConfig

if TYPE_CHECKING:
    pass


class ProviderFactory:
    """Creates the appropriate :class:`LLMProvider` from an :class:`LLMConfig`.

    Usage::

        config = LLMConfig(provider_name="gpt", api_key="sk-...")
        provider = ProviderFactory.create(config)
        response = provider.send(LLMRequest(prompt="Hello"))
    """

    _PROVIDER_MODULES: dict[str, tuple[str, str]] = {
        "claude": ("llm_client.anthropic_provider", "ClaudeProvider"),
        "gpt": ("llm_client.openai_provider", "OpenAIProvider"),
        "deepseek": ("llm_client.deepseek_provider", "DeepSeekProvider"),
    }
    _PROVIDER_MAP: dict[str, type[LLMProvider]] = {}

    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        """Create and return a provider matching *config.provider_name*."""
        if not config.api_key:
            raise ValueError("api_key is required")
        provider_cls = cls._resolve_provider(config.provider_name)
        if provider_cls is None:
            raise ValueError(
                f"Unknown provider: '{config.provider_name}'. "
                f"Available: {sorted(set(cls._PROVIDER_MODULES) | set(cls._PROVIDER_MAP))}"
            )
        return provider_cls(config)

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        """Register a new provider class under *name*."""
        cls._PROVIDER_MAP[name] = provider_cls

    @classmethod
    def _resolve_provider(cls, name: str) -> type[LLMProvider] | None:
        """Resolve and cache a built-in provider on first use."""
        registered = cls._PROVIDER_MAP.get(name)
        if registered is not None:
            return registered
        target = cls._PROVIDER_MODULES.get(name)
        if target is None:
            return None
        module_name, class_name = target
        provider_cls = getattr(import_module(module_name), class_name)
        cls._PROVIDER_MAP[name] = provider_cls
        return provider_cls
