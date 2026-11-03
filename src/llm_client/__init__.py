"""Unified LLM client library for POC generation.

Provider SDKs are loaded on first use.  Importing this package only loads the
provider-neutral request/response types, so a DeepSeek-only installation does
not require the Anthropic SDK.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from llm_client.base import LLMRequest, LLMResponse, LLMConfig, LLMProvider
from llm_client.factory import ProviderFactory

_LAZY_EXPORTS = {
    "OpenAIProvider": ("llm_client.openai_provider", "OpenAIProvider"),
    "ClaudeProvider": ("llm_client.anthropic_provider", "ClaudeProvider"),
    "DeepSeekProvider": ("llm_client.deepseek_provider", "DeepSeekProvider"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = target
    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value

__all__ = [
    "LLMRequest",
    "LLMResponse",
    "LLMConfig",
    "LLMProvider",
    "ProviderFactory",
    "OpenAIProvider",
    "ClaudeProvider",
    "DeepSeekProvider",
]
