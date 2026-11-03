"""Built-in provider defaults shared by the CLI and provider factory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    """Environment and model defaults for one built-in provider."""

    name: str
    api_key_env: str
    default_model: str


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec("deepseek", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "gpt": ProviderSpec("gpt", "OPENAI_API_KEY", "gpt-4o"),
    "claude": ProviderSpec("claude", "ANTHROPIC_API_KEY", "claude-sonnet-4-20250514"),
}


def provider_spec(name: str) -> ProviderSpec:
    """Return the built-in spec for *name*."""
    try:
        return PROVIDER_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown provider {name!r}; choose from {', '.join(PROVIDER_SPECS)}"
        ) from exc
